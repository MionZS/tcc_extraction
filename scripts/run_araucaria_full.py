"""Pipeline completo para extração de TODOS os alimentadores de Araucária.

Extrai o CIS completo de Araucária, cruza com o GEO (todos os alimentadores),
extrai o MDM por lotes com monitoramento em tempo real de filas e workers paralelos,
consolida o dataset particionado e suporta sincronização inteligente (auto-backfill
da data mais recente até ontem + preenchimento de lacunas históricas).

Usage:
    # Sincronização inteligente (auto-backfill da última data até ontem + gaps)
    uv run scripts/run_araucaria_full.py --auto-backfill

    # Período customizado
    uv run scripts/run_araucaria_full.py --start-date 2026-08-01 --end-date 2026-10-18

    # Dia específico
    uv run scripts/run_araucaria_full.py --days-back 1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
import queue
import re
import shutil
import sys
import threading
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.normalize import (
    normalize_and_sink_mdm_batches,
    normalize_electrical_hierarchy,
    normalize_meter_installation_history,
    normalize_uc_context,
)
from src.datasets.partitioning import (
    ensure_partition_dirs,
    measurements_partition_path,
)
from src.datasets.writers import (
    write_context_table,
    write_features,
    write_training_dataset,
    write_window_features,
)
from src.db import create_engine
from src.features.meter_day import build_meter_day_features
from src.features.uc_window import build_uc_window_features
from src.run_manifest import create_manifest

console = Console()

QUERIES_DIR = ROOT_DIR / "queries"
CIS_SQL = QUERIES_DIR / "cis_araucaria_ml_extract_lightweight_alt.sql"
GEO_ALL_FEEDERS_SQL = QUERIES_DIR / "geo_araucaria_all_feeders.sql"
GEO_FEEDER_SQL = QUERIES_DIR / "geo_feeder_direct.sql"
MDM_SQL = QUERIES_DIR / "mdm_coluna.sql"

# Código IBGE do Município de Araucária - PR
ARAUCARIA_COD_MUN = 4101804


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim][{now}][/dim] {msg}")


def _date_token(d: date) -> str:
    return d.strftime("%Y%m%d")


def _normalize_key(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value).strip())
    return digits.lstrip("0") if digits else None


def _materialize_scalar(value: Any) -> Any:
    if hasattr(value, "read"):
        try:
            value = value.read()
        except Exception:
            return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin1", errors="replace")
    return value


def _rows_to_frame(rows: Sequence[Any], columns: Sequence[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={str(c).strip().upper(): pl.String for c in columns})

    norm_columns = [str(c).strip().upper() for c in columns]
    materialized_rows = [
        tuple(_materialize_scalar(v) for v in row)
        for row in rows
    ]
    return pl.DataFrame(
        materialized_rows,
        schema=norm_columns,
        orient="row",
        infer_schema_length=None,
        strict=False,
    )


def _build_oracle_string_list(connection: Connection, values: Sequence[str]):
    raw_conn = connection.connection
    driver_connection = raw_conn.driver_connection
    object_type = driver_connection.gettype("SYS.ODCIVARCHAR2LIST")
    bind_object = object_type.newobject()
    bind_object.extend(list(values))
    return bind_object


def _chunked(values: Sequence[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def find_datalake_date_coverage(output_dir: Path) -> tuple[set[date], date | None, date | None]:
    """Scan existing instantaneous partitions to discover all valid extracted dates (Parquet or CSV)."""
    existing_dates: set[date] = set()
    instant_base = output_dir / "measurements" / "ami_instantaneous"
    if instant_base.exists():
        for pat in ("report_year=*/report_month=*/report_day=*/data.parquet", "report_year=*/report_month=*/report_day=*/data.csv"):
            for p in instant_base.glob(pat):
                if p.stat().st_size > 1000:
                    parts = p.parts
                    try:
                        y = int(next(part.split("=")[1] for part in parts if part.startswith("report_year=")))
                        m = int(next(part.split("=")[1] for part in parts if part.startswith("report_month=")))
                        d = int(next(part.split("=")[1] for part in parts if part.startswith("report_day=")))
                        existing_dates.add(date(y, m, d))
                    except Exception:
                        continue

    min_date = min(existing_dates) if existing_dates else None
    max_date = max(existing_dates) if existing_dates else None
    return existing_dates, min_date, max_date


def determine_smart_extraction_dates(
    output_dir: Path,
    default_start_date: date = date(2026, 8, 1),
    fill_gaps: bool = True,
) -> tuple[list[date], str]:
    """Determine dates to extract: from (max_extracted + 1) to yesterday, plus internal gaps."""
    yesterday = date.today() - timedelta(days=1)
    existing_dates, min_date, max_date = find_datalake_date_coverage(output_dir)

    if not existing_dates:
        all_days = [default_start_date + timedelta(days=i) for i in range((yesterday - default_start_date).days + 1)]
        return all_days, f"Inicialização: Data Lake sem partições. Extraindo histórico completo ({default_start_date} → {yesterday}, {len(all_days)} dias)."

    target_dates: set[date] = set()

    # 1. Tail incremental: from max_date + 1 up to yesterday
    tail_start = max_date + timedelta(days=1)
    if tail_start <= yesterday:
        tail_days = [tail_start + timedelta(days=i) for i in range((yesterday - tail_start).days + 1)]
        target_dates.update(tail_days)

    # 2. Historical gaps: check for missing days within range
    if fill_gaps:
        start_bound = min(default_start_date, min_date) if min_date else default_start_date
        full_range = [start_bound + timedelta(days=i) for i in range((yesterday - start_bound).days + 1)]
        gaps = [d for d in full_range if d not in existing_dates]
        target_dates.update(gaps)

    sorted_dates = sorted(list(target_dates))
    if not sorted_dates:
        return [], f"Data Lake 100% atualizado e contínuo até ontem ({yesterday.isoformat()}). Nenhuma extração necessária."

    desc = (
        f"Sincronização inteligente: {len(sorted_dates)} dia(s) a processar "
        f"(Mais recente no lake: {max_date.isoformat()} | Alvo: {yesterday.isoformat()})."
    )
    return sorted_dates, desc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de extração completo para todos os alimentadores de Araucária."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Dias retroativos a partir de hoje (padrão: se omitido, usa auto-backfill)",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Data inicial (AAAA-MM-DD) para extração em lote de período contínuo (ex: 2026-08-01)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Data final (AAAA-MM-DD) para extração em lote. Padrão: ontem",
    )
    parser.add_argument(
        "--auto-backfill",
        action="store_true",
        default=False,
        help="Sincronização inteligente: extrai da data mais recente até ontem e cobre lacunas históricas.",
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        default=True,
        help="Preenche lacunas históricas dentro do período (padrão: True).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="NIOs por lote de extração do MDM (padrão: 500)",
    )
    parser.add_argument(
        "--no-cis-cache",
        action="store_true",
        help="Força re-extração do CIS mesmo que o arquivo do dia já exista",
    )
    parser.add_argument(
        "--feeder-name",
        type=str,
        default=None,
        help="Opcional: filtrar apenas um alimentador específico (ex: Fonte_Nova)",
    )
    parser.add_argument(
        "--feeder-geo-id",
        type=int,
        default=None,
        help="Opcional: ID GEO do alimentador específico. Requerido se --feeder-name for fornecido.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Número de workers paralelos para extração e normalização do MDM (padrão: 5)",
    )
    parser.add_argument(
        "--output-format",
        choices=["both", "csv", "parquet"],
        default="both",
        help="Formato de armazenamento dos dados: 'both' (Parquet + CSV), 'csv' ou 'parquet' (padrão: both)",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        default=False,
        help="Executa o treinamento do modelo após a geração do dataset",
    )
    return parser.parse_args()


def run_pipeline(
    start_date: date | None = None,
    end_date: date | None = None,
    days_back: int | None = None,
    auto_backfill: bool = False,
    fill_gaps: bool = True,
    batch_size: int = 500,
    no_cis_cache: bool = False,
    feeder_name: str | None = None,
    feeder_geo_id: int | None = None,
    max_workers: int = 5,
    output_format: str = "both",
    train_model: bool = False,
) -> int:
    """Execute the full Araucária pipeline with multi-worker queue visualization and smart backfill."""
    today = date.today()
    output_dir = ROOT_DIR / "output"
    write_csv = output_format in ("both", "csv")
    write_parquet = output_format in ("both", "parquet")

    # Determine dates to process
    if start_date is not None:
        if end_date is None:
            end_date = today - timedelta(days=1)
        if start_date > end_date:
            raise ValueError(f"start_date ({start_date}) não pode ser posterior a end_date ({end_date})")
        days_to_process = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        report_day = end_date
    elif days_back is not None:
        report_day = today - timedelta(days=days_back)
        days_to_process = [report_day]
    else:
        # Default / auto-backfill mode: smart sync from latest datalake date to yesterday
        days_to_process, sync_msg = determine_smart_extraction_dates(output_dir, fill_gaps=fill_gaps)
        console.print(Panel(sync_msg, title="Sincronização Inteligente do Data Lake", border_style="cyan"))
        if not days_to_process:
            return 0
        report_day = max(days_to_process)

    token = _date_token(report_day)
    scope_label = f"Alimentador {feeder_name}" if feeder_name else "Araucária COMPLETA"

    console.print("\n" + "=" * 65)
    console.print(f"[bold cyan]PIPELINE DE EXTRAÇÃO & NORMALIZAÇÃO: {scope_label}[/bold cyan]")
    if len(days_to_process) > 1:
        console.print(f"Período: [bold white]{days_to_process[0].isoformat()} → {days_to_process[-1].isoformat()}[/bold white] ({len(days_to_process)} dias na fila)")
    else:
        console.print(f"Dia de Referência: [bold white]{report_day.isoformat()}[/bold white]")
    console.print("=" * 65 + "\n")

    manifest = create_manifest(report_day, sample_size=0)

    # ── Step 1: CIS Population ────────────────────────────────────────────────
    raw_cis_dir = output_dir / "raw" / "CIS"
    raw_cis_dir.mkdir(parents=True, exist_ok=True)
    cis_file = raw_cis_dir / f"araucaria_cis_{token}.csv"

    if cis_file.exists() and not no_cis_cache:
        _log(f"[*] Reutilizando CIS cached: {cis_file.name}")
        cis_df = pl.read_csv(cis_file, separator=";", infer_schema_length=0)
    else:
        _log("[*] Extraindo população CIS do Oracle...")
        engine = create_engine("cis")
        sql = CIS_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            cols = [str(c).upper() for c in result.keys()]
            cis_df = _rows_to_frame(rows, cols)
        engine.dispose()
        cis_df.write_csv(cis_file, separator=";")
        _log(f"[+] CIS extraído: {cis_df.height:,} linhas.")

    manifest.record_step("cis_extract", rows=cis_df.height, output=str(cis_file))

    # ── Step 2: GEO Population ────────────────────────────────────────────────
    raw_geo_dir = output_dir / "raw" / "GEO"
    raw_geo_dir.mkdir(parents=True, exist_ok=True)

    if feeder_name and feeder_geo_id:
        raw_geo_file_label = f"feeder_{feeder_name}"
        geo_file = raw_geo_dir / f"geo_{raw_geo_file_label}_{token}.parquet"
        geo_sql_text = GEO_FEEDER_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
        geo_params = {"ALIMENTADOR_ID": feeder_geo_id}
        geo_label = f"Alimentador {feeder_name} (ID {feeder_geo_id})"
    else:
        raw_geo_file_label = "all_feeders"
        geo_file = raw_geo_dir / f"geo_{raw_geo_file_label}_{token}.parquet"
        geo_sql_text = GEO_ALL_FEEDERS_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
        geo_params = {"MUNICIPIO_COD": ARAUCARIA_COD_MUN, "MUNICIPIO": ARAUCARIA_COD_MUN}
        geo_label = "Araucária Todos os Alimentadores"

    if geo_file.exists() and not no_cis_cache:
        _log(f"[*] Reutilizando GEO cached: {geo_file.name}")
        geo_df = pl.read_parquet(geo_file)
    else:
        _log(f"[*] Extraindo GEO ({geo_label}) do Oracle...")
        engine_geo = create_engine("geo")
        with engine_geo.connect() as conn:
            result = conn.execute(text(geo_sql_text), geo_params)
            rows = result.fetchall()
            cols = [str(c).upper() for c in result.keys()]
            geo_df = _rows_to_frame(rows, cols)
        engine_geo.dispose()
        geo_df.write_parquet(geo_file, compression="zstd")
        _log(f"[+] GEO extraído: {geo_df.height:,} linhas.")

    manifest.record_step("geo_extract", rows=geo_df.height, output=str(geo_file))

    # ── Step 3: Join CIS × GEO ────────────────────────────────────────────────
    _log("[*] Cruzando CIS × GEO...")

    # Normalize CIS columns
    cis_cols = {c.upper(): c for c in cis_df.columns}
    cis_uc_col = cis_cols.get("UC", "UC")
    cis_nio_col = cis_cols.get("NIO") or cis_cols.get("NUM_MEDIDOR", "NIO")
    cis_posto_col = cis_cols.get("NUM_POSTO_OPERACIONAL") or cis_cols.get("POSTO_OPERACIONAL")

    cis_exprs = [
        pl.col(cis_uc_col).cast(pl.String, strict=False).alias("UC_CLEAN"),
        pl.col(cis_nio_col).map_elements(_normalize_key, return_dtype=pl.String).alias("NIO_CLEAN"),
    ]
    if cis_posto_col and cis_posto_col in cis_df.columns:
        cis_exprs.append(pl.col(cis_posto_col).cast(pl.String, strict=False).alias("POSTO_OPERACIONAL"))
    else:
        cis_exprs.append(pl.lit(None, dtype=pl.String).alias("POSTO_OPERACIONAL"))

    cis_clean = cis_df.with_columns(cis_exprs)

    # Normalize GEO columns
    geo_cols = {c.upper(): c for c in geo_df.columns}
    geo_uc_col = geo_cols.get("UC", "UC")
    geo_med_col = geo_cols.get("MEDIDOR") or geo_cols.get("NIO")
    geo_posto_col = geo_cols.get("POSTO_OPERACIONAL", "POSTO_OPERACIONAL")

    geo_exprs = [
        pl.col(geo_uc_col).cast(pl.String, strict=False).alias("UC_CLEAN"),
    ]
    if geo_med_col and geo_med_col in geo_df.columns:
        geo_exprs.append(pl.col(geo_med_col).map_elements(_normalize_key, return_dtype=pl.String).alias("NIO_CLEAN"))
    else:
        geo_exprs.append(pl.lit(None, dtype=pl.String).alias("NIO_CLEAN"))

    if geo_posto_col and geo_posto_col in geo_df.columns:
        geo_exprs.append(pl.col(geo_posto_col).cast(pl.String, strict=False).alias("POSTO_OPERACIONAL"))
    else:
        geo_exprs.append(pl.lit(None, dtype=pl.String).alias("POSTO_OPERACIONAL"))

    geo_clean = geo_df.with_columns(geo_exprs)

    # Priority 1: Join on UC
    geo_by_uc = geo_clean.filter(pl.col("UC_CLEAN").is_not_null()).unique(subset=["UC_CLEAN"])
    geo_attrs = [c for c in geo_by_uc.columns if c not in ("UC_CLEAN", "NIO_CLEAN", "POSTO_OPERACIONAL", "UC")]

    joined_uc = cis_clean.join(
        geo_by_uc.select(["UC_CLEAN"] + geo_attrs),
        on="UC_CLEAN",
        how="inner",
    )

    remaining_cis = cis_clean.join(joined_uc.select("UC_CLEAN"), on="UC_CLEAN", how="anti")

    # Priority 2: Join on Posto Operacional (se disponível)
    if "POSTO_OPERACIONAL" in geo_clean.columns and remaining_cis.filter(pl.col("POSTO_OPERACIONAL").is_not_null()).height > 0:
        geo_by_posto = geo_clean.filter(pl.col("POSTO_OPERACIONAL").is_not_null()).unique(subset=["POSTO_OPERACIONAL"])
        joined_posto = remaining_cis.join(
            geo_by_posto.select(["POSTO_OPERACIONAL"] + geo_attrs),
            on="POSTO_OPERACIONAL",
            how="inner",
        )
        remaining_cis = remaining_cis.join(joined_posto.select("UC_CLEAN"), on="UC_CLEAN", how="anti")
    else:
        joined_posto = pl.DataFrame()

    all_frames = [f for f in (joined_uc, joined_posto, remaining_cis) if not f.is_empty()]
    population_df = pl.concat(all_frames, how="diagonal").unique(subset=["UC_CLEAN", "NIO_CLEAN"])

    # Ensure NUM_MEDIDOR and ALIMENTADOR exist for downstream steps
    if "NUM_MEDIDOR" not in population_df.columns:
        population_df = population_df.with_columns(pl.col("NIO_CLEAN").alias("NUM_MEDIDOR"))
    if "ALIMENTADOR" not in population_df.columns:
        population_df = population_df.with_columns(pl.lit("DESCONHECIDO").alias("ALIMENTADOR"))

    _log(f"[+] População resultante: {population_df.height:,} medidores vinculados a alimentadores.")

    population_nios = [
        str(n).strip()
        for n in population_df.get_column("NUM_MEDIDOR").drop_nulls().to_list()
        if str(n).strip() and str(n).strip().lower() != "none"
    ]
    manifest.record_step("join_population", rows=len(population_nios))

    # ── Step 4: Normalização do Contexto ─────────────────────────────────────
    _log("[*] Normalizando tabelas de contexto...")
    uc_context = normalize_uc_context(population_df, run_id=manifest.run_id)
    write_context_table(uc_context, "uc_context", base_dir=output_dir, write_csv=write_csv)

    meter_history = normalize_meter_installation_history(population_df, run_id=manifest.run_id)
    write_context_table(meter_history, "meter_installation_history", base_dir=output_dir, write_csv=write_csv)

    hierarchy = normalize_electrical_hierarchy(population_df, run_id=manifest.run_id)
    write_context_table(hierarchy, "electrical_hierarchy", base_dir=output_dir, write_csv=write_csv)

    # ── Step 5: MDM Extraction & Normalization per day with Multi-Worker Progress
    raw_orca_dir = output_dir / "raw" / "ORCA" / raw_geo_file_label
    engine_orca = create_engine("orca")
    mdm_sql_text = MDM_SQL.read_text(encoding="utf-8").strip().rstrip(";/")

    # Parse installation and removal dates for temporal filtering
    has_inst_col = "DATA_INSTALACAO_MEDIDOR" in population_df.columns
    has_ret_col = "DATA_RETIRADA_MEDIDOR" in population_df.columns
    pop_temporal = population_df
    if has_inst_col:
        pop_temporal = pop_temporal.with_columns(
            pl.col("DATA_INSTALACAO_MEDIDOR").cast(pl.Date, strict=False).alias("_DTA_INST_DT")
        )
    if has_ret_col:
        pop_temporal = pop_temporal.with_columns(
            pl.col("DATA_RETIRADA_MEDIDOR").cast(pl.Date, strict=False).alias("_DTA_RET_DT")
        )

    for day_idx, current_day in enumerate(days_to_process, 1):
        curr_token = _date_token(current_day)
        curr_days_back = (today - current_day).days

        # Dynamic temporal filtering:
        # Include meter if:
        #   (DATA_INSTALACAO is null or <= current_day) AND
        #   (DATA_RETIRADA is null or >= current_day)
        # On replacement day (DATA_RETIRADA == current_day), BOTH old and new meters are included!
        if has_inst_col or has_ret_col:
            cond = pl.lit(True)
            if has_inst_col:
                cond = cond & (pl.col("_DTA_INST_DT").is_null() | (pl.col("_DTA_INST_DT") <= current_day))
            if has_ret_col:
                cond = cond & (pl.col("_DTA_RET_DT").is_null() | (pl.col("_DTA_RET_DT") >= current_day))
            day_pop = pop_temporal.filter(cond)
            day_nios = [
                str(n).strip()
                for n in day_pop.get_column("NUM_MEDIDOR").drop_nulls().to_list()
                if str(n).strip() and str(n).strip().lower() != "none"
            ]
        else:
            day_nios = population_nios

        if not day_nios:
            _log(f"[{day_idx}/{len(days_to_process)}] [yellow]⚠[/yellow] Nenhum medidor ativo em {current_day.isoformat()}. Pulando.")
            continue

        chunks = list(enumerate(_chunked(day_nios, batch_size), start=1))
        total_batches = len(chunks)

        interval_dir = measurements_partition_path("ami_interval", current_day, base_dir=output_dir)
        instant_dir = measurements_partition_path("ami_instantaneous", current_day, base_dir=output_dir)
        register_dir = measurements_partition_path("ami_registers", current_day, base_dir=output_dir)

        instant_parquet = instant_dir / "data.parquet"
        instant_csv = instant_dir / "data.csv"
        interval_parquet = interval_dir / "data.parquet"
        register_parquet = register_dir / "data.parquet"

        has_data = (instant_parquet.exists() and instant_parquet.stat().st_size > 1000) or (instant_csv.exists() and instant_csv.stat().st_size > 1000)
        if has_data:
            _log(f"[{day_idx}/{len(days_to_process)}] [bold green]✔[/bold green] Partição para {current_day.isoformat()} já existente. Pulando.")
            continue

        raw_orca_batches_dir = raw_orca_dir / f"batches_{curr_token}"
        raw_orca_batches_dir.mkdir(parents=True, exist_ok=True)
        mdm_file_parquet = raw_orca_dir / f"mdm_{raw_geo_file_label}_{curr_token}.parquet"

        lock = threading.Lock()
        batch_results: list[tuple[int, Path, int]] = []
        total_mdm_rows = 0

        # Worker pool queue for dynamic task-to-worker mapping
        worker_pool = queue.Queue()
        for w in range(max_workers):
            worker_pool.put(w)

        # Worker connection pool for persistent session reuse (zero TCP/TLS handshake overhead per batch)
        worker_conns: dict[int, Connection] = {}
        conn_lock = threading.Lock()

        def _get_worker_conn(w: int) -> Connection:
            with conn_lock:
                conn = worker_conns.get(w)
                if conn is None or conn.closed:
                    conn = engine_orca.connect()
                    worker_conns[w] = conn
                return conn

        # Multi-bar progress display: 1 overall queue bar + 1 bar per worker
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
            refresh_per_second=4,
        ) as progress:
            overall_task = progress.add_task(
                f"[bold cyan]📦 Fila Geral MDM ({current_day.isoformat()} - {len(day_nios):,} medidores ativos/transição)[/]",
                total=total_batches,
                completed=0,
            )

            worker_tasks = {}
            for w in range(max_workers):
                worker_tasks[w] = progress.add_task(
                    f"[dim]Worker #{w + 1:02d}: Aguardando lote na fila...[/]",
                    total=100,
                    completed=0,
                )

            def _extract_single_batch(item: tuple[int, list[str]]) -> tuple[int, Path | None, int]:
                b_idx, b_nios = item
                b_path = raw_orca_batches_dir / f"mdm_batch_{b_idx:05d}.parquet"
                w_id = worker_pool.get()

                try:
                    # 1. Check local batch cache
                    if b_path.exists() and b_path.stat().st_size > 0:
                        try:
                            cached_df = pl.read_parquet(b_path)
                            progress.update(
                                worker_tasks[w_id],
                                description=f"[dim green]⚡ Worker #{w_id + 1:02d}: Lote #{b_idx:04d} ({cached_df.height:,} lins) [Cache][/]",
                                completed=100,
                            )
                            progress.update(overall_task, advance=1)
                            return b_idx, b_path, cached_df.height
                        except Exception:
                            pass

                    # 2. Update status: Querying Oracle MDM
                    progress.update(
                        worker_tasks[w_id],
                        description=f"[bold green]⚡ Worker #{w_id + 1:02d}[/]: Lote #{b_idx:04d} ({len(b_nios)} NIOs) [cyan]Consultando Oracle MDM...[/]",
                        completed=25,
                    )

                    conn = _get_worker_conn(w_id)
                    bind_list = _build_oracle_string_list(conn, b_nios)
                    result = conn.execute(
                        text(mdm_sql_text),
                        {"DAYS_BACK": curr_days_back, "UCS": bind_list},
                    )
                    cols = [str(c).upper() for c in result.keys()]

                    progress.update(
                        worker_tasks[w_id],
                        description=f"[bold green]⚡ Worker #{w_id + 1:02d}[/]: Lote #{b_idx:04d} [yellow]Baixando registros...[/]",
                        completed=60,
                    )
                    b_rows = result.fetchall()

                    if b_rows:
                        progress.update(
                            worker_tasks[w_id],
                            description=f"[bold green]⚡ Worker #{w_id + 1:02d}[/]: Lote #{b_idx:04d} ({len(b_rows):,} lins) [magenta]Gravando Parquet...[/]",
                            completed=85,
                        )
                        b_df = _rows_to_frame(b_rows, cols)
                        b_df.write_parquet(b_path, compression="zstd")
                        n_rows = b_df.height
                        del b_df, b_rows

                        progress.update(
                            worker_tasks[w_id],
                            description=f"[bold green]✔ Worker #{w_id + 1:02d}[/]: Lote #{b_idx:04d} finalizado ({n_rows:,} lins)",
                            completed=100,
                        )
                        progress.update(overall_task, advance=1)
                        return b_idx, b_path, n_rows
                    else:
                        progress.update(
                            worker_tasks[w_id],
                            description=f"[dim]Worker #{w_id + 1:02d}: Lote #{b_idx:04d} (0 linhas)[/]",
                            completed=100,
                        )
                        progress.update(overall_task, advance=1)
                        return b_idx, None, 0

                except Exception as exc:
                    progress.update(
                        worker_tasks[w_id],
                        description=f"[bold red]❌ Worker #{w_id + 1:02d}: Erro no lote #{b_idx:04d}[/]",
                        completed=100,
                    )
                    raise
                finally:
                    progress.update(
                        worker_tasks[w_id],
                        description=f"[dim]Worker #{w_id + 1:02d}: Aguardando na fila...[/]",
                        completed=0,
                    )
                    worker_pool.put(w_id)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_extract_single_batch, chunk) for chunk in chunks]
                for fut in as_completed(futures):
                    b_idx, b_path, n_rows = fut.result()
                    if b_path is not None and n_rows > 0:
                        batch_results.append((b_idx, b_path, n_rows))
                        total_mdm_rows += n_rows

            for w in range(max_workers):
                progress.update(
                    worker_tasks[w],
                    description=f"[bold green]✔ Worker #{w + 1:02d}: Finalizado[/]",
                    completed=100,
                )

        # Release worker connections
        for c in worker_conns.values():
            try:
                c.close()
            except Exception:
                pass
        worker_conns.clear()

        batch_results.sort(key=lambda x: x[0])
        batch_files: list[Path] = [x[1] for x in batch_results]

        if batch_files:
            _log(f"[*] Consolidando {len(batch_files)} lotes via Lazy-Sink...")
            raw_lazy = pl.concat([pl.scan_parquet(str(p)) for p in batch_files], how="vertical_relaxed")
            raw_lazy.sink_parquet(mdm_file_parquet, compression="zstd")
            _log(f"[+] MDM completo {current_day.isoformat()}: {total_mdm_rows:,} linhas → {mdm_file_parquet.name}")
        else:
            _log(f"[-] Nenhuma linha MDM retornada para {current_day.isoformat()}.")
            continue

        manifest.record_step(f"mdm_extract_{curr_token}", rows=total_mdm_rows)

        # Normalization
        _log(f"[*] Normalizando camadas semânticas para {current_day.isoformat()} com {max_workers} workers...")
        ensure_partition_dirs(interval_dir)
        ensure_partition_dirs(instant_dir)
        ensure_partition_dirs(register_dir)

        norm_temp_dir = output_dir / "tmp" / f"norm_{curr_token}"
        total_interval, total_instant, total_register = normalize_and_sink_mdm_batches(
            batch_files,
            report_day=current_day,
            interval_final_parquet=interval_parquet,
            instant_final_parquet=instant_parquet,
            register_final_parquet=register_parquet,
            temp_dir=norm_temp_dir,
            run_id=manifest.run_id,
            write_csv=write_csv,
            compression="zstd",
            max_workers=max_workers,
        )
        shutil.rmtree(norm_temp_dir, ignore_errors=True)
        _log(f"[+] Medições normalizadas ({current_day.isoformat()}): {total_interval:,} intervalos | {total_instant:,} instantâneos | {total_register:,} registradores.")

    engine_orca.dispose()

    # Consolidate daily features and training dataset for latest day in range
    def _read_layer(parquet_p: Path) -> pl.DataFrame:
        if parquet_p.exists() and parquet_p.stat().st_size > 0:
            return pl.read_parquet(parquet_p)
        csv_p = parquet_p.with_suffix(".csv")
        if csv_p.exists() and csv_p.stat().st_size > 0:
            return pl.read_csv(csv_p, separator=";")
        return pl.DataFrame()

    interval_df = _read_layer(interval_parquet)
    instant_df = _read_layer(instant_parquet)
    register_df = _read_layer(register_parquet)

    meter_to_uc: dict[str, str] = {}
    if not meter_history.is_empty():
        for r in meter_history.select(["NIO", "UC"]).iter_rows():
            if r[0] and r[1]:
                meter_to_uc[str(r[0])] = str(r[1])

    # Daily features
    daily_features = build_meter_day_features(
        interval_df,
        instant_df,
        register_df,
        metadata_df=uc_context,
        report_day=report_day,
        meter_to_uc_map=meter_to_uc,
    )
    write_features(daily_features, "v1", report_day, base_dir=output_dir, write_csv=write_csv)

    # Window features
    window_features = build_uc_window_features(
        daily_features,
        meter_history_df=meter_history,
        cutoff_date=report_day,
        window_days=30,
        feature_set_version="v1",
    )
    write_window_features(window_features, "v1", report_day, base_dir=output_dir, write_csv=write_csv)

    # Training Dataset with meta__feeder
    training_df = build_training_dataset(
        window_features,
        uc_context_df=uc_context,
        hierarchy_df=hierarchy,
        split_policy="train",
    )

    dataset_path = write_training_dataset(training_df, "v1", base_dir=output_dir, write_csv=write_csv)
    _log(f"[+] Dataset de treinamento consolidado: {dataset_path}")
    _log(f"    Total de amostras: {training_df.height:,}")

    if "meta__feeder" in training_df.columns:
        f_dist = training_df.group_by("meta__feeder").agg(pl.len().alias("n_ucs")).sort("n_ucs", descending=True)
        with pl.Config(tbl_rows=100, tbl_cols=10):
            _log(f"    Distribuição por alimentador:\n{f_dist}")

    manifest.record_step("build_training_dataset", rows=training_df.height, output=str(dataset_path))
    manifest.finalize()
    m_path = manifest.save()
    _log(f"[+] Manifest salvo: {m_path}")

    if train_model and not training_df.is_empty():
        _log("\n[*] Disparando treinamento do classificador de tensão...")
        from scripts.train_voltage_classifier import main as train_voltage_main
        old_argv = sys.argv
        sys.argv = [
            "train_voltage_classifier.py",
            "--instantaneous-dir", str(output_dir / "measurements" / "ami_instantaneous"),
            "--tag", "production",
            "--description", f"Pipeline batch {report_day.isoformat()}",
        ]
        try:
            train_voltage_main()
        finally:
            sys.argv = old_argv

    console.print("\n" + "=" * 65)
    console.print(f"[bold green]✔ PIPELINE CONCLUÍDO COM SUCESSO: {scope_label}[/bold green]")
    console.print("=" * 65)
    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        days_back=args.days_back,
        auto_backfill=args.auto_backfill,
        fill_gaps=args.fill_gaps,
        batch_size=args.batch_size,
        no_cis_cache=args.no_cis_cache,
        feeder_name=args.feeder_name,
        feeder_geo_id=args.feeder_geo_id,
        max_workers=args.max_workers,
        output_format=args.output_format,
        train_model=args.train_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
