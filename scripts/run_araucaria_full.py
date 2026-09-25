"""Pipeline completo para extração de TODOS os alimentadores de Araucária.

Extrai o CIS completo de Araucária, cruza com o GEO (todos os alimentadores),
extrai o MDM por lotes e consolida um único dataset particionado por ALIMENTADOR.

O dataset resultante contém todas as UCs de Araucária e pode ser filtrado por
alimentador na etapa de treinamento de modelos (via DatasetSplitter).

Opcionalmente, pode filtrar apenas um alimentador específico via --feeder-name.

Flow:
1. CIS: Extração completa de Araucária (com cache diário).
2. GEO: Extração de TODOS os alimentadores via geo_araucaria_all_feeders.sql.
   Ou, se --feeder-geo-id fornecido, extrai apenas aquele alimentador.
3. JOIN CIS × GEO → population completa com ALIMENTADOR como coluna.
4. MDM: Extração em lotes para todos os NIOs resultantes.
5. Normalização em camadas Parquet com lazy-sink.
6. Build do training_dataset.parquet com coluna meta__feeder para particionamento.

Usage:
    # Extrai Araucária inteira (recomendado)
    uv run scripts/run_araucaria_full.py --days-back 1

    # Extrai apenas um alimentador específico
    uv run scripts/run_araucaria_full.py --feeder-name Fonte_Nova --feeder-geo-id 6352460

    # Sem re-extrair o CIS (usa cache do dia)
    uv run scripts/run_araucaria_full.py --days-back 1 --no-cis-cache
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.db import create_engine
from src.datasets.normalize import (
    normalize_uc_context,
    normalize_meter_installation_history,
    normalize_electrical_hierarchy,
    normalize_and_sink_mdm_batches,
)
from src.datasets.partitioning import (
    measurements_partition_path,
    ensure_partition_dirs,
)
from src.datasets.writers import (
    write_context_table,
    write_features,
    write_window_features,
    write_training_dataset,
)
from src.features.meter_day import build_meter_day_features
from src.features.uc_window import build_uc_window_features
from src.datasets.build_training_dataset import build_training_dataset
from src.run_manifest import create_manifest

QUERIES_DIR = ROOT_DIR / "queries"
CIS_SQL = QUERIES_DIR / "cis_araucaria_ml_extract_lightweight_alt.sql"
GEO_ALL_FEEDERS_SQL = QUERIES_DIR / "geo_araucaria_all_feeders.sql"
GEO_FEEDER_SQL = QUERIES_DIR / "geo_feeder_direct.sql"
MDM_SQL = QUERIES_DIR / "mdm_coluna.sql"

# Código IBGE do Município de Araucária - PR
ARAUCARIA_COD_MUN = 4101804


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de extração completo para todos os alimentadores de Araucária."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Dias retroativos a partir de hoje (padrão: 1 = ontem)",
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
        "--train-model",
        action="store_true",
        default=False,
        help="Executa o treinamento do modelo após a geração do dataset",
    )
    return parser.parse_args()


def run_pipeline(
    days_back: int = 1,
    batch_size: int = 500,
    no_cis_cache: bool = False,
    feeder_name: str | None = None,
    feeder_geo_id: int | None = None,
    max_workers: int = 5,
    train_model: bool = False,
) -> int:
    """Execute the full Araucária pipeline. Can be called programmatically from TUI."""
    report_day = date.today() - timedelta(days=days_back)
    token = _date_token(report_day)
    output_dir = ROOT_DIR / "output"

    scope_label = f"Alimentador {feeder_name}" if feeder_name else "Araucária COMPLETA"
    _log("=" * 65)
    _log(f"PIPELINE: {scope_label}")
    _log(f"Dia de Referência: {report_day.isoformat()} (days_back={days_back})")
    _log("=" * 65)

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

    manifest.record_step("cis_extract", rows=cis_df.height)
    cis_df = cis_df.rename({c: c.upper() for c in cis_df.columns})
    cis_df = cis_df.with_columns(
        pl.col("UC").map_elements(_normalize_key, return_dtype=pl.String).alias("UC"),
        pl.col("NIO").map_elements(_normalize_key, return_dtype=pl.String).alias("NIO"),
    )

    # ── Step 2: GEO Extraction ────────────────────────────────────────────────
    engine_geo = create_engine("geo")

    if feeder_name and feeder_geo_id:
        # Single feeder mode
        _log(f"[*] Extraindo GEO para alimentador específico: {feeder_name} (ID {feeder_geo_id})...")
        geo_sql = GEO_FEEDER_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
        with engine_geo.connect() as conn:
            result = conn.execute(text(geo_sql), {"FEEDER_GEO_ID": feeder_geo_id})
            rows = result.fetchall()
            cols = [str(c).upper() for c in result.keys()]
            geo_df = _rows_to_frame(rows, cols)
        # Inject ALIMENTADOR name if not present
        if "ALIMENTADOR" not in geo_df.columns:
            geo_df = geo_df.with_columns(pl.lit(feeder_name).alias("ALIMENTADOR"))
        raw_geo_file_label = f"feeder_{feeder_name}"
    else:
        # Full Araucária mode — all feeders
        _log("[*] Extraindo GEO para TODOS os alimentadores de Araucária...")
        geo_sql = GEO_ALL_FEEDERS_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
        with engine_geo.connect() as conn:
            result = conn.execute(text(geo_sql), {"MUNICIPIO_COD": ARAUCARIA_COD_MUN})
            rows = result.fetchall()
            cols = [str(c).upper() for c in result.keys()]
            geo_df = _rows_to_frame(rows, cols)
        raw_geo_file_label = "araucaria_all_feeders"

    engine_geo.dispose()

    geo_raw_dir = output_dir / "raw" / "GEO"
    geo_raw_dir.mkdir(parents=True, exist_ok=True)
    geo_file = geo_raw_dir / f"geo_{raw_geo_file_label}_{token}.csv"
    geo_df.write_csv(geo_file, separator=";")

    n_feeders = geo_df.get_column("ALIMENTADOR").drop_nulls().n_unique() if "ALIMENTADOR" in geo_df.columns else "?"
    _log(f"[+] GEO extraído: {geo_df.height:,} UCs, {n_feeders} alimentadores distintos.")
    manifest.record_step("geo_extract", rows=geo_df.height)

    if geo_df.is_empty():
        _log("[-] Nenhuma UC encontrada no GEO. Abortando.")
        return 1

    geo_df = geo_df.with_columns(
        pl.col("UC").map_elements(_normalize_key, return_dtype=pl.String).alias("UC")
    )

    # ── Step 3: JOIN CIS × GEO ───────────────────────────────────────────────
    cis_df = cis_df.unique(subset=["UC", "NIO"])
    geo_df = geo_df.unique(subset=["UC"])
    population_df = cis_df.join(geo_df, on="UC", how="inner", suffix="_GEO")
    _log(f"[+] Join CIS × GEO: {population_df.height:,} registros UC/medidor encontrados.")
    manifest.record_step("population_join", rows=population_df.height)

    if "ALIMENTADOR" in population_df.columns:
        feeder_counts = (
            population_df.group_by("ALIMENTADOR")
            .agg(pl.len().alias("N_NIOS"))
            .sort("N_NIOS", descending=True)
        )
        with pl.Config(tbl_rows=100, tbl_cols=10):
            _log(f"[+] Distribuição por alimentador:\n{feeder_counts}")

    population_nios = (
        population_df.get_column("NIO").drop_nulls().unique().to_list()
    )
    _log(f"[+] NIOs únicos para extração MDM: {len(population_nios):,}")

    # ── Step 4: MDM Extraction (Parallel Lazy-Sink batch) ───────────────────────
    raw_orca_dir = output_dir / "raw" / "ORCA" / raw_geo_file_label
    raw_orca_batches_dir = raw_orca_dir / f"batches_{token}"
    raw_orca_batches_dir.mkdir(parents=True, exist_ok=True)
    mdm_file_parquet = raw_orca_dir / f"mdm_{raw_geo_file_label}_{token}.parquet"

    chunks = list(enumerate(_chunked(population_nios, batch_size), start=1))
    total_batches = len(chunks)
    _log(
        f"[*] Extraindo telemetria MDM para {len(population_nios):,} NIOs "
        f"em {total_batches} lotes com {max_workers} workers paralelos..."
    )

    engine_orca = create_engine("orca")
    mdm_sql_text = MDM_SQL.read_text(encoding="utf-8").strip().rstrip(";/")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    completed_count = 0
    batch_results: list[tuple[int, Path, int]] = []
    total_mdm_rows = 0

    def _extract_single_batch(item: tuple[int, list[str]]) -> tuple[int, Path | None, int]:
        nonlocal completed_count
        b_idx, b_nios = item
        b_path = raw_orca_batches_dir / f"mdm_batch_{b_idx:05d}.parquet"

        # Check if valid cached batch already exists (fast resume)
        if b_path.exists() and b_path.stat().st_size > 0:
            try:
                cached_df = pl.read_parquet(b_path)
                with lock:
                    completed_count += 1
                    _log(f"    [{completed_count:03d}/{total_batches}] Batch {b_idx:05d} cached: {cached_df.height:,} linhas.")
                return b_idx, b_path, cached_df.height
            except Exception:
                pass

        try:
            with engine_orca.connect() as conn:
                bind_list = _build_oracle_string_list(conn, b_nios)
                result = conn.execute(
                    text(mdm_sql_text),
                    {"DAYS_BACK": days_back, "UCS": bind_list},
                )
                cols = [str(c).upper() for c in result.keys()]
                b_rows = result.fetchall()
                if b_rows:
                    b_df = _rows_to_frame(b_rows, cols)
                    b_df.write_parquet(b_path, compression="zstd")
                    n_rows = b_df.height
                    del b_df, b_rows
                    with lock:
                        completed_count += 1
                        _log(f"    [{completed_count:03d}/{total_batches}] Batch {b_idx:05d} extraído: {n_rows:,} linhas ({len(b_nios)} NIOs).")
                    return b_idx, b_path, n_rows
        except Exception as exc:
            with lock:
                _log(f"    [-] Erro no lote {b_idx:05d}: {exc}")
            raise

        with lock:
            completed_count += 1
            _log(f"    [{completed_count:03d}/{total_batches}] Batch {b_idx:05d}: 0 linhas.")
        return b_idx, None, 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_single_batch, chunk) for chunk in chunks]
        for fut in as_completed(futures):
            b_idx, b_path, n_rows = fut.result()
            if b_path is not None and n_rows > 0:
                batch_results.append((b_idx, b_path, n_rows))
                total_mdm_rows += n_rows

    engine_orca.dispose()

    # Sort batches by index to maintain deterministic ordering
    batch_results.sort(key=lambda x: x[0])
    batch_files: list[Path] = [x[1] for x in batch_results]

    if batch_files:
        _log(f"[*] Consolidando {len(batch_files)} lotes via Lazy-Sink...")
        raw_lazy = pl.concat([pl.scan_parquet(str(p)) for p in batch_files], how="vertical_relaxed")
        raw_lazy.sink_parquet(mdm_file_parquet, compression="zstd")
        _log(f"[+] MDM completo: {total_mdm_rows:,} linhas → {mdm_file_parquet.name}")
    else:
        _log("[-] Nenhuma linha MDM retornada.")

    manifest.record_step("mdm_extract", rows=total_mdm_rows)

    # ── Step 5: Semantic Layers Normalization ─────────────────────────────────
    _log(f"[*] Normalizando camadas semânticas do datalake com {max_workers} workers...")

    uc_context = normalize_uc_context(population_df, run_id=manifest.run_id)
    write_context_table(uc_context, "uc_context", base_dir=output_dir, write_csv=False)

    meter_history = normalize_meter_installation_history(population_df, run_id=manifest.run_id)
    write_context_table(meter_history, "meter_installation_history", base_dir=output_dir, write_csv=False)

    hierarchy = normalize_electrical_hierarchy(population_df, run_id=manifest.run_id)
    write_context_table(hierarchy, "electrical_hierarchy", base_dir=output_dir, write_csv=False)

    interval_dir = measurements_partition_path("ami_interval", report_day, base_dir=output_dir)
    ensure_partition_dirs(interval_dir)
    interval_parquet = interval_dir / "data.parquet"

    instant_dir = measurements_partition_path("ami_instantaneous", report_day, base_dir=output_dir)
    ensure_partition_dirs(instant_dir)
    instant_parquet = instant_dir / "data.parquet"

    register_dir = measurements_partition_path("ami_registers", report_day, base_dir=output_dir)
    ensure_partition_dirs(register_dir)
    register_parquet = register_dir / "data.parquet"

    norm_temp_dir = output_dir / "tmp" / f"norm_{token}"
    total_interval, total_instant, total_register = normalize_and_sink_mdm_batches(
        batch_files,
        report_day=report_day,
        interval_final_parquet=interval_parquet,
        instant_final_parquet=instant_parquet,
        register_final_parquet=register_parquet,
        temp_dir=norm_temp_dir,
        run_id=manifest.run_id,
        write_csv=False,
        compression="zstd",
        max_workers=max_workers,
    )
    shutil.rmtree(norm_temp_dir, ignore_errors=True)
    _log(f"[+] Medições normalizadas: {total_interval:,} intervalos | {total_instant:,} instantâneos | {total_register:,} registradores.")

    interval_df = pl.read_parquet(interval_parquet) if interval_parquet.exists() else pl.DataFrame()
    instant_df = pl.read_parquet(instant_parquet) if instant_parquet.exists() else pl.DataFrame()
    register_df = pl.read_parquet(register_parquet) if register_parquet.exists() else pl.DataFrame()

    meter_to_uc: dict[str, str] = {}
    if not meter_history.is_empty():
        for r in meter_history.select(["NIO", "UC"]).iter_rows():
            if r[0] and r[1]:
                meter_to_uc[str(r[0])] = str(r[1])

    # Daily features
    daily_features = build_meter_day_features(
        interval_df, instant_df, register_df,
        metadata_df=uc_context,
        report_day=report_day,
        meter_to_uc_map=meter_to_uc,
    )
    write_features(daily_features, "v1", report_day, base_dir=output_dir, write_csv=False)

    # Window features
    window_features = build_uc_window_features(
        daily_features,
        meter_history_df=meter_history,
        cutoff_date=report_day,
        window_days=30,
        feature_set_version="v1",
    )
    write_window_features(window_features, "v1", report_day, base_dir=output_dir, write_csv=False)

    # ── Step 6: Training Dataset — inclui coluna meta__feeder ────────────────
    training_df = build_training_dataset(
        window_features,
        uc_context_df=uc_context,
        hierarchy_df=hierarchy,
        split_policy="train",
    )

    dataset_path = write_training_dataset(training_df, "v1", base_dir=output_dir, write_csv=False)
    _log(f"[+] Dataset de treinamento gerado: {dataset_path}")
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
        _log("\n[*] Disparando treinamento do modelo de anomalia...")
        from scripts.train_anomaly_model import main as train_main
        old_argv = sys.argv
        sys.argv = [
            "train_anomaly_model.py",
            "--input", str(dataset_path),
            "--hierarchy-input", str(output_dir / "context" / "electrical_hierarchy.parquet"),
            "--mode", "composite",
            "--contamination", "0.05",
        ]
        try:
            train_main()
        finally:
            sys.argv = old_argv

    _log("\n" + "=" * 65)
    _log(f"PIPELINE CONCLUÍDO: {scope_label}")
    _log("=" * 65)
    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(
        days_back=args.days_back,
        batch_size=args.batch_size,
        no_cis_cache=args.no_cis_cache,
        feeder_name=args.feeder_name,
        feeder_geo_id=args.feeder_geo_id,
        train_model=args.train_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
