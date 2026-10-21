"""Script de Benchmark de Carga com Batches Heterogêneos em um Único Dia.

Divide a população real de medidores de um único dia em faixas de teste com diferentes
tamanhos de batch (ex: 250, 500, 1000, 2000, 4000, 8000), executando tudo em uma única
passagem contínua.

Recursos:
1. Thread em background de monitoramento em streaming de RAM (RSS/VMS) e CPU a cada 100ms.
2. Cronometragem detalhada de cada round-trip Oracle no mesmo processo:
   - Bind SYS.ODCIVARCHAR2LIST
   - Query execute no banco (RTT de rede + tempo de plano no Oracle)
   - Fetchall overhead (tempo de streaming de dados pela rede)
   - Rows to Polars DataFrame materialization
   - Compressão e gravação Parquet em disco
3. Profiling da etapa de normalização semântica (parsing de JSON dos 288 intervalos).
4. Métricas consolidadas por regime de batch:
   - Throughput (medidores/segundo e linhas/segundo)
   - Latência média por medidor
   - Eficiência de I/O e uso de RAM
5. Exporta relatório completo em JSON e tabela formatada no terminal.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.datasets.normalize import normalize_and_sink_mdm_batches
from src.db import create_engine

console = Console()
MDM_SQL_PATH = ROOT_DIR / "queries" / "mdm_coluna.sql"


class StreamingTelemetryMonitor(threading.Thread):
    """Thread em background que amostra telemetria de RAM e CPU em streaming de alta frequência."""

    def __init__(self, sample_interval_sec: float = 0.1):
        super().__init__(daemon=True)
        self.sample_interval_sec = sample_interval_sec
        self.process = psutil.Process(os.getpid())
        self.running = True
        self.samples: list[dict[str, Any]] = []
        self.peak_rss_mb = 0.0
        self.current_regime = "INIT"
        self._lock = threading.Lock()

    def set_regime(self, regime_name: str) -> None:
        with self._lock:
            self.current_regime = regime_name

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            try:
                mem_info = self.process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                vms_mb = mem_info.vms / (1024 * 1024)
                cpu_pct = self.process.cpu_percent(interval=None)

                with self._lock:
                    if rss_mb > self.peak_rss_mb:
                        self.peak_rss_mb = rss_mb
                    self.samples.append({
                        "epoch": time.time(),
                        "regime": self.current_regime,
                        "rss_mb": round(rss_mb, 2),
                        "vms_mb": round(vms_mb, 2),
                        "cpu_pct": round(cpu_pct, 1),
                    })
            except Exception:
                pass
            time.sleep(self.sample_interval_sec)


def _log(msg: str) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    console.print(f"[dim][{now_str}][/dim] {msg}")


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


def generate_heterogeneous_batches(
    nios: list[str],
    batch_specs: list[tuple[int, int]],
    *,
    fill_remaining: bool = False,
) -> list[dict[str, Any]]:
    """Particiona a lista de medidores em lotes conforme a especificação."""
    batches: list[dict[str, Any]] = []
    curr_pos = 0
    total_nios = len(nios)
    batch_idx = 1

    for b_size, target_count in batch_specs:
        batches_created = 0
        while curr_pos < total_nios and (target_count is None or batches_created < target_count):
            chunk = nios[curr_pos : curr_pos + b_size]
            if not chunk:
                break
            batches.append({
                "batch_idx": batch_idx,
                "regime_name": f"Batch_{b_size}",
                "target_size": b_size,
                "actual_size": len(chunk),
                "nios": chunk,
            })
            curr_pos += len(chunk)
            batch_idx += 1
            batches_created += 1

    # Se ainda restarem medidores e fill_remaining for True, preenche com o último tamanho
    if fill_remaining and curr_pos < total_nios:
        fallback_size = batch_specs[-1][0] if batch_specs else 1000
        while curr_pos < total_nios:
            chunk = nios[curr_pos : curr_pos + fallback_size]
            batches.append({
                "batch_idx": batch_idx,
                "regime_name": f"Batch_{fallback_size}",
                "target_size": fallback_size,
                "actual_size": len(chunk),
                "nios": chunk,
            })
            curr_pos += len(chunk)
            batch_idx += 1

    return batches


def run_heterogeneous_benchmark(
    target_day: date,
    *,
    max_workers: int = 5,
    max_meters_limit: int | None = None,
    batch_plan: list[tuple[int, int]] | None = None,
    equal_batches: int | None = None,
) -> dict[str, Any]:
    today = date.today()
    curr_days_back = (today - target_day).days
    day_token = target_day.strftime("%Y%m%d")

    _log(f"[bold cyan]▶ Iniciando Benchmark Heterogêneo para o dia {target_day.isoformat()}[/bold cyan]")
    _log(f"   Workers simultâneos: {max_workers} | Days back: {curr_days_back}")

    # Inicia telemetria em streaming
    monitor = StreamingTelemetryMonitor(sample_interval_sec=0.1)
    monitor.start()

    # 1. Carregar população ativa de medidores
    monitor.set_regime("1_POPULATION_LOAD")
    all_nios: list[str] = []

    # Procura na ordem: 1) uc_context/meter_installation_history normalizados, 2) CIS brutos em output/raw/CIS
    context_parquet = ROOT_DIR / "output" / "context" / "meter_installation_history.parquet"
    if context_parquet.exists():
        _log(f"[*] Carregando população de medidores a partir de {context_parquet.name}...")
        c_df = pl.read_parquet(context_parquet)
        if "NIO" in c_df.columns:
            all_nios = [str(n).strip() for n in c_df["NIO"].drop_nulls().to_list() if str(n).strip()]

    if not all_nios:
        raw_cis_dir = ROOT_DIR / "output" / "raw" / "CIS"
        cis_candidates = sorted(list(raw_cis_dir.glob("araucaria_cis_*.csv")), reverse=True)
        if cis_candidates:
            latest_cis = cis_candidates[0]
            _log(f"[*] Carregando população de medidores a partir de {latest_cis.name}...")
            cis_df = pl.read_csv(latest_cis, separator=";", infer_schema_length=0)
            col_cand = [c for c in cis_df.columns if c.upper() in ("NUM_MEDIDOR", "NIO_CLEAN", "NIO", "NUMERO_MEDIDOR")]
            if col_cand:
                all_nios = [str(n).strip() for n in cis_df[col_cand[0]].drop_nulls().to_list() if str(n).strip()]

    if not all_nios:
        raise FileNotFoundError(
            "Nenhum arquivo de população encontrado. Esperava encontrar 'output/context/meter_installation_history.parquet' "
            "ou arquivos em 'output/raw/CIS/araucaria_cis_*.csv'."
        )

    # Remove duplicados preservando ordem estável
    seen = set()
    dedup_nios = []
    for n in all_nios:
        if n not in seen:
            seen.add(n)
            dedup_nios.append(n)
    all_nios = dedup_nios

    if max_meters_limit and max_meters_limit < len(all_nios):
        _log(f"[yellow]⚠ Limitando teste para {max_meters_limit:,} medidores (de {len(all_nios):,}).[/yellow]")
    if equal_batches and equal_batches > 0:
        chunk_size = (len(all_nios) + equal_batches - 1) // equal_batches
        planned_batches = []
        for b_i in range(equal_batches):
            chunk = all_nios[b_i * chunk_size : (b_i + 1) * chunk_size]
            if not chunk:
                break
            planned_batches.append({
                "batch_idx": b_i + 1,
                "regime_name": f"Batch_{len(chunk)}",
                "target_size": len(chunk),
                "actual_size": len(chunk),
                "nios": chunk,
            })
    else:
        fill_rem = False if batch_plan is not None else True
        if not batch_plan:
            batch_plan = [
                (1000, 2),   # 2.000 medidores (baseline de controle)
                (2000, 3),   # 6.000 medidores
                (3500, 3),   # 10.500 medidores
                (5000, 3),   # 15.000 medidores
                (8000, 2),   # 16.000 medidores
                (12000, 2),  # 24.000 medidores
            ]
        planned_batches = generate_heterogeneous_batches(all_nios, batch_plan, fill_remaining=fill_rem)

    total_meters = sum(b["actual_size"] for b in planned_batches)
    total_batches = len(planned_batches)

    # Diretório temporário de execução
    bench_dir = ROOT_DIR / "output" / "benchmarks" / f"hetero_bench_{day_token}_{int(time.time())}"
    batches_dir = bench_dir / "raw_batches"
    norm_tmp_dir = bench_dir / "norm_tmp"
    final_dir = bench_dir / "final"
    batches_dir.mkdir(parents=True, exist_ok=True)
    norm_tmp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[+] Total de medidores: {total_meters:,} distribuídos em {total_batches} lotes heterogêneos.")

    engine_orca = create_engine("orca")
    mdm_sql_text = MDM_SQL_PATH.read_text(encoding="utf-8").strip().rstrip(";/")

    # Worker pool
    worker_conns: dict[int, Connection] = {}
    conn_lock = threading.Lock()

    def _get_conn(w: int) -> Connection:
        with conn_lock:
            c = worker_conns.get(w)
            if c is None or c.closed:
                c = engine_orca.connect()
                worker_conns[w] = c
            return c

    worker_pool = queue.Queue()
    for w in range(max_workers):
        worker_pool.put(w)

    detailed_timings: list[dict[str, Any]] = []
    timings_lock = threading.Lock()

    monitor.set_regime("2_ORACLE_EXTRACTION")
    t0_extract = time.perf_counter()

    def _process_single_batch(batch_info: dict[str, Any]) -> tuple[int, Path | None, int]:
        b_idx = batch_info["batch_idx"]
        b_regime = batch_info["regime_name"]
        b_target_size = batch_info["target_size"]
        b_nios = batch_info["nios"]
        actual_size = len(b_nios)
        b_path = batches_dir / f"batch_{b_idx:05d}_{b_regime}.parquet"

        w_id = worker_pool.get()
        t_start = time.perf_counter()

        _log(f"[{b_idx:03d}/{total_batches:03d}] [bold cyan]⚡ Worker #{w_id + 1:02d}[/]: Iniciando {b_regime} ({actual_size:,} NIOs) no Oracle...")

        try:
            conn = _get_conn(w_id)

            # 1. Bind
            t_b0 = time.perf_counter()
            bind_list = _build_oracle_string_list(conn, b_nios)
            t_bind = time.perf_counter() - t_b0

            # 2. Execute query (Round-trip banco + plan cost)
            t_e0 = time.perf_counter()
            stmt = text(mdm_sql_text).execution_options(yield_per=1000)
            result = conn.execute(
                stmt,
                {"DAYS_BACK": curr_days_back, "UCS": bind_list},
            )
            t_exec = time.perf_counter() - t_e0
            cols = [str(c).upper() for c in result.keys()]

            # Acelera streaming de rede aumentando o buffer de linhas do Oracle DBAPI
            try:
                raw_cur = getattr(result, "cursor", None)
                if raw_cur is not None and hasattr(raw_cur, "arraysize"):
                    raw_cur.arraysize = 1000
            except Exception:
                pass

            _log(f"[{b_idx:03d}/{total_batches:03d}] [yellow]⚡ Worker #{w_id + 1:02d}[/]: {b_regime} query executada ({t_exec*1000:.1f}ms). Baixando linhas (fetch)...")

            # 3. Fetchall (Streaming de rede dos dados brutos)
            t_f0 = time.perf_counter()
            b_rows = result.fetchall()
            t_fetch = time.perf_counter() - t_f0

            n_rows = len(b_rows)
            t_df = 0.0
            t_write = 0.0

            if n_rows > 0:
                # 4. Materialização Polars
                t_d0 = time.perf_counter()
                b_df = _rows_to_frame(b_rows, cols)
                t_df = time.perf_counter() - t_d0

                # 5. Escrita Parquet ZSTD
                t_w0 = time.perf_counter()
                b_df.write_parquet(b_path, compression="zstd")
                t_write = time.perf_counter() - t_w0

                file_size_kb = b_path.stat().st_size / 1024
                del b_df, b_rows

                total_batch_t = time.perf_counter() - t_start
                meters_per_sec = actual_size / total_batch_t if total_batch_t > 0 else 0
                ms_per_meter = (total_batch_t * 1000) / actual_size if actual_size > 0 else 0

                timing_record = {
                    "batch_idx": b_idx,
                    "regime": b_regime,
                    "target_size": b_target_size,
                    "actual_meters": actual_size,
                    "worker_id": w_id,
                    "returned_rows": n_rows,
                    "file_size_kb": round(file_size_kb, 1),
                    "bind_ms": round(t_bind * 1000, 1),
                    "oracle_exec_ms": round(t_exec * 1000, 1),
                    "fetchall_ms": round(t_fetch * 1000, 1),
                    "rows_to_frame_ms": round(t_df * 1000, 1),
                    "parquet_write_ms": round(t_write * 1000, 1),
                    "total_batch_sec": round(total_batch_t, 3),
                    "meters_per_sec": round(meters_per_sec, 1),
                    "ms_per_meter": round(ms_per_meter, 2),
                }

                with timings_lock:
                    detailed_timings.append(timing_record)

                _log(
                    f"[{b_idx:03d}/{total_batches:03d}] [bold green]✔ {b_regime}[/] (NIOs: {actual_size:,} | Linhas: {n_rows:,}) → "
                    f"Exec: [cyan]{timing_record['oracle_exec_ms']}ms[/] | "
                    f"Fetch: [yellow]{timing_record['fetchall_ms']}ms[/] | "
                    f"Total: [bold]{timing_record['total_batch_sec']}s[/] "
                    f"([green]{timing_record['ms_per_meter']} ms/medidor[/])"
                )
                return b_idx, b_path, n_rows
            else:
                _log(f"[{b_idx:03d}/{total_batches:03d}] [dim]{b_regime} (0 linhas retornadas)[/]")
                return b_idx, None, 0

        finally:
            worker_pool.put(w_id)

    # Executa batches heterogêneos
    batch_results: list[tuple[int, Path, int]] = []
    total_mdm_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_batch, b) for b in planned_batches]
        for fut in as_completed(futures):
            b_idx, b_path, n_rows = fut.result()
            if b_path is not None and n_rows > 0:
                batch_results.append((b_idx, b_path, n_rows))
                total_mdm_rows += n_rows

    for c in worker_conns.values():
        try:
            c.close()
        except Exception:
            pass
    worker_conns.clear()

    t_extract_sec = time.perf_counter() - t0_extract
    _log(f"\n[+] Extração de todos os regimes concluída em {t_extract_sec:.2f}s ({t_extract_sec/60:.2f} min).")

    # Consolidação Lazy-Sink
    monitor.set_regime("3_LAZY_SINK")
    t0_sink = time.perf_counter()
    batch_results.sort(key=lambda x: x[0])
    batch_files = [x[1] for x in batch_results]

    consolidated_parquet = bench_dir / f"mdm_hetero_consolidated_{day_token}.parquet"
    if batch_files:
        _log(f"[*] Consolidando {len(batch_files)} lotes heterogêneos via Lazy-Sink...")
        raw_lazy = pl.concat([pl.scan_parquet(str(p)) for p in batch_files], how="vertical_relaxed")
        raw_lazy.sink_parquet(consolidated_parquet, compression="zstd")
    t_sink_sec = time.perf_counter() - t0_sink
    _log(f"[+] Lazy-Sink finalizado em {t_sink_sec:.2f}s.")

    # Normalização Semântica
    monitor.set_regime("4_SEMANTIC_NORM")
    t0_norm = time.perf_counter()
    _log(f"[*] Normalizando camadas semânticas para {len(batch_files)} arquivos com {max_workers} workers...")

    interval_parquet = final_dir / "interval.parquet"
    instant_parquet = final_dir / "instantaneous.parquet"
    register_parquet = final_dir / "registers.parquet"

    tot_interval, tot_instant, tot_register = normalize_and_sink_mdm_batches(
        batch_files,
        report_day=target_day,
        interval_final_parquet=interval_parquet,
        instant_final_parquet=instant_parquet,
        register_final_parquet=register_parquet,
        temp_dir=norm_tmp_dir,
        run_id=f"hetero_{int(time.time())}",
        write_csv=False,
        compression="zstd",
        max_workers=max_workers,
    )
    t_norm_sec = time.perf_counter() - t0_norm
    _log(f"[+] Normalização semântica finalizada em {t_norm_sec:.2f}s ({t_norm_sec/60:.2f} min).")

    monitor.stop()
    engine_orca.dispose()

    # ── Agrupamento e Análise Estatística por Regime de Batch ─────────────────
    regimes_summary: dict[str, Any] = {}
    df_timings = pl.DataFrame(detailed_timings) if detailed_timings else pl.DataFrame()

    if not df_timings.is_empty():
        for regime_name in sorted(df_timings["regime"].unique().to_list()):
            sub = df_timings.filter(pl.col("regime") == regime_name)
            count = sub.height
            total_regime_meters = sub["actual_meters"].sum()
            avg_exec = sub["oracle_exec_ms"].mean()
            avg_fetch = sub["fetchall_ms"].mean()
            avg_df = sub["rows_to_frame_ms"].mean()
            avg_write = sub["parquet_write_ms"].mean()
            avg_total_sec = sub["total_batch_sec"].mean()
            avg_ms_per_meter = sub["ms_per_meter"].mean()
            throughput_meters_sec = total_regime_meters / sub["total_batch_sec"].sum() if sub["total_batch_sec"].sum() > 0 else 0

            regimes_summary[regime_name] = {
                "batch_count": count,
                "total_meters": total_regime_meters,
                "avg_oracle_exec_ms": round(float(avg_exec), 1),
                "avg_fetchall_ms": round(float(avg_fetch), 1),
                "avg_rows_to_frame_ms": round(float(avg_df), 1),
                "avg_parquet_write_ms": round(float(avg_write), 1),
                "avg_batch_duration_sec": round(float(avg_total_sec), 2),
                "avg_latency_ms_per_meter": round(float(avg_ms_per_meter), 2),
                "throughput_meters_per_sec": round(float(throughput_meters_sec), 1),
            }

    # Salva relatório completo
    benchmark_report = {
        "target_day": target_day.isoformat(),
        "total_meters": total_meters,
        "total_batches": total_batches,
        "total_mdm_rows": total_mdm_rows,
        "total_interval_rows": tot_interval,
        "timings_summary_sec": {
            "oracle_extract_total": round(t_extract_sec, 2),
            "lazy_sink_total": round(t_sink_sec, 2),
            "semantic_norm_total": round(t_norm_sec, 2),
            "total_day_pipeline": round(t_extract_sec + t_sink_sec + t_norm_sec, 2),
        },
        "regimes_comparison": regimes_summary,
        "memory_telemetry": {
            "peak_rss_mb": round(monitor.peak_rss_mb, 2),
            "total_telemetry_samples": len(monitor.samples),
        },
        "detailed_batches": detailed_timings,
    }

    report_path = bench_dir / "heterogeneous_benchmark_report.json"
    report_path.write_text(json.dumps(benchmark_report, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"[bold green]✔ Relatório consolidado gravado em {report_path}[/bold green]")

    # Exibe tabela comparativa na tela
    table = Table(title=f"Comparativo de Performance por Tamanho de Lote ({target_day.isoformat()})")
    table.add_column("Regime de Batch", style="cyan bold")
    table.add_column("Lotes", justify="right")
    table.add_column("Total Medidores", justify="right")
    table.add_column("Exec Banco (ms)", justify="right")
    table.add_column("Fetch Dados (ms)", justify="right")
    table.add_column("Tempo Lote (s)", justify="right")
    table.add_column("Custo por Medidor (ms/medidor)", justify="right", style="bold yellow")
    table.add_column("Throughput (med/s)", justify="right", style="bold green")

    for reg_name, stats in regimes_summary.items():
        table.add_row(
            reg_name,
            str(stats["batch_count"]),
            f"{stats['total_meters']:,}",
            f"{stats['avg_oracle_exec_ms']} ms",
            f"{stats['avg_fetchall_ms']} ms",
            f"{stats['avg_batch_duration_sec']} s",
            f"{stats['avg_latency_ms_per_meter']} ms",
            f"{stats['throughput_meters_per_sec']} med/s",
        )

    console.print("\n")
    console.print(table)
    console.print(f"[bold magenta]Pico de Consumo de RAM registrado:[/] {monitor.peak_rss_mb:.1f} MB\n")

    return benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark com Batches Heterogêneos em um único dia.")
    parser.add_argument(
        "--day",
        type=str,
        default="2026-10-04",
        help="Data alvo do teste (AAAA-MM-DD)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Número de conexões paralelas no pool (padrão: 5)",
    )
    parser.add_argument(
        "--meters-limit",
        type=int,
        default=None,
        help="Opcional: limitar para os primeiros N medidores (ex: 30000 para teste)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Lista de tamanhos de batch a intercalar (ex: --batch-sizes 2000 3500 5000 8000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamanho de batch fixo para todos os lotes (ex: 12000, 14000, 16000)",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=None,
        help="Número exato de lotes desse tamanho (ex: 5)",
    )
    parser.add_argument(
        "--equal-batches",
        type=int,
        default=None,
        help="Divide a população do dia em N lotes perfeitamente iguais e simultâneos (ex: --equal-batches 5)",
    )
    args = parser.parse_args()

    custom_plan = None
    if args.batch_size and args.batch_count:
        custom_plan = [(args.batch_size, args.batch_count)]
    elif args.batch_sizes:
        limit = args.meters_limit or 72000
        n_sizes = len(args.batch_sizes)
        quota_per_size = limit // n_sizes

        custom_plan = []
        for bs in args.batch_sizes:
            count = max(1, quota_per_size // bs)
            custom_plan.append((bs, count))

    target_d = date.fromisoformat(args.day)
    run_heterogeneous_benchmark(
        target_d,
        max_workers=args.workers,
        max_meters_limit=args.meters_limit,
        batch_plan=custom_plan,
        equal_batches=args.equal_batches,
    )


if __name__ == "__main__":
    main()
