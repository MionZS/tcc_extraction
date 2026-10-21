"""Script de Benchmark de Carga, Profiling em Decisegundos e Monitoramento de RAM em Streaming.

Executa testes de estresse progressivos na extração e normalização do MDM:
1. Monitora RAM (RSS / VMS) e CPU em tempo real através de uma thread em streaming (amostragem a cada 200ms).
2. Cronometra com precisão sub-segundo (decisegundos/milissegundos) cada micro-etapa:
   - Oracle query execute
   - Fetchall overhead
   - Row-to-Polars materialization
   - Raw batch Parquet write
   - Batch JSON Parsing & normalization
   - Final Lazy-Sink I/O
3. Permite testar diferentes batch sizes (ex: 500, 1000, 2500, 5000 ou dia inteiro) e número de workers.
4. Gera log detalhado estruturado no console e em arquivo de relatório (JSON + Markdown).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import queue
import re
import shutil
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
from src.datasets.partitioning import ensure_partition_dirs, measurements_partition_path
from src.db import create_engine

console = Console()

MDM_SQL_PATH = ROOT_DIR / "queries" / "mdm_coluna.sql"


class StreamingMemoryMonitor(threading.Thread):
    """Thread em background que monitora e registra o consumo de RAM em streaming."""

    def __init__(self, interval_sec: float = 0.2):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self.running = True
        self.samples: list[dict[str, Any]] = []
        self.peak_rss_mb = 0.0
        self.current_phase = "INIT"
        self._lock = threading.Lock()

    def set_phase(self, phase_name: str) -> None:
        with self._lock:
            self.current_phase = phase_name

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
                        "timestamp": time.time(),
                        "phase": self.current_phase,
                        "rss_mb": round(rss_mb, 2),
                        "vms_mb": round(vms_mb, 2),
                        "cpu_pct": round(cpu_pct, 1),
                    })
            except Exception:
                pass
            time.sleep(self.interval_sec)


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


def _chunked(values: Sequence[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def run_benchmark_for_day(
    target_day: date,
    *,
    batch_size: int,
    max_workers: int,
    max_meters_limit: int | None = None,
    benchmark_dir: Path,
) -> dict[str, Any]:
    """Executa o benchmark completo para um dia específico registrando tempos em decisegundos."""
    today = date.today()
    curr_days_back = (today - target_day).days
    day_token = target_day.strftime("%Y%m%d")

    _log(f"[bold cyan]▶ Iniciando Benchmark de Extração para {target_day} (Batch: {batch_size} | Workers: {max_workers})[/bold cyan]")

    mem_monitor = StreamingMemoryMonitor(interval_sec=0.2)
    mem_monitor.start()

    engine_orca = create_engine("orca")
    mdm_sql_text = MDM_SQL_PATH.read_text(encoding="utf-8").strip().rstrip(";/")

    # 1. Obter a população ativa para o dia a partir do arquivo CIS já salvo ou consulta direta
    mem_monitor.set_phase("1_DISCOVER_POPULATION")
    t0_pop = time.perf_counter()

    cis_path = ROOT_DIR / "output" / "raw" / "ORCA" / "araucaria_todos_alimentadores" / "cis_latest_araucaria.parquet"
    if cis_path.exists():
        _log(f"[*] Carregando população de medidores a partir de {cis_path.name}...")
        cis_df = pl.read_parquet(cis_path)
        col_cand = [c for c in cis_df.columns if c.upper() in ("NUM_MEDIDOR", "NIO_CLEAN", "NIO", "NUMERO_MEDIDOR")]
        if col_cand:
            nios = [str(n).strip() for n in cis_df[col_cand[0]].drop_nulls().to_list() if str(n).strip()]
        else:
            raise RuntimeError("Nenhuma coluna de número de medidor encontrada no CIS existente.")
    else:
        raise FileNotFoundError(f"Arquivo de população não encontrado em {cis_path}.")

    if max_meters_limit and max_meters_limit < len(nios):
        _log(f"[yellow]⚠ Limitando benchmark para os primeiros {max_meters_limit:,} medidores (de {len(nios):,}).[/yellow]")
        nios = nios[:max_meters_limit]

    total_meters = len(nios)
    chunks = list(enumerate(_chunked(nios, batch_size), start=1))
    total_batches = len(chunks)

    t_pop_sec = time.perf_counter() - t0_pop
    _log(f"[+] População resolvida: {total_meters:,} medidores divididos em {total_batches} lotes (Levou {t_pop_sec:.2f}s).")

    # Diretório temporário dedicado para isolamento de benchmark
    run_benchmark_dir = benchmark_dir / f"run_b{batch_size}_w{max_workers}_{day_token}_{int(time.time())}"
    batches_dir = run_benchmark_dir / "raw_batches"
    norm_temp_dir = run_benchmark_dir / "norm_tmp"
    final_output_dir = run_benchmark_dir / "final"
    batches_dir.mkdir(parents=True, exist_ok=True)
    norm_temp_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Worker connections pool
    mem_monitor.set_phase("2_EXTRACT_ORCA_BATCHES")
    t0_extract = time.perf_counter()

    worker_conns: dict[int, Connection] = {}
    conn_lock = threading.Lock()

    def _get_worker_conn(w: int) -> Connection:
        with conn_lock:
            conn = worker_conns.get(w)
            if conn is None or conn.closed:
                conn = engine_orca.connect()
                worker_conns[w] = conn
            return conn

    worker_pool = queue.Queue()
    for w in range(max_workers):
        worker_pool.put(w)

    batch_timings: list[dict[str, Any]] = []
    timings_lock = threading.Lock()

    def _extract_single_batch(chunk_item: tuple[int, list[str]]) -> tuple[int, Path | None, int]:
        b_idx, b_nios = chunk_item
        b_path = batches_dir / f"batch_{b_idx:05d}.parquet"
        w_id = worker_pool.get()

        t_start = time.perf_counter()
        t_bind_done = t_start
        t_exec_done = t_start
        t_fetch_done = t_start
        t_df_done = t_start
        t_write_done = t_start

        try:
            conn = _get_worker_conn(w_id)

            t_sub0 = time.perf_counter()
            bind_list = _build_oracle_string_list(conn, b_nios)
            t_bind_done = time.perf_counter()

            result = conn.execute(
                text(mdm_sql_text),
                {"DAYS_BACK": curr_days_back, "UCS": bind_list},
            )
            t_exec_done = time.perf_counter()
            cols = [str(c).upper() for c in result.keys()]

            b_rows = result.fetchall()
            t_fetch_done = time.perf_counter()

            n_rows = len(b_rows)
            if n_rows > 0:
                b_df = _rows_to_frame(b_rows, cols)
                t_df_done = time.perf_counter()

                b_df.write_parquet(b_path, compression="zstd")
                t_write_done = time.perf_counter()

                del b_df, b_rows

                with timings_lock:
                    batch_timings.append({
                        "batch_idx": b_idx,
                        "worker_id": w_id,
                        "meter_count": len(b_nios),
                        "row_count": n_rows,
                        "time_bind_sec": round(t_bind_done - t_sub0, 3),
                        "time_exec_sec": round(t_exec_done - t_bind_done, 3),
                        "time_fetch_sec": round(t_fetch_done - t_exec_done, 3),
                        "time_df_materialize_sec": round(t_df_done - t_fetch_done, 3),
                        "time_parquet_write_sec": round(t_write_done - t_df_done, 3),
                        "total_batch_sec": round(t_write_done - t_start, 3),
                    })
                return b_idx, b_path, n_rows
            else:
                return b_idx, None, 0
        finally:
            worker_pool.put(w_id)

    _log(f"[*] Disparando extração paralela de {total_batches} lotes no Oracle com {max_workers} conexões...")
    batch_results: list[tuple[int, Path, int]] = []
    total_mdm_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_single_batch, chunk) for chunk in chunks]
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
    _log(f"[+] Extração Oracle finalizada: {total_mdm_rows:,} linhas em {t_extract_sec:.2f}s ({t_extract_sec/60:.2f} min).")

    # 3. Consolidação Lazy-Sink
    mem_monitor.set_phase("3_LAZY_SINK_CONSOLIDATION")
    t0_sink = time.perf_counter()
    batch_results.sort(key=lambda x: x[0])
    batch_files = [x[1] for x in batch_results]

    raw_consolidated_parquet = run_benchmark_dir / f"mdm_consolidated_{day_token}.parquet"
    if batch_files:
        _log(f"[*] Consolidando {len(batch_files)} lotes em arquivo único Parquet...")
        raw_lazy = pl.concat([pl.scan_parquet(str(p)) for p in batch_files], how="vertical_relaxed")
        raw_lazy.sink_parquet(raw_consolidated_parquet, compression="zstd")
    t_sink_sec = time.perf_counter() - t0_sink
    _log(f"[+] Consolidação Lazy-Sink finalizada em {t_sink_sec:.2f}s.")

    # 4. Normalização Semântica
    mem_monitor.set_phase("4_SEMANTIC_NORMALIZATION")
    t0_norm = time.perf_counter()
    _log(f"[*] Iniciando normalização de camadas semânticas com {max_workers} workers...")

    interval_parquet = final_output_dir / "interval.parquet"
    instant_parquet = final_output_dir / "instantaneous.parquet"
    register_parquet = final_output_dir / "registers.parquet"

    tot_interval, tot_instant, tot_register = normalize_and_sink_mdm_batches(
        batch_files,
        report_day=target_day,
        interval_final_parquet=interval_parquet,
        instant_final_parquet=instant_parquet,
        register_final_parquet=register_parquet,
        temp_dir=norm_temp_dir,
        run_id=f"bench_{int(time.time())}",
        write_csv=False,
        compression="zstd",
        max_workers=max_workers,
    )
    t_norm_sec = time.perf_counter() - t0_norm
    _log(f"[+] Normalização semântica concluída em {t_norm_sec:.2f}s ({t_norm_sec/60:.2f} min):")
    _log(f"    - Intervalos: {tot_interval:,}")
    _log(f"    - Instantâneos: {tot_instant:,}")
    _log(f"    - Registradores: {tot_register:,}")

    mem_monitor.stop()
    engine_orca.dispose()

    # Estatísticas de micro-timing
    avg_bind = sum(b["time_bind_sec"] for b in batch_timings) / len(batch_timings) if batch_timings else 0
    avg_exec = sum(b["time_exec_sec"] for b in batch_timings) / len(batch_timings) if batch_timings else 0
    avg_fetch = sum(b["time_fetch_sec"] for b in batch_timings) / len(batch_timings) if batch_timings else 0
    avg_df = sum(b["time_df_materialize_sec"] for b in batch_timings) / len(batch_timings) if batch_timings else 0
    avg_write = sum(b["time_parquet_write_sec"] for b in batch_timings) / len(batch_timings) if batch_timings else 0

    total_pipeline_sec = t_pop_sec + t_extract_sec + t_sink_sec + t_norm_sec

    summary_result = {
        "target_day": target_day.isoformat(),
        "batch_size": batch_size,
        "max_workers": max_workers,
        "total_meters": total_meters,
        "total_batches": total_batches,
        "total_mdm_rows": total_mdm_rows,
        "total_interval_rows": tot_interval,
        "total_pipeline_sec": round(total_pipeline_sec, 2),
        "total_pipeline_min": round(total_pipeline_sec / 60, 2),
        "stage_times": {
            "population_discovery_sec": round(t_pop_sec, 2),
            "oracle_extract_total_sec": round(t_extract_sec, 2),
            "lazy_sink_consolidate_sec": round(t_sink_sec, 2),
            "semantic_normalization_sec": round(t_norm_sec, 2),
        },
        "average_per_batch_ms": {
            "bind_ms": round(avg_bind * 1000, 1),
            "oracle_exec_ms": round(avg_exec * 1000, 1),
            "fetchall_ms": round(avg_fetch * 1000, 1),
            "rows_to_frame_ms": round(avg_df * 1000, 1),
            "parquet_write_ms": round(avg_write * 1000, 1),
        },
        "memory_telemetry": {
            "peak_rss_mb": round(mem_monitor.peak_rss_mb, 2),
            "total_samples": len(mem_monitor.samples),
        },
    }

    # Salva relatório do run em JSON
    report_file = run_benchmark_dir / "benchmark_report.json"
    report_file.write_text(json.dumps(summary_result, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"[bold green]✔ Relatório de Benchmark salvo em {report_file}[/bold green]")

    return summary_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de Estresse e Carga do Pipeline de Extração.")
    parser.add_argument(
        "--days",
        nargs="+",
        default=["2026-10-04", "2026-10-05"],
        help="Datas para benchmark (AAAA-MM-DD)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000],
        help="Tamanhos de batch para testar progressivamente (ex: 500 1000 2000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Número de workers simultâneos (padrão: 5)",
    )
    parser.add_argument(
        "--meters-limit",
        type=int,
        default=None,
        help="Opcional: limitar a N medidores por teste (útil para dry-run ultra rápido, ex: 5000)",
    )
    args = parser.parse_args()

    benchmark_dir = ROOT_DIR / "output" / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    console.print(Panel.fit(
        f"[bold cyan]BENCHMARK DE CARGA E STREAMING DE TELEMETRIA[/bold cyan]\n\n"
        f"• Datas de Teste: {args.days}\n"
        f"• Variações de Batch: {args.batch_sizes}\n"
        f"• Workers Simultâneos: {args.workers}\n"
        f"• Limite de Medidores: {args.meters_limit or 'Completo (População Total)'}\n"
        f"• Amostragem de RAM: 200ms em thread isolada",
        border_style="cyan"
    ))

    for day_str in args.days:
        target_d = date.fromisoformat(day_str)
        for b_size in args.batch_sizes:
            res = run_benchmark_for_day(
                target_d,
                batch_size=b_size,
                max_workers=args.workers,
                max_meters_limit=args.meters_limit,
                benchmark_dir=benchmark_dir,
            )
            all_results.append(res)

    # Imprime tabela comparativa final
    table = Table(title="Resultados Comparativos de Benchmark")
    table.add_column("Data", style="cyan")
    table.add_column("Batch Size", justify="right")
    table.add_column("Workers", justify="right")
    table.add_column("Medidores", justify="right")
    table.add_column("Tempo Total", justify="right", style="bold green")
    table.add_column("Extração (s)", justify="right")
    table.add_column("Normalização (s)", justify="right")
    table.add_column("Fetch Médio (ms)", justify="right")
    table.add_column("Peak RAM (MB)", justify="right", style="magenta")

    for r in all_results:
        table.add_row(
            r["target_day"],
            str(r["batch_size"]),
            str(r["max_workers"]),
            f"{r['total_meters']:,}",
            f"{r['total_pipeline_min']} min",
            f"{r['stage_times']['oracle_extract_total_sec']}s",
            f"{r['stage_times']['semantic_normalization_sec']}s",
            f"{r['average_per_batch_ms']['fetchall_ms']} ms",
            f"{r['memory_telemetry']['peak_rss_mb']:,} MB",
        )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    main()
