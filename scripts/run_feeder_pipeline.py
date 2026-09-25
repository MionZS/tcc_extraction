"""Execute complete extraction, normalization, and ML modeling for a specific Feeder.

Defaults to Alimentador Fonte Nova (GEO ID 6352460).

Flow:
1. CIS population extract (or reuse existing daily raw/CIS file).
2. Direct GEO extract for all UCs belonging to the feeder.
3. Inner join CIS with GEO to get exact feeder population and NIOs.
4. MDM extract in batches of 500 for the feeder's meters.
5. Export semantic layers (context, measurements, features, model_input).
6. Train baseline unsupervised ML anomaly detection model.

Usage:
    python scripts/run_feeder_pipeline.py --days-back 1
    python scripts/run_feeder_pipeline.py --feeder-geo-id 6352460 --feeder-name Fonte_Nova
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
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
GEO_FEEDER_SQL = QUERIES_DIR / "geo_feeder_direct.sql"
MDM_SQL = QUERIES_DIR / "mdm_coluna.sql"


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
    parser = argparse.ArgumentParser(description="Run complete pipeline for a specific feeder.")
    parser.add_argument(
        "--feeder-geo-id",
        type=int,
        default=6352460,
        help="GEO numeric ID of the feeder (default: 6352460 - Fonte Nova)",
    )
    parser.add_argument(
        "--feeder-name",
        type=str,
        default="Fonte_Nova",
        help="Label for the feeder output (default: Fonte_Nova)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Days back from today (default: 1 = yesterday)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="NIOs per MDM batch (default: 500)",
    )
    parser.add_argument(
        "--no-cis-cache",
        action="store_true",
        help="Force re-extraction of CIS from DB even if raw file exists",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of concurrent workers for MDM extract and normalization (default: 5)",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        default=True,
        help="Train baseline ML model after dataset generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_day = date.today() - timedelta(days=args.days_back)
    token = _date_token(report_day)
    output_dir = ROOT_DIR / "output"

    _log("=" * 60)
    _log(f"FEEDER PIPELINE: {args.feeder_name} (ID {args.feeder_geo_id})")
    _log(f"Report Day: {report_day.isoformat()} (days_back={args.days_back})")
    _log("=" * 60)

    manifest = create_manifest(report_day, sample_size=0)

    # ── Step 1: CIS Population ────────────────────────────────────────────
    raw_cis_dir = output_dir / "raw" / "CIS"
    raw_cis_dir.mkdir(parents=True, exist_ok=True)
    cis_file = raw_cis_dir / f"araucaria_cis_{token}.csv"

    if cis_file.exists() and not args.no_cis_cache:
        _log(f"[*] Reusing existing CIS raw extract: {cis_file.name}")
        cis_df = pl.read_csv(cis_file, separator=";", infer_schema_length=0)
    else:
        _log("[*] Extracting CIS population from Oracle...")
        engine = create_engine("cis")
        sql = CIS_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            cols = [str(c).upper() for c in result.keys()]
            cis_df = _rows_to_frame(rows, cols)
        engine.dispose()
        cis_df.write_csv(cis_file, separator=";")
        _log(f"[+] CIS extraction complete: {cis_df.height:,} rows.")

    manifest.record_step("cis_extract", rows=cis_df.height)

    # Normalize UC column in CIS
    cis_df = cis_df.rename({c: c.upper() for c in cis_df.columns})
    cis_df = cis_df.with_columns(
        pl.col("UC").map_elements(_normalize_key, return_dtype=pl.String).alias("UC"),
        pl.col("NIO").map_elements(_normalize_key, return_dtype=pl.String).alias("NIO"),
    )

    # ── Step 2: Direct GEO Feeder Extract ─────────────────────────────────
    _log(f"[*] Querying GEO for feeder {args.feeder_name} (ID: {args.feeder_geo_id})...")
    engine_geo = create_engine("geo")
    geo_sql = GEO_FEEDER_SQL.read_text(encoding="utf-8").strip().rstrip(";/")
    with engine_geo.connect() as conn:
        result = conn.execute(text(geo_sql), {"FEEDER_GEO_ID": args.feeder_geo_id})
        rows = result.fetchall()
        cols = [str(c).upper() for c in result.keys()]
        geo_df = _rows_to_frame(rows, cols)
    engine_geo.dispose()

    geo_feeder_dir = output_dir / "raw" / "GEO" / f"feeder_{args.feeder_name}"
    geo_feeder_dir.mkdir(parents=True, exist_ok=True)
    geo_file = geo_feeder_dir / f"geo_{args.feeder_name}_{token}.csv"
    geo_df.write_csv(geo_file, separator=";")
    _log(f"[+] Feeder GEO extract complete: {geo_df.height:,} UCs found in feeder.")
    manifest.record_step("geo_feeder_extract", rows=geo_df.height)

    if geo_df.is_empty():
        _log("[-] No UCs found for this feeder. Aborting.")
        return 1

    geo_df = geo_df.with_columns(
        pl.col("UC").map_elements(_normalize_key, return_dtype=pl.String).alias("UC")
    )

    # ── Step 3: Inner Join Feeder UCs with CIS ────────────────────────────
    # Deduplicate CIS by (UC, NIO) and GEO by UC to prevent cartesian fan-out
    cis_df = cis_df.unique(subset=["UC", "NIO"])
    geo_df = geo_df.unique(subset=["UC"])
    feeder_cis_geo = cis_df.join(geo_df, on="UC", how="inner", suffix="_GEO")
    _log(f"[+] Matched {feeder_cis_geo.height:,} UCs/meters on feeder {args.feeder_name}.")
    manifest.record_step("feeder_join", rows=feeder_cis_geo.height)

    feeder_nios = (
        feeder_cis_geo.get_column("NIO")
        .drop_nulls()
        .unique()
        .to_list()
    )
    _log(f"[+] Unique meters (NIOs) to extract in MDM: {len(feeder_nios):,}")

    # ── Step 4: MDM Extract for Feeder (Parallel Lazy-Sink Batch Extraction)
    raw_orca_dir = output_dir / "raw" / "ORCA" / f"feeder_{args.feeder_name}"
    raw_orca_batches_dir = raw_orca_dir / f"batches_{token}"
    raw_orca_batches_dir.mkdir(parents=True, exist_ok=True)
    mdm_file = raw_orca_dir / f"mdm_{args.feeder_name}_{token}.csv"
    mdm_file_parquet = raw_orca_dir / f"mdm_{args.feeder_name}_{token}.parquet"

    chunks = list(enumerate(_chunked(feeder_nios, args.batch_size), start=1))
    total_batches = len(chunks)
    _log(
        f"[*] Extracting MDM telemetry for {len(feeder_nios):,} meters in {total_batches} batches "
        f"with {args.max_workers} concurrent workers..."
    )
    engine_orca = create_engine("orca")
    mdm_sql_text = MDM_SQL.read_text(encoding="utf-8").strip().rstrip(";/")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    completed_count = 0
    batch_results: list[tuple[int, Path, int]] = []
    total_mdm_rows = 0

    def _extract_feeder_batch(item: tuple[int, list[str]]) -> tuple[int, Path | None, int]:
        nonlocal completed_count
        b_idx, b_nios = item
        b_path = raw_orca_batches_dir / f"mdm_batch_{b_idx:05d}.parquet"

        if b_path.exists() and b_path.stat().st_size > 0:
            try:
                cached_df = pl.read_parquet(b_path)
                with lock:
                    completed_count += 1
                    _log(f"    [{completed_count:03d}/{total_batches}] Batch {b_idx:05d} cached: {cached_df.height:,} rows.")
                return b_idx, b_path, cached_df.height
            except Exception:
                pass

        try:
            with engine_orca.connect() as conn:
                bind_list = _build_oracle_string_list(conn, b_nios)
                result = conn.execute(
                    text(mdm_sql_text),
                    {"DAYS_BACK": args.days_back, "UCS": bind_list},
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
                        _log(f"    [{completed_count:03d}/{total_batches}] Batch {b_idx:05d} extracted: {n_rows:,} rows ({len(b_nios)} NIOs).")
                    return b_idx, b_path, n_rows
        except Exception as exc:
            with lock:
                _log(f"    [-] Error in batch {b_idx:05d}: {exc}")
            raise

        with lock:
            completed_count += 1
            _log(f"    [{completed_count:03d}/{total_batches}] Batch {b_idx:05d}: 0 rows.")
        return b_idx, None, 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_extract_feeder_batch, chunk) for chunk in chunks]
        for fut in as_completed(futures):
            b_idx, b_path, n_rows = fut.result()
            if b_path is not None and n_rows > 0:
                batch_results.append((b_idx, b_path, n_rows))
                total_mdm_rows += n_rows

    engine_orca.dispose()

    batch_results.sort(key=lambda x: x[0])
    batch_files: list[Path] = [x[1] for x in batch_results]

    if batch_files:
        _log(f"[*] Streaming {len(batch_files)} raw MDM batches to consolidated files via Lazy-Sink...")
        raw_lazy = pl.concat([pl.scan_parquet(str(p)) for p in batch_files], how="vertical_relaxed")
        raw_lazy.sink_parquet(mdm_file_parquet, compression="zstd")
        raw_lazy.sink_csv(mdm_file, separator=";")
        _log(f"[+] MDM raw telemetry complete: {total_mdm_rows:,} rows saved to {mdm_file_parquet.name} and {mdm_file.name}.")
    else:
        _log("[-] No MDM rows returned.")

    manifest.record_step("mdm_extract", rows=total_mdm_rows)

    # ── Step 5: Semantic Layers Normalization (Lazy-Sink Streaming) ────────
    _log(f"[*] Generating multi-layer data lake (context, measurements, features, model_input) with {args.max_workers} workers...")

    # 1. Context
    uc_context = normalize_uc_context(feeder_cis_geo, run_id=manifest.run_id)
    write_context_table(uc_context, "uc_context", base_dir=output_dir, write_csv=True)

    meter_history = normalize_meter_installation_history(feeder_cis_geo, run_id=manifest.run_id)
    write_context_table(meter_history, "meter_installation_history", base_dir=output_dir, write_csv=True)

    hierarchy = normalize_electrical_hierarchy(feeder_cis_geo, run_id=manifest.run_id)
    write_context_table(hierarchy, "electrical_hierarchy", base_dir=output_dir, write_csv=True)

    # 2. Measurements via Lazy-Sink
    _log("[*] Normalizing MDM measurements via Lazy-Sink streaming (zero OOM)...")
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
        write_csv=True,
        compression="zstd",
        max_workers=args.max_workers,
    )
    shutil.rmtree(norm_temp_dir, ignore_errors=True)
    _log(f"[+] Normalized measurements sunk: {total_interval:,} interval, {total_instant:,} instant, {total_register:,} register rows.")

    interval_df = pl.read_parquet(interval_parquet) if interval_parquet.exists() else pl.DataFrame()
    instant_df = pl.read_parquet(instant_parquet) if instant_parquet.exists() else pl.DataFrame()
    register_df = pl.read_parquet(register_parquet) if register_parquet.exists() else pl.DataFrame()

    # Map NIO to UC
    meter_to_uc = {}
    if not meter_history.is_empty():
        for r in meter_history.select(["NIO", "UC"]).iter_rows():
            if r[0] and r[1]:
                meter_to_uc[str(r[0])] = str(r[1])

    # 3. Daily features
    daily_features = build_meter_day_features(
        interval_df, instant_df, register_df,
        metadata_df=uc_context,
        report_day=report_day,
        meter_to_uc_map=meter_to_uc,
    )
    write_features(daily_features, "v1", report_day, base_dir=output_dir, write_csv=True)

    # 4. Window features
    window_features = build_uc_window_features(
        daily_features,
        meter_history_df=meter_history,
        cutoff_date=report_day,
        window_days=30,
        feature_set_version="v1",
    )
    write_window_features(window_features, "v1", report_day, base_dir=output_dir, write_csv=True)

    # 5. Training Dataset
    training_df = build_training_dataset(
        window_features,
        uc_context_df=uc_context,
        hierarchy_df=hierarchy,
        split_policy="train",
    )
    dataset_path = write_training_dataset(training_df, "v1", base_dir=output_dir, write_csv=True)
    _log(f"[+] Model input dataset ready: {dataset_path}")
    manifest.record_step("build_training_dataset", rows=training_df.height, output=str(dataset_path))

    manifest.finalize()
    m_path = manifest.save()
    _log(f"[+] Run manifest saved: {m_path}")

    # ── Step 6: Train Baseline Model ──────────────────────────────────────
    if args.train_model and not training_df.is_empty():
        _log("\n[*] Triggering Baseline Anomaly Model Training...")
        from scripts.train_anomaly_model import main as train_main
        import sys
        old_argv = sys.argv
        sys.argv = [
            "train_anomaly_model.py",
            "--input", str(dataset_path),
            "--output-dir", str(output_dir),
            "--contamination", "0.05",
        ]
        try:
            train_main()
        finally:
            sys.argv = old_argv

    _log("\n" + "=" * 60)
    _log("FEEDER RUN COMPLETED SUCCESSFULLY!")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

