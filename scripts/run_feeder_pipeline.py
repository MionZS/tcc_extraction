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
    normalize_mdm_dataframe,
)
from src.datasets.writers import (
    write_context_table,
    write_measurements_table,
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
            cis_df = pl.DataFrame(
                [[getattr(v, "read", lambda: v)() if hasattr(v, "read") else v for v in r] for r in rows],
                schema=cols,
                orient="row",
            )
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
        geo_df = pl.DataFrame(
            [[getattr(v, "read", lambda: v)() if hasattr(v, "read") else v for v in r] for r in rows],
            schema=cols,
            orient="row",
        )
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

    # ── Step 4: MDM Extract for Feeder ────────────────────────────────────
    raw_orca_dir = output_dir / "raw" / "ORCA" / f"feeder_{args.feeder_name}"
    raw_orca_dir.mkdir(parents=True, exist_ok=True)
    mdm_file = raw_orca_dir / f"mdm_{args.feeder_name}_{token}.csv"

    _log(f"[*] Extracting MDM telemetry for {len(feeder_nios):,} meters...")
    engine_orca = create_engine("orca")
    mdm_sql_text = MDM_SQL.read_text(encoding="utf-8").strip().rstrip(";/")

    mdm_dfs: list[pl.DataFrame] = []
    with engine_orca.connect() as conn:
        for batch_idx, batch_nios in enumerate(_chunked(feeder_nios, args.batch_size), start=1):
            _log(f"    MDM batch {batch_idx}: {len(batch_nios)} NIOs...")
            bind_list = _build_oracle_string_list(conn, batch_nios)
            result = conn.execute(
                text(mdm_sql_text),
                {"DAYS_BACK": args.days_back, "UCS": bind_list},
            )
            cols = [str(c).upper() for c in result.keys()]
            b_rows = result.fetchall()
            if b_rows:
                b_df = pl.DataFrame(
                    [[getattr(v, "read", lambda: v)() if hasattr(v, "read") else v for v in r] for r in b_rows],
                    schema=cols,
                    orient="row",
                )
                mdm_dfs.append(b_df)
    engine_orca.dispose()

    if mdm_dfs:
        mdm_df = pl.concat(mdm_dfs, how="vertical_relaxed")
        mdm_df.write_csv(mdm_file, separator=";")
        _log(f"[+] MDM telemetry complete: {mdm_df.height:,} rows saved to {mdm_file.name}.")
    else:
        mdm_df = pl.DataFrame()
        _log("[-] No MDM rows returned.")

    manifest.record_step("mdm_extract", rows=mdm_df.height)

    # ── Step 5: Semantic Layers Normalization ─────────────────────────────
    _log("[*] Generating multi-layer data lake (context, measurements, features, model_input)...")

    # 1. Context
    uc_context = normalize_uc_context(feeder_cis_geo, run_id=manifest.run_id)
    write_context_table(uc_context, "uc_context", base_dir=output_dir, write_csv=True)

    meter_history = normalize_meter_installation_history(feeder_cis_geo, run_id=manifest.run_id)
    write_context_table(meter_history, "meter_installation_history", base_dir=output_dir, write_csv=True)

    hierarchy = normalize_electrical_hierarchy(feeder_cis_geo, run_id=manifest.run_id)
    write_context_table(hierarchy, "electrical_hierarchy", base_dir=output_dir, write_csv=True)

    # 2. Measurements
    interval_df, instant_df, register_df = normalize_mdm_dataframe(
        mdm_df, report_day=report_day, run_id=manifest.run_id
    )
    write_measurements_table(interval_df, "ami_interval", report_day, base_dir=output_dir, write_csv=True)
    write_measurements_table(instant_df, "ami_instantaneous", report_day, base_dir=output_dir, write_csv=True)
    write_measurements_table(register_df, "ami_registers", report_day, base_dir=output_dir, write_csv=True)

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

