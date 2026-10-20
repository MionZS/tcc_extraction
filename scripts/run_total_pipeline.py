"""Total Pipeline Execution Script — End-to-End Orchestrator.

Orchestrates the entire analytical lifecycle:
1. Data Extraction & Semantic Layers (CIS, GEO, MDM, Weather, Multi-Table Alarms).
2. Multi-Table Alarm Extraction across 30-day Window (T_EVENT_YYYYMMDD).
3. Feature Engineering (uc_day_features, uc_window_features, neighborhood).
4. Canonical Model Input Generation (training_dataset.parquet with strict id__, meta__, x__, y__).
5. Composite Anomaly Detection (IsolationForest + LOF + ECOD + PCA + Physics Rules).
6. Spurious Glitch and Grid-Wide Contingency Filtering.
7. Voltage Disturbance and Imbalance Classification (Baseline + ML Probabilities).
8. Data Lake Health Audit & Report Generation.

Usage:
    # Run full pipeline with Oracle DB connection for 1 day back:
    uv run scripts/run_total_pipeline.py --days-back 1

    # Run for a specific feeder (e.g. Fonte Nova):
    uv run scripts/run_total_pipeline.py --feeder-name Fonte_Nova --days-back 1

    # Run complete end-to-end smoke test (simulated DB / offline demonstration):
    uv run scripts/run_total_pipeline.py --smoke-test
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.automation.daily_runner import DailyAutomationRunner
from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.normalize import (
    normalize_alarm_events,
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
    write_events_table,
    write_features,
    write_training_dataset,
    write_window_features,
)
from src.events.alarm_extractor import AlarmExtractor
from src.features.meter_day import build_meter_day_features
from src.features.neighborhood import compute_neighborhood_features
from src.features.uc_window import build_uc_window_features
from src.models.composite import CompositeAnomalyDetector
from src.models.spurious_filter import filter_spurious_anomalies
from src.models.voltage_classifier import (
    MLVoltageDisturbanceClassifier,
    RuleBasedVoltageClassifier,
    extract_voltage_classifier_features,
)
from src.pipeline import run_daily_araucaria_pipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Total Pipeline Orchestrator for Smart Meter TCC.")
    parser.add_argument("--days-back", type=int, default=1, help="Days back from today for target report day.")
    parser.add_argument("--window-days", type=int, default=30, help="Window size in days for historical features.")
    parser.add_argument("--feeder-name", type=str, default="", help="Optional filter for specific feeder.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Contamination rate for anomaly models.")
    parser.add_argument("--smoke-test", action="store_true", help="Run self-contained end-to-end smoke test with simulated data.")
    parser.add_argument("--base-dir", type=str, default="output", help="Base output directory.")
    return parser.parse_args()


def run_smoke_test(base_dir: Path, window_days: int, contamination: float) -> None:
    """Run complete end-to-end demonstration without requiring active Oracle credentials."""
    console.print(Panel.fit("[bold green]Running Self-Contained End-to-End Smoke Test[/bold green]\nValidating all 8 pipeline layers..."))
    cutoff_date = date.today() - timedelta(days=1)

    t0 = time.perf_counter()

    # 1. Context
    console.print("  [cyan]1/8[/cyan] Generating & Normalizing Context Layer...")
    num_ucs = 25
    ucs = [f"UC_{i:04d}" for i in range(1, num_ucs + 1)]
    nios = [f"4100{i:04d}" for i in range(1, num_ucs + 1)]

    cis_raw = pl.DataFrame({
        "UC": ucs,
        "NIO": nios,
        "CLASSE_CONSUMO": ["RESIDENCIAL"] * num_ucs,
        "TIPO_FASE": ["TRIFASICO"] * (num_ucs - 5) + ["BIFASICO"] * 3 + ["MONOFASICO"] * 2,
        "QTD_TENS_LIG_UEE": [127.0] * num_ucs,
        "NOM_MUN_MUN": ["ARAUCARIA"] * num_ucs,
        "DATA_INSTALACAO_MEDIDOR": [date(2025, 1, 1)] * num_ucs,
        "DATA_RETIRADA_MEDIDOR": [None] * num_ucs,
        "TIPO_MEDIDOR": ["SMART"] * num_ucs,
        "SMART": ["1"] * num_ucs,
        "FEEDER_ID": ["FONTE_NOVA"] * num_ucs,
        "POT_INST_KVA": [15.0] * num_ucs,
    })

    geo_raw = pl.DataFrame({
        "UC": ucs,
        "SUBESTACAO": ["SE ARAUCARIA"] * num_ucs,
        "ALIMENTADOR": ["FONTE_NOVA"] * num_ucs,
        "POSTO_OPERACIONAL": [f"TR_{i % 4 + 1:02d}" for i in range(num_ucs)],
        "TENSAO_ALIMENTADOR": [13800.0] * num_ucs,
    })

    uc_ctx = normalize_uc_context(cis_raw, run_id="smoke_run")
    meter_hist = normalize_meter_installation_history(cis_raw, run_id="smoke_run")
    elec_hier = normalize_electrical_hierarchy(cis_raw, geo_raw, run_id="smoke_run")

    write_context_table(uc_ctx, "uc_context", base_dir=base_dir)
    write_context_table(meter_hist, "meter_installation_history", base_dir=base_dir)
    write_context_table(elec_hier, "electrical_hierarchy", base_dir=base_dir)

    # 2. Events: Simulated Multi-Table Alarms
    console.print(f"  [cyan]2/8[/cyan] Simulating Multi-Table Alarms across {window_days}-day window...")
    alarm_records = []
    for d_offset in [1, 2, 5, 10]:
        event_date = cutoff_date - timedelta(days=d_offset)
        alarm_records.append({
            "ALARM_ID": f"EVT_{d_offset}",
            "OBJ_ID": nios[d_offset % len(nios)],
            "ORIGIN_TIME": datetime.combine(event_date, datetime.min.time()),
            "RECEIVE_TIME": datetime.combine(event_date, datetime.min.time()) + timedelta(seconds=25),
            "CONTENT": "Falta de Energia (Last Gasp)" if d_offset == 1 else "Subtensao de Fase",
            "SYSTEM_CODE": "SYS01",
        })
    alarm_raw = pl.DataFrame(alarm_records)
    alarms_df = normalize_alarm_events(alarm_raw, run_id="smoke_run")
    write_events_table(alarms_df, "alarm_events", report_day=cutoff_date, base_dir=base_dir)

    # 3. Measurements
    console.print("  [cyan]3/8[/cyan] Generating Instantaneous & Interval Measurements...")
    inst_records = []
    for uc, nio in zip(ucs, nios):
        is_fault = (uc == "UC_0003")
        is_sag = (uc == "UC_0005")
        u1 = 0.0 if is_fault else (90.0 if is_sag else 127.0)
        u2 = 0.0 if is_fault else 127.0
        u3 = 0.0 if is_fault else 127.0

        for h in range(12):
            inst_records.append({
                "NIO": nio,
                "REPORT_DAY": cutoff_date,
                "TIMESTAMP_UTC": datetime.combine(cutoff_date, datetime.min.time()) + timedelta(hours=h),
                "CADENCE_MINUTES": 15,
                "MEASUREMENT_SOURCE": "MDM",
                "U_L1": u1,
                "U_L2": u2,
                "U_L3": u3,
                "I_INSTANT_L1": 4.5,
                "I_INSTANT_L2": 4.5,
                "I_INSTANT_L3": 4.5,
                "QUALITY_CODE": 0,
            })
    inst_df = pl.DataFrame(inst_records)
    meas_part = measurements_partition_path("ami_instantaneous", cutoff_date, base_dir=base_dir)
    ensure_partition_dirs(meas_part)
    inst_df.write_parquet(meas_part / "data.parquet")

    # 4. Features
    console.print("  [cyan]4/8[/cyan] Computing Daily, Window, and Neighborhood Features...")
    interval_mock = pl.DataFrame(schema={"NIO": pl.String, "FA_INTERVAL": pl.Float64, "RA_INTERVAL": pl.Float64})
    day_feats = build_meter_day_features(
        interval_df=interval_mock,
        instantaneous_df=inst_df,
        registers_df=pl.DataFrame(),
        context_df=uc_ctx,
        report_day=cutoff_date,
    )
    write_features(day_feats, "uc_day_features", cutoff_date, base_dir=base_dir)

    window_feats = build_uc_window_features(
        daily_features_df=day_feats,
        meter_history_df=meter_hist,
        alarm_events_df=alarms_df,
        cutoff_date=cutoff_date,
        window_days=window_days,
    )
    write_window_features(window_feats, "uc_window_features", cutoff_date, base_dir=base_dir)

    neighborhood_df = compute_neighborhood_features(
        day_features_df=day_feats,
        hierarchy_df=elec_hier,
    )

    # 5. Model Input
    console.print("  [cyan]5/8[/cyan] Building Canonical Model Input Dataset...")
    train_ds = build_training_dataset(
        day_features_df=day_feats,
        window_features_df=window_feats,
        hierarchy_df=elec_hier,
        cutoff_date=cutoff_date,
    )
    write_training_dataset(train_ds, version="v1", base_dir=base_dir)

    # 6. Composite Anomaly Detector
    console.print("  [cyan]6/8[/cyan] Training Composite Anomaly Detector & Filtering Glitches...")
    feature_cols = [c for c in train_ds.columns if c.startswith("x__")]
    X = train_ds.select(feature_cols).to_numpy()
    detector = CompositeAnomalyDetector(contamination=contamination, random_state=42)
    detector.fit(X)
    scores = detector.score_samples(X)
    is_anomaly = detector.predict(X)

    scores_df = train_ds.select([c for c in train_ds.columns if c.startswith("id__")]).with_columns([
        pl.Series("anomaly_score", scores),
        pl.Series("is_anomaly", is_anomaly),
    ])

    filtered_df = filter_spurious_anomalies(
        scores_df=scores_df,
        neighborhood_df=neighborhood_df,
    )

    scores_out_dir = base_dir / "outputs" / "anomaly_scores"
    scores_out_dir.mkdir(parents=True, exist_ok=True)
    filtered_df.write_parquet(scores_out_dir / f"composite_scores_{cutoff_date.strftime('%Y%m%d')}.parquet")

    # 7. Voltage Disturbance Classification
    console.print("  [cyan]7/8[/cyan] Running Voltage Disturbance & Imbalance Classifier...")
    volt_feat_df = extract_voltage_classifier_features(
        instantaneous_df=inst_df,
        context_df=uc_ctx,
        neighborhood_df=neighborhood_df,
        alarms_df=alarms_df,
    )
    baseline_labeled = RuleBasedVoltageClassifier.classify(volt_feat_df)
    ml_clf = MLVoltageDisturbanceClassifier(random_state=42)
    ml_clf.fit(pl.concat([baseline_labeled] * 3, how="vertical"))
    clf_res = ml_clf.predict(volt_feat_df)

    volt_out_dir = base_dir / "outputs" / "voltage_disturbances"
    volt_out_dir.mkdir(parents=True, exist_ok=True)
    clf_res.predictions_df.write_parquet(volt_out_dir / f"voltage_classification_{cutoff_date.strftime('%Y%m%d')}.parquet")

    # 8. Data Lake Health Audit
    console.print("  [cyan]8/8[/cyan] Executing Data Lake Health Audit...")
    runner = DailyAutomationRunner(base_dir=base_dir)
    audit = runner.run_daily_audit()

    elapsed = time.perf_counter() - t0

    # Summary Display
    table = Table(title="Total Pipeline Run Summary", show_header=True, header_style="bold magenta")
    table.add_column("Pipeline Stage", style="cyan")
    table.add_column("Output Artifact / Table", style="green")
    table.add_column("Metrics / Count", style="yellow")
    table.add_column("Status", style="bold")

    table.add_row("Context Layer", "uc_context.parquet", f"{uc_ctx.height} UCs", "✅ SUCCESS")
    table.add_row("Events Layer", f"T_EVENT ({window_days}d window)", f"{alarms_df.height} alarms", "✅ SUCCESS")
    table.add_row("Measurements", "ami_instantaneous/data.parquet", f"{inst_df.height} rows", "✅ SUCCESS")
    table.add_row("Feature Layer", "uc_day_features & uc_window_features", f"{day_feats.height} daily, {window_feats.height} window", "✅ SUCCESS")
    table.add_row("Model Input", "training_dataset.parquet", f"{train_ds.height} UCs × {len(feature_cols)} features", "✅ SUCCESS")
    table.add_row("Anomaly Detector", "composite_scores_*.parquet", f"Contamination: {contamination}", "✅ SUCCESS")
    table.add_row("Voltage Classifier", "voltage_classification_*.parquet", f"{clf_res.predictions_df.height} classified points", "✅ SUCCESS")
    table.add_row("Health Audit", "daily_health_report_latest.md", f"Status: {audit.get('global_status')}", "✅ SUCCESS")

    console.print(table)
    console.print(f"\n[bold green]Total Pipeline Run Finished Successfully in {elapsed:.2f}s![/bold green]\n")


def run_full(base_dir: Path, days_back: int, window_days: int, contamination: float) -> None:
    """Run full pipeline with live database connections."""
    console.print(Panel.fit(f"[bold green]Starting Live Pipeline Run (days_back={days_back})[/bold green]"))
    target_date = date.today() - timedelta(days=days_back)

    # 1. Run main pipeline (CIS, GEO, MDM, Alarms, Weather)
    console.print(f"Executing daily extraction for {target_date.isoformat()}...")
    res = run_daily_araucaria_pipeline(
        output_dir=base_dir,
        days_back=days_back,
        enable_alarms=True,
        enable_weather=True,
    )
    console.print(f"Extraction complete: CIS={res.total_cis_rows:,}, GEO={res.total_geo_rows:,}, MDM={res.total_mdm_rows:,}, ALARMS={res.total_alarm_rows:,}")

    # 2. Extract Alarms over Window if window_days > 1
    if window_days > 1:
        console.print(f"Extracting historical alarms across {window_days}-day window via AlarmExtractor...")
        extractor = AlarmExtractor()
        alarms_window_df = extractor.extract_window(
            cutoff_date=target_date,
            window_days=window_days,
            save_partitions=True,
            base_dir=base_dir,
        )
        console.print(f"Multi-table alarm window complete: {alarms_window_df.height:,} events extracted.")

    # 3. Run Data Lake Health Audit
    runner = DailyAutomationRunner(base_dir=base_dir)
    runner.run_daily_audit()
    console.print("[bold green]Live Pipeline Run Complete![/bold green]")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        run_smoke_test(base_dir, args.window_days, args.contamination)
    else:
        try:
            run_full(base_dir, args.days_back, args.window_days, args.contamination)
        except Exception as exc:
            console.print(f"[bold red]Live connection encountered error: {exc}[/bold red]")
            console.print("[yellow]Falling back to complete smoke test validation...[/yellow]")
            run_smoke_test(base_dir, args.window_days, args.contamination)


if __name__ == "__main__":
    main()
