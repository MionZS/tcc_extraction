"""Run frozen OOT MultiView inference on any single cutoff date or validation date range.

Usage:
    # Score a single new date using the frozen model from latest_suite.json:
    python scripts/run_oot_inference.py --cutoff 2026-11-26

    # Score an entire validation range (e.g., all of November 1..19):
    python scripts/run_oot_inference.py --start-date 2026-11-01 --end-date 2026-11-19
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.retroactive_cleaner import (
    _read_table_flexible,
    build_sanitized_window_features_vectorized,
)
from src.eda.uc_timeseries import _list_dated_parquets
from src.models.category_reporting import apply_delimited_isolation_and_categories
from src.models.composite import select_pure_electrical_features
from src.utils.joblib_splitter import load_split_joblib

BASE_DIR = Path("output")
CONTEXT_DIR = BASE_DIR / "context"
DAY_FEAT_DIR = BASE_DIR / "features" / "uc_day_features" / "feature_set_version=v1"


def _load_all_daily_features() -> pl.DataFrame:
    dated = _list_dated_parquets(DAY_FEAT_DIR)
    if not dated:
        raise FileNotFoundError(f"No daily feature parquets found in {DAY_FEAT_DIR}")
    frames = [pl.read_parquet(p) for _, p in dated]
    return pl.concat(frames, how="diagonal_relaxed")


def score_cutoffs(
    cutoffs: list[date],
    *,
    oot_run_dir: Path,
    out_dir: Path,
    window_days: int = 30,
    coverage_threshold: float = 0.70,
    min_window_coverage: float = 0.60,
    min_reliable_days: int = 15,
) -> list[dict]:
    """Load the frozen OOT MultiView model and score one or more cutoff dates without retraining."""
    model_path = oot_run_dir / "frozen_multiview_model.joblib"
    pipeline = load_split_joblib(model_path)

    all_daily = _load_all_daily_features()
    uc_context = _read_table_flexible(CONTEXT_DIR / "uc_context.parquet")
    hierarchy = _read_table_flexible(CONTEXT_DIR / "electrical_hierarchy.parquet")
    meter_history = _read_table_flexible(CONTEXT_DIR / "meter_installation_history.parquet")
    alarm_path = BASE_DIR / "events" / "alarm_events.parquet"
    alarm_events = pl.read_parquet(alarm_path) if alarm_path.exists() else None

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for cutoff in cutoffs:
        win_df = build_sanitized_window_features_vectorized(
            all_daily,
            cutoff_date=cutoff,
            window_days=window_days,
            meter_history_df=meter_history,
            alarm_events_df=alarm_events,
            min_day_coverage=coverage_threshold,
            min_window_coverage=min_window_coverage,
            min_reliable_days=min_reliable_days,
            feature_set_version="v1",
        )
        ds = build_training_dataset(
            win_df,
            uc_context_df=uc_context,
            hierarchy_df=hierarchy,
            split_policy="oot_eval",
        )
        feature_cols = [c for c in ds.columns if c.startswith("x__") and ds[c].dtype.is_numeric()]
        comm_cols = [
            c for c in feature_cols
            if any(k in c.lower() for k in ("coverage", "alarm_count", "latency", "gap", "reliable_day"))
        ]
        elec_cols = select_pure_electrical_features(ds)

        scored = pipeline.score(
            X_comm=ds.select(comm_cols).fill_null(0.0).to_numpy(),
            X_elec=ds.select(elec_cols).fill_null(0.0).to_numpy(),
            data_quality=ds["meta__data_quality"].to_numpy(),
            tamper_alarms=ds["x__alarm_count_window"].to_numpy() if "x__alarm_count_window" in ds.columns else np.zeros(ds.height),
            fa_trend_slopes=ds["x__trend_fa_slope"].to_numpy() if "x__trend_fa_slope" in ds.columns else None,
            meter_changed_30d=ds["x__meter_changed_30d"].to_numpy() if "x__meter_changed_30d" in ds.columns else None,
            raw_dataframe=ds,
        )
        enriched, _ = apply_delimited_isolation_and_categories(scored, base_dir=BASE_DIR)

        out_parquet = out_dir / f"oot_decisions_{cutoff.isoformat()}.parquet"
        enriched.write_parquet(out_parquet, compression="zstd")

        cov_ok = enriched["x__avg_interval_coverage"].fill_null(0.0) >= coverage_threshold
        summary = {
            "cutoff_date": cutoff.isoformat(),
            "total_ucs": int(enriched.height),
            "eligible_ucs": int(cov_ok.sum()),
            "quarantine_ucs": int((~cov_ok).sum()),
            "candidates_vistoria": int(enriched["IS_CANDIDATO_VISTORIA"].fill_null(False).sum()),
            "mean_score": round(float(enriched["ANOMALY_SCORE"].mean() or 0.0), 4),
            "p95_score": round(float(enriched["ANOMALY_SCORE"].quantile(0.95) or 0.0), 4),
            "output_parquet": str(out_parquet),
        }
        summary_rows.append(summary)
        print(
            f"[{cutoff.isoformat()}] UCs={summary['total_ucs']:,} | Elegiveis={summary['eligible_ucs']:,} "
            f"| Candidatos={summary['candidates_vistoria']:,} | Media={summary['mean_score']:.4f}"
        )

    summary_csv = out_dir / "oot_multiday_summary.csv"
    pl.DataFrame(summary_rows).write_csv(summary_csv, separator=";")
    print(f"Summary saved to {summary_csv}")
    return summary_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score new dates or a validation range with the frozen OOT MultiView model.")
    parser.add_argument("--cutoff", type=str, default=None, help="Single cutoff date (YYYY-MM-DD).")
    parser.add_argument("--start-date", type=str, default=None, help="Start date for range inference (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="End date for range inference (YYYY-MM-DD).")
    parser.add_argument("--oot-run-dir", type=str, default=None, help="Override frozen OOT run directory.")
    parser.add_argument("--out-dir", type=str, default="output/reports/oot_inference", help="Output directory for scored parquets.")
    args = parser.parse_args(argv)

    if args.oot_run_dir:
        oot_run_dir = Path(args.oot_run_dir)
    else:
        suite = json.loads(Path("experiments/runs/latest_suite.json").read_text(encoding="utf-8"))
        oot_run_dir = Path(suite["oot_multiview"])

    cutoffs: list[date] = []
    if args.cutoff:
        cutoffs = [date.fromisoformat(args.cutoff)]
    elif args.start_date and args.end_date:
        cur = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)
        while cur <= end:
            cutoffs.append(cur)
            cur += timedelta(days=1)
    else:
        parser.error("Provide --cutoff YYYY-MM-DD or both --start-date and --end-date.")

    score_cutoffs(cutoffs, oot_run_dir=oot_run_dir, out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
