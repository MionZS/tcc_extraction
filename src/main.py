from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.export_manager import DEFAULT_PUBLISH_TARGET
from src.pipeline import run_daily_araucaria_pipeline, run_period_pipeline



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ARAUCARIA daily extraction pipeline: CIS → GEO → weather → MDM → join → publish.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                          # yesterday (default)\n"
            "  %(prog)s --days-back 3            # 3 days ago\n"
            "  %(prog)s --start-date 2026-06-01  # single day via period mode\n"
            "  %(prog)s --start-date 2026-06-01 --end-date 2026-06-10\n"
            "  %(prog)s --no-weather --no-publish\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory (raw/ and refined/ subfolders are created inside).",
    )

    # ── Single-day mode ────────────────────────────────────────────────────
    day_group = parser.add_argument_group("single-day mode")
    day_group.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="How many days back from today.  1 = yesterday (default).  "
             "Ignored when --start-date is given.",
    )

    # ── Period mode ────────────────────────────────────────────────────────
    period_group = parser.add_argument_group("period mode (--start-date / --end-date)")
    period_group.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Start date (YYYY-MM-DD).  Triggers period mode — pipeline runs "
             "for each day from start-date to end-date (or yesterday if "
             "--end-date is omitted).",
    )
    period_group.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="End date (YYYY-MM-DD).  Defaults to yesterday.",
    )

    # ── Pipeline tuning ────────────────────────────────────────────────────
    tune_group = parser.add_argument_group("batching / performance")
    tune_group.add_argument(
        "--mdm-batch-size",
        type=int,
        default=500,
        help="NIOs per MDM batch (default: 500).",
    )
    tune_group.add_argument(
        "--geo-batch-size",
        type=int,
        default=500,
        help="UCs per GEO batch (default: 500).",
    )
    tune_group.add_argument(
        "--fetch-size",
        type=int,
        default=1000,
        help="Rows fetched per round-trip (default: 1 000).",
    )

    # ── SQL files ──────────────────────────────────────────────────────────
    sql_group = parser.add_argument_group("SQL query paths")
    sql_group.add_argument(
        "--cis-sql",
        type=Path,
        default=Path("queries/cis_araucaria_ml_extract_lightweight_alt.sql"),
        help="CIS extraction query.",
    )
    sql_group.add_argument(
        "--geo-sql",
        type=Path,
        default=Path("queries/geo_ucs.sql"),
        help="GEO (UCs) extraction query.",
    )
    sql_group.add_argument(
        "--mdm-sql",
        type=Path,
        default=Path("queries/mdm_coluna.sql"),
        help="MDM / ORCA extraction query.",
    )
    sql_group.add_argument(
        "--timegrid-sql",
        type=Path,
        default=Path("queries/memoria_de_massa_nio_list.sql"),
        help="TIMEGRID (5-min) extraction query.",
    )

    # ── Step toggles ───────────────────────────────────────────────────────
    toggle_group = parser.add_argument_group("step toggles")
    toggle_group.add_argument(
        "--no-timegrid",
        action="store_true",
        help="Skip the TIMEGRID (5-min grade) step.",
    )
    toggle_group.add_argument(
        "--no-weather",
        action="store_true",
        help="Skip the WEATHER (Open-Meteo) step.",
    )
    toggle_group.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publishing the final CSV to OneDrive.",
    )

    # ── Publish / debug ────────────────────────────────────────────────────
    pub_group = parser.add_argument_group("publishing / debugging")
    pub_group.add_argument(
        "--publish-target",
        type=Path,
        default=DEFAULT_PUBLISH_TARGET,
        help=f"OneDrive target directory (default: {DEFAULT_PUBLISH_TARGET}).",
    )
    pub_group.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary batch parquet files (for debugging).",
    )
    pub_group.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="In period mode, continue to the next day even if one fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    publish_target: Path | None = None if args.no_publish else args.publish_target

    if args.start_date is not None:
        # ── Period mode ────────────────────────────────────────────────
        end_date = args.end_date or _yesterday()
        if args.start_date > end_date:
            print(f"Error: start-date ({args.start_date}) cannot be after end-date ({end_date})")
            return 1

        result = run_period_pipeline(
            start_date=args.start_date,
            end_date=end_date,
            output_dir=args.output_dir,
            mdm_batch_size=args.mdm_batch_size,
            geo_batch_size=args.geo_batch_size,
            fetch_size=args.fetch_size,
            cis_sql_path=args.cis_sql,
            geo_sql_path=args.geo_sql,
            mdm_sql_path=args.mdm_sql,
            timegrid_sql_path=None if args.no_timegrid else args.timegrid_sql,
            enable_weather=not args.no_weather,
            publish_target=publish_target,
            keep_temp=args.keep_temp,
            continue_on_error=args.continue_on_error,
        )

        print()
        print("═" * 45)
        print("  PERIOD SUMMARY")
        print("═" * 45)
        print(f"  Period     : {result.start_date.isoformat()}  →  {result.end_date.isoformat()}")
        print(f"  Days       : {result.total_days} total  ·  {result.success_days} ok  ·  {result.failed_days} failed")
        print(f"  Duration   : {result.duration_seconds:.1f}s")
        print("═" * 45)
        return 0 if result.failed_days == 0 else 1

    else:
        # ── Single-day mode ────────────────────────────────────────────
        result = run_daily_araucaria_pipeline(
            output_dir=args.output_dir,
            days_back=args.days_back,
            mdm_batch_size=args.mdm_batch_size,
            geo_batch_size=args.geo_batch_size,
            fetch_size=args.fetch_size,
            cis_sql_path=args.cis_sql,
            geo_sql_path=args.geo_sql,
            mdm_sql_path=args.mdm_sql,
            timegrid_sql_path=None if args.no_timegrid else args.timegrid_sql,
            enable_weather=not args.no_weather,
            publish_target=publish_target,
            keep_temp=args.keep_temp,
        )

        print()
        print("═" * 45)
        print("  PIPELINE SUMMARY")
        print("═" * 45)
        print(f"  Report day  :  {result.report_day.isoformat()}")
        print(f"  Weather     :  {'✓' if result.total_weather_rows else '–'}  {result.total_weather_rows:,} rows")
        print(f"  Timegrid    :  {'✓' if result.total_timegrid_rows else '–'}  {result.total_timegrid_rows:,} rows")
        print(f"  CIS rows    :  {result.total_cis_rows:,}")
        print(f"  GEO rows    :  {result.total_geo_rows:,}")
        print(f"  NIOs        :  {result.total_nios:,}")
        print(f"  MDM rows    :  {result.total_mdm_rows:,}")
        print("─" * 45)
        if result.joined_csv is not None:
            print(f"  Joined      :  {result.joined_csv}")
        if result.weather_csv is not None:
            print(f"  Weather CSV :  {result.weather_csv}")
        if result.timegrid_csv is not None:
            print(f"  Timegrid CSV:  {result.timegrid_csv}")
        print("═" * 45)

        return 0


def _yesterday() -> date:
    from datetime import timedelta
    return date.today() - timedelta(days=1)


if __name__ == "__main__":
    raise SystemExit(main())
