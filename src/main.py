from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.export_manager import DEFAULT_PUBLISH_TARGET
from src.pipeline import run_daily_araucaria_pipeline, run_period_pipeline



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ARAUCARIA CIS -> GEO -> MDM extraction pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory; raw/ and refined/ subfolders will be created under it.",
    )

    # Single day mode
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Days back for the MDM query. With the current SQL, 1 = yesterday. "
             "Ignored if --start-date is provided.",
    )

    # Period mode
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Start date (YYYY-MM-DD) for period execution. "
             "When provided, runs pipeline for each day from start-date to end-date.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="End date (YYYY-MM-DD) for period execution. Defaults to yesterday.",
    )

    parser.add_argument(
        "--mdm-batch-size",
        type=int,
        default=500,
        help="How many NIOs to send per MDM batch.",
    )
    parser.add_argument(
        "--geo-batch-size",
        type=int,
        default=500,
        help="How many UCs to send per GEO batch.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=1000,
        help="How many rows to fetch per round-trip while reading results.",
    )
    parser.add_argument(
        "--cis-sql",
        type=Path,
        default=Path("queries/cis_araucaria_ml_extract_lightweight_alt.sql"),
        help="Path to the CIS SQL file.",
    )
    parser.add_argument(
        "--geo-sql",
        type=Path,
        default=Path("queries/geo_ucs.sql"),
        help="Path to the GEO (UCs) SQL file.",
    )
    parser.add_argument(
        "--mdm-sql",
        type=Path,
        default=Path("queries/mdm_coluna.sql"),
        help="Path to the MDM SQL file.",
    )
    parser.add_argument(
        "--timegrid-sql",
        type=Path,
        default=Path("queries/memoria_de_massa_nio_list.sql"),
        help="Path to the TIMEGRID SQL file.",
    )
    parser.add_argument(
        "--no-timegrid",
        action="store_true",
        help="Disable the TIMEGRID (grade 5 min) step.",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Disable the WEATHER (Open-Meteo) step.",
    )
    parser.add_argument(
        "--publish-target",
        type=Path,
        default=DEFAULT_PUBLISH_TARGET,
        help="Target directory for publishing the final CSV. Default: OneDrive sample200.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Disable publishing to OneDrive.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary batch parquet files for debugging.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="In period mode, continue to next day even if one fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Normalize publish target
    publish_target: Path | None = None if args.no_publish else args.publish_target

    if args.start_date is not None:
        # Period mode
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
        print("Period Summary")
        print("--------------")
        print(f"Period      : {result.start_date.isoformat()} to {result.end_date.isoformat()}")
        print(f"Total days  : {result.total_days}")
        print(f"Success     : {result.success_days}")
        print(f"Failed      : {result.failed_days}")
        print(f"Duration    : {result.duration_seconds:.1f}s")
        return 0 if result.failed_days == 0 else 1
    else:
        # Single day mode
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
        print("Summary")
        print("-------")
        print(f"Report day : {result.report_day.isoformat()}")
        print(f"CIS rows   : {result.total_cis_rows:,}")
        print(f"GEO rows   : {result.total_geo_rows:,}")
        print(f"NIOs       : {result.total_nios:,}")
        print(f"MDM rows   : {result.total_mdm_rows:,}")
        print(f"WEATHER rows: {result.total_weather_rows:,}")
        print(f"TIMEGRID rows: {result.total_timegrid_rows:,}")
        print(f"CIS CSV    : {result.cis_csv}")
        if result.geo_csv is not None:
            print(f"GEO CSV    : {result.geo_csv}")
        if result.mdm_csv is not None:
            print(f"MDM CSV    : {result.mdm_csv}")
        if result.weather_csv is not None:
            print(f"WEATHER CSV: {result.weather_csv}")
        if result.timegrid_csv is not None:
            print(f"TIMEGRID CSV: {result.timegrid_csv}")
        if result.joined_csv is not None:
            print(f"Joined CSV : {result.joined_csv}")

        return 0


def _yesterday() -> date:
    from datetime import timedelta
    return date.today() - timedelta(days=1)


if __name__ == "__main__":
    raise SystemExit(main())
