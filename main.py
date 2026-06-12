from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_daily_araucaria_pipeline



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the daily ARAUCARIA CIS -> MDM extraction pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where CSV/Parquet outputs will be written.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Days back for the MDM query. With the current SQL, 1 = yesterday.",
    )
    parser.add_argument(
        "--mdm-batch-size",
        type=int,
        default=500,
        help="How many NIOs to send per MDM batch.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=1000,
        help="How many rows to fetch per round-trip while reading MDM results.",
    )
    parser.add_argument(
        "--cis-sql",
        type=Path,
        default=Path("queries/cis_araucaria_ml_extract_lightweight_alt.sql"),
        help="Path to the CIS SQL file.",
    )
    parser.add_argument(
        "--mdm-sql",
        type=Path,
        default=Path("queries/mdm_coluna.sql"),
        help="Path to the MDM SQL file.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary batch parquet files for debugging.",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()
    result = run_daily_araucaria_pipeline(
        output_dir=args.output_dir,
        days_back=args.days_back,
        mdm_batch_size=args.mdm_batch_size,
        fetch_size=args.fetch_size,
        cis_sql_path=args.cis_sql,
        mdm_sql_path=args.mdm_sql,
        keep_temp=args.keep_temp,
    )

    print()
    print("Summary")
    print("-------")
    print(f"Report day : {result.report_day.isoformat()}")
    print(f"CIS rows   : {result.total_cis_rows:,}")
    print(f"NIOs       : {result.total_nios:,}")
    print(f"MDM rows   : {result.total_mdm_rows:,}")
    print(f"CIS CSV    : {result.cis_csv}")
    if result.mdm_csv is not None:
        print(f"MDM CSV    : {result.mdm_csv}")
    if result.joined_csv is not None:
        print(f"Joined CSV : {result.joined_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
