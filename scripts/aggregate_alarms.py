from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from sqlalchemy import text
from src.db import create_engine

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate alarm counts across T_EVENT tables.")
    parser.add_argument("--start-date", type=str, default="20260701",
                        help="Start date in YYYYMMDD format (inclusive).")
    parser.add_argument("--end-date", type=str, default="20260930",
                        help="End date in YYYYMMDD format (inclusive).")
    parser.add_argument("--filter", type=str, default="",
                        help="Comma‑separated list of alarm names/codes to include. Empty means all.")
    parser.add_argument("--output-format", type=str, choices=["parquet", "csv", "both"], default="parquet",
                        help="Output format(s) for the aggregated table.")
    parser.add_argument("--summary-only", type=str, default="",
                        help="Skip DB extraction. Read an existing aggregate CSV/Parquet and output only the distinct alarm summary.")
    return parser.parse_args()

def date_range(start: str, end: str):
    start_dt = datetime.datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.datetime.strptime(end, "%Y%m%d")
    delta = datetime.timedelta(days=1)
    while start_dt <= end_dt:
        yield start_dt.strftime("%Y%m%d")
        start_dt += delta

def write_summary(df: pl.DataFrame, out_dir: Path, base_name: str) -> Path:
    """Write a distinct alarm summary CSV with total count across all days."""
    summary = (
        df.group_by("alarm_name")
        .agg(pl.col("cnt").sum().alias("total_count"))
        .sort("total_count", descending=True)
    )
    summary_path = out_dir / f"{base_name}_distinct_alarms.csv"
    summary.write_csv(summary_path)
    print(f"\nDistinct alarm summary written to {summary_path}")
    print(f"  {summary.height} distinct alarms")
    print(f"  Top 10:")
    for row in summary.head(10).iter_rows(named=True):
        print(f"    {row['total_count']:>8,}  {row['alarm_name']}")
    return summary_path

def main():
    args = parse_args()

    out_dir = Path("output/events")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --summary-only mode: read existing file, output distinct alarms, done.
    if args.summary_only:
        src_path = Path(args.summary_only)
        if not src_path.exists():
            print(f"File not found: {src_path}")
            return
        if src_path.suffix == ".parquet":
            df = pl.read_parquet(src_path)
        else:
            df = pl.read_csv(src_path, try_parse_dates=True)
        base_name = src_path.stem
        write_summary(df, out_dir, base_name)
        return

    # Normal extraction mode
    filter_list = [f.strip() for f in args.filter.split(",") if f.strip()] or None
    engine = create_engine("orca")
    rows = []
    skipped = []
    with engine.connect() as conn:
        for date_str in date_range(args.start_date, args.end_date):
            table_name = f"AMI.T_EVENT_{date_str}"
            sql = f"""
                SELECT CONTENT AS alarm_name,
                       COUNT(*) AS cnt,
                       TRUNC(ORIGIN_TIME) AS event_date
                FROM   {table_name}
                {"WHERE CONTENT IN (:filter_list)" if filter_list else ""}
                GROUP BY CONTENT, TRUNC(ORIGIN_TIME)
            """
            try:
                if filter_list:
                    bind_list = conn.connection.gettype("SYS.ODCIVARCHAR2LIST")()
                    bind_list.extend(filter_list)
                    result = conn.execute(text(sql), {"filter_list": bind_list})
                else:
                    result = conn.execute(text(sql))
                count = 0
                for alarm_name, cnt, event_date in result:
                    rows.append({
                        "alarm_name": alarm_name,
                        "cnt": cnt,
                        "event_date": event_date.date() if hasattr(event_date, "date") else event_date,
                    })
                    count += 1
                print(f"  ✓ {table_name}: {count} alarm groups")
            except Exception as e:
                if "ORA-00942" in str(e):
                    skipped.append(date_str)
                    print(f"  ✗ {table_name}: table not found, skipping")
                else:
                    raise
    if not rows:
        print("No alarm data found for the given range/filters.")
        return
    df = pl.DataFrame(rows)
    df = df.with_columns([
        pl.col("alarm_name").cast(pl.Utf8),
        pl.col("cnt").cast(pl.Int64),
        pl.col("event_date").cast(pl.Date),
    ])
    base_name = f"alarms_aggregate_{args.start_date}-{args.end_date}"
    if args.output_format in ("parquet", "both"):
        parquet_path = out_dir / f"{base_name}.parquet"
        df.write_parquet(parquet_path)
        print(f"Parquet output written to {parquet_path}")
    if args.output_format in ("csv", "both"):
        csv_path = out_dir / f"{base_name}.csv"
        df.write_csv(csv_path)
        print(f"CSV output written to {csv_path}")
    print(f"\nAggregated {df.height} rows covering {df.select(pl.col('alarm_name')).unique().height} distinct alarms.")
    if skipped:
        print(f"Skipped {len(skipped)} missing tables (first: {skipped[0]}, last: {skipped[-1]}).")

    # Always write distinct alarm summary
    write_summary(df, out_dir, base_name)

if __name__ == "__main__":
    main()
