import os
from pathlib import Path
import polars as pl

base = Path("output")
print("=== MEASUREMENTS LAKE ===")
meas_dir = base / "measurements"
if meas_dir.exists():
    for sub in meas_dir.iterdir():
        if sub.is_dir():
            files = list(sub.glob("**/*.parquet"))
            print(f"  {sub.name}: {len(files)} arquivos parquet")
            # find min and max date from path
            days = []
            for f in files:
                parts = f.parts
                y = [p.split("=")[1] for p in parts if p.startswith("report_year=")]
                m = [p.split("=")[1] for p in parts if p.startswith("report_month=")]
                d = [p.split("=")[1] for p in parts if p.startswith("report_day=")]
                if y and m and d:
                    days.append(f"{y[0]}-{m[0].zfill(2)}-{d[0].zfill(2)}")
            if days:
                print(f"    Datas: min={min(days)}, max={max(days)}, total dias={len(set(days))}")
                # Sample schema of the latest file
                latest_f = [f for f in files if max(days).replace("-", "/") in str(f).replace("=", "/").replace("report_year", "").replace("report_month", "").replace("report_day", "") or max(days).split("-")[2] in str(f)]
                if latest_f:
                    df = pl.read_parquet(latest_f[0])
                    print(f"    Schema ({latest_f[0].name}): {df.columns[:10]}... total {len(df.columns)}, linhas={len(df):,}")

print("\n=== PIPELINE CODE CHECK ===")
# Where does the pipeline look for features and training datasets?
# Let's inspect src/ or scripts/ to see how datasets/features are built
print("Checking model_input and hierarchical_scores:")
for p in [base / "model_input" / "v1", base / "hierarchical_scores" / "v1", base / "reports"]:
    if p.exists():
        for f in p.glob("*.*"):
            print(f"  {p.name}/{f.name}: size={f.stat().st_size} bytes")
