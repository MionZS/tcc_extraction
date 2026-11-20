import os
from pathlib import Path
import polars as pl

base = Path("output")

print("Checking output/raw/ORCA/all_feeders:")
orca_feeders = list((base / "raw" / "ORCA" / "all_feeders").glob("*.parquet"))
print(f"Total files: {len(orca_feeders)}")
for f in sorted(orca_feeders)[:5]:
    print(" ", f.name)
for f in sorted(orca_feeders)[-5:]:
    print(" ", f.name)

# inspect schema and dates of one orca file
if orca_feeders:
    df = pl.read_parquet(orca_feeders[-1])
    print(f"\nSample {orca_feeders[-1].name}:")
    print("  Columns:", df.columns)
    print("  Shape:", df.shape)
    for c in df.columns:
        if any(k in c.upper() for k in ["DATE", "DATA", "TIME"]):
            print(f"  {c}: min={df[c].min()}, max={df[c].max()}")

# Check where features or lake are
print("\nSearching for any parquet with 'features' or 'register' or 'interval':")
for p in base.glob("**/*.parquet"):
    s = p.as_posix()
    if any(k in s for k in ["feature", "register", "interval", "ami", "classified"]):
        print("  Found:", s)
