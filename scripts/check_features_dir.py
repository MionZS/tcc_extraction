import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl

base = Path("output")
feat_dir = base / "features"
print("Checking output/features:")
if feat_dir.exists():
    for p in feat_dir.glob("**/*.*"):
        print(" ", p)
else:
    print("output/features does not exist!")

# Also check data/
data_dir = Path("data")
print("Checking data/ directory:")
if data_dir.exists():
    for p in data_dir.glob("**/*.*"):
        print(" ", p)
else:
    print("data/ does not exist")
