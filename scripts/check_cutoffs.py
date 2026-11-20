import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl
from pathlib import Path

df = pl.read_parquet("output/model_input/v1/training_dataset.parquet")
print("Cutoffs in training_dataset.parquet:")
print(df["meta__cutoff_date"].value_counts().sort("meta__cutoff_date"))

print("\nSplit by cutoff:")
print(df.group_by(["meta__cutoff_date", "meta__split"]).len().sort(["meta__cutoff_date", "meta__split"]))
