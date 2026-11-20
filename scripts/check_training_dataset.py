import polars as pl
from pathlib import Path

p = Path("output/model_input/v1/training_dataset.parquet")
if p.exists():
    df = pl.read_parquet(p)
    print("training_dataset.parquet:")
    print("  Shape:", df.shape)
    print("  Columns:", df.columns[:20], "... total", len(df.columns))
    for c in df.columns:
        if any(k in c.upper() for k in ["DATE", "DATA", "CUTOFF", "WINDOW", "DAY"]):
            s = df[c].drop_nulls()
            if len(s) > 0:
                print(f"  {c}: min={s.min()}, max={s.max()}, unique={s.n_unique()}")
    if "id__uc_id" in df.columns:
        print("  id__uc_id unique:", df["id__uc_id"].n_unique())
