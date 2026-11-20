import os
import sys
from pathlib import Path
import polars as pl
import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

base = Path("output")
print("=== INSPEÇÃO DAS FONTES DE DADOS ===")

# 1. Context Tables
context_dir = base / "context"
for f in ["meter_installation_history.parquet", "uc_context.parquet", "electrical_hierarchy.parquet"]:
    p = context_dir / f
    if p.exists():
        df = pl.read_parquet(p)
        print(f"\n[Context] {f}:")
        print(f"  Linhas: {len(df):,}")
        print(f"  Colunas: {df.columns}")
        if "UC" in df.columns:
            print(f"  UCs distintas: {df['UC'].n_unique():,}")
        if "NIO" in df.columns:
            print(f"  NIOs distintos: {df['NIO'].n_unique():,}")
        if "NUM_MEDIDOR" in df.columns:
            print(f"  Medidores distintos: {df['NUM_MEDIDOR'].n_unique():,}")
        # Date columns
        for c in df.columns:
            if any(k in c.upper() for k in ["DATA", "DATE", "UPDATED"]):
                try:
                    s = df[c].drop_nulls()
                    if len(s) > 0:
                        print(f"  {c}: min={s.min()}, max={s.max()}")
                except Exception as e:
                    pass

# 2. Daily Features
features_dir = base / "features" / "daily"
if features_dir.exists():
    feat_files = list(features_dir.glob("*.parquet"))
    print(f"\n[Features Daily] {features_dir}: {len(feat_files)} arquivos")
    dates = []
    for f in sorted(feat_files):
        name = f.stem
        try:
            d_str = name.replace("features_", "")
            d = datetime.date.fromisoformat(d_str)
            dates.append(d)
        except Exception:
            pass
    if dates:
        print(f"  Data min={min(dates)}, max={max(dates)}, total dias={len(dates)}")
        df_first = pl.read_parquet(features_dir / f"features_{min(dates)}.parquet")
        df_last = pl.read_parquet(features_dir / f"features_{max(dates)}.parquet")
        print(f"  Primeiro ({min(dates)}): {len(df_first):,} linhas, {df_first['id__uc_id'].n_unique():,} UCs")
        print(f"  Último ({max(dates)}): {len(df_last):,} linhas, {df_last['id__uc_id'].n_unique():,} UCs")
        print(f"  Colunas exemplo: {df_last.columns[:15]} ... (total {len(df_last.columns)})")

# 3. Directory structure of output/lake or output/raw
print("\n[Data Lake / Raw Directories]")
for root, dirs, files in os.walk(base):
    rel = Path(root).relative_to(base)
    # only show depth <= 3
    if len(rel.parts) <= 3 and any(f.endswith(".parquet") or f.endswith(".csv") for f in files):
        pq_count = sum(1 for f in files if f.endswith(".parquet"))
        csv_count = sum(1 for f in files if f.endswith(".csv"))
        print(f"  output/{rel}: {pq_count} parquets, {csv_count} csvs")

# 4. Check Datasets
train_dir = base / "datasets"
if train_dir.exists():
    for f in train_dir.glob("*.parquet"):
        df = pl.read_parquet(f)
        print(f"\n[Dataset] {f.name}: {len(df):,} linhas, cols={len(df.columns)}")
        for c in df.columns:
            if any(k in c.upper() for k in ["DATE", "DATA", "CUTOFF"]):
                try:
                    s = df[c].drop_nulls()
                    if len(s) > 0:
                        print(f"  {c}: min={s.min()}, max={s.max()}")
                except Exception:
                    pass
