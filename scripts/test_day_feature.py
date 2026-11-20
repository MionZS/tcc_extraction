import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import time
from datetime import date
from pathlib import Path
import polars as pl

from src.datasets.retroactive_cleaner import (
    _read_table_flexible,
    _build_nio_to_uc_bridge,
    compute_clean_day_features_vectorized,
)
from src.datasets.partitioning import feature_partition_path

base_dir = Path("output")
uc_context = _read_table_flexible(base_dir / "context" / "uc_context.parquet")
meter_history = _read_table_flexible(base_dir / "context" / "meter_installation_history.parquet")
hierarchy = _read_table_flexible(base_dir / "context" / "electrical_hierarchy.parquet")

t0 = time.time()
test_day = date(2026, 11, 13)
print(f"Testing feature computation for {test_day}...")
df_feat = compute_clean_day_features_vectorized(
    test_day,
    base_dir=base_dir,
    uc_context_df=uc_context,
    meter_history_df=meter_history,
    hierarchy_df=hierarchy,
)
elapsed = time.time() - t0
print(f"Computed {len(df_feat):,} features in {elapsed:.2f}s!")
print("Columns:", df_feat.columns[:10])
