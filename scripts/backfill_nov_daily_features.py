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
    compute_clean_day_features_vectorized,
)
from src.datasets.partitioning import feature_partition_path
from src.datasets.writers import write_features

base_dir = Path("output")
uc_context = _read_table_flexible(base_dir / "context" / "uc_context.parquet")
meter_history = _read_table_flexible(base_dir / "context" / "meter_installation_history.parquet")
hierarchy = _read_table_flexible(base_dir / "context" / "electrical_hierarchy.parquet")

days_to_compute = [date(2026, 11, d) for d in range(13, 20)]
print(f"Backfilling daily features for {len(days_to_compute)} days: {days_to_compute[0]} to {days_to_compute[-1]}...")

for d in days_to_compute:
    t0 = time.time()
    out_part = feature_partition_path("v1", d, base_dir=base_dir)
    pq_file = out_part / "data.parquet"
    if pq_file.exists() and pq_file.stat().st_size > 1000:
        print(f"  [{d}] Já existe ({pq_file.stat().st_size:,} bytes). Pulando.")
        continue
    
    df_feat = compute_clean_day_features_vectorized(
        d,
        base_dir=base_dir,
        uc_context_df=uc_context,
        meter_history_df=meter_history,
        hierarchy_df=hierarchy,
    )
    if not df_feat.is_empty():
        write_features(
            df_feat,
            "v1",
            d,
            base_dir=base_dir,
            write_csv=False,
        )
        elapsed = time.time() - t0
        print(f"  [{d}] Gravados {len(df_feat):,} registros em {elapsed:.2f}s!")
    else:
        print(f"  [{d}] Vazio ou sem medições.")

print("Backfill de features diárias concluído com sucesso!")
