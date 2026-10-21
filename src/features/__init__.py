"""Features module — derive scikit-learn-ready features from normalized datasets.

Implements the feature engineering pipeline defined in
docs/dataset_architecture_decision.md §5.5.
"""

from src.features.meter_day import build_meter_day_features
from src.features.electrical import (
    compute_voltage_imbalance,
    compute_current_imbalance,
    compute_load_factor,
    compute_ra_reversal_ratio,
)
from src.features.quality import (
    compute_coverage,
    compute_null_ratio,
)
from src.features.temporal_grid import (
    build_daily_time_grid,
    regularize_interval_telemetry,
)

__all__ = [
    "build_meter_day_features",
    "compute_voltage_imbalance",
    "compute_current_imbalance",
    "compute_load_factor",
    "compute_ra_reversal_ratio",
    "compute_coverage",
    "compute_null_ratio",
    "build_daily_time_grid",
    "regularize_interval_telemetry",
]
