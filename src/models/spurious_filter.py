"""Spurious Measurement and Telemetry Glitch Filtering.

Filters out noise, communication glitches, and grid-wide disturbances to prevent
false positives in anomaly classification:

1. Grid Event vs. UC Anomaly:
   If an entire cluster of UCs on the same transformer post (POSTO_OPERACIONAL)
   exhibits voltage drop or power failure simultaneously, this represents a network
   event (contingency/outage), not a local anomaly or meter fraud.

2. Isolated Telemetry Spikes:
   A single isolated 5-minute glitch in 30 days without persistence, especially
   correlated with low packet coverage, is flagged as SPURIOUS.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl


def filter_spurious_anomalies(
    scores_df: pl.DataFrame,
    neighborhood_df: pl.DataFrame | None = None,
    *,
    grid_event_threshold: float = 0.60,
) -> pl.DataFrame:
    """Filter raw anomaly scores to eliminate spurious telemetry and grid-level events.

    Args:
        scores_df: Table containing UC, anomaly_score, and is_anomaly.
        neighborhood_df: Optional table from compute_neighborhood_features.
        grid_event_threshold: Fraction of transformer peers that must be anomalous
                              to classify as a grid-wide event rather than UC theft.

    Returns:
        DataFrame with additional columns:
        - IS_SPURIOUS_GLITCH (Boolean)
        - IS_GRID_EVENT (Boolean)
        - FILTERED_ANOMALY_SCORE (Float in [0.0, 1.0])
        - FINAL_IS_ANOMALY (Boolean)
        - ANOMALY_CATEGORY (String: 'GENUINE_UC_ANOMALY', 'GRID_DISTURBANCE', 'TELEMETRY_NOISE', 'NORMAL')
    """
    if scores_df.is_empty():
        return scores_df

    uc_col = "id__uc_id" if "id__uc_id" in scores_df.columns else "UC"
    if uc_col not in scores_df.columns:
        return scores_df

    result = scores_df.clone()

    # Join neighborhood features if available
    if neighborhood_df is not None and not neighborhood_df.is_empty() and "UC" in neighborhood_df.columns:
        result = result.join(
            neighborhood_df.select([
                c for c in ["UC", "TRAFO_ID", "TRAFO_PEER_COUNT", "IS_ISOLATED_DEVIATION", "VOLTAGE_ZSCORE_IN_TRAFO"]
                if c in neighborhood_df.columns
            ]),
            left_on=uc_col,
            right_on="UC",
            how="left",
        )

    # 1. Identify grid-wide events (when most peers on the same trafo are anomalous)
    is_grid_event = np.zeros(result.height, dtype=bool)
    if "TRAFO_ID" in result.columns and "is_anomaly" in result.columns:
        trafos = result.partition_by("TRAFO_ID", as_dict=True)
        for _, t_df in trafos.items():
            if t_df.height >= 3:
                anom_rate = float(t_df.get_column("is_anomaly").mean())
                if anom_rate >= grid_event_threshold:
                    # Mark all UCs in this trafo as affected by grid event
                    trafo_ucs = set(t_df.get_column(uc_col).to_list())
                    mask = result.get_column(uc_col).is_in(trafo_ucs).to_numpy()
                    is_grid_event |= mask

    # 2. Identify transient noise / spurious glitches
    # An anomaly that is NOT isolated (peers moved together) or has negligible persistent deviation
    is_spurious = np.zeros(result.height, dtype=bool)
    if "IS_ISOLATED_DEVIATION" in result.columns:
        # If flagged by models but peers also deviated (not isolated) and peer count >= 3
        peer_count = result.get_column("TRAFO_PEER_COUNT").fill_null(1).to_numpy()
        isolated = result.get_column("IS_ISOLATED_DEVIATION").fill_null(True).to_numpy()
        is_spurious = (peer_count >= 3) & (~isolated) & (~is_grid_event)

    # Compute adjusted scores and categories
    base_scores = (
        result.get_column("anomaly_score").to_numpy()
        if "anomaly_score" in result.columns
        else (result.get_column("consensus_score").to_numpy() if "consensus_score" in result.columns else np.zeros(result.height))
    )
    base_is_anom = (
        result.get_column("is_anomaly").to_numpy()
        if "is_anomaly" in result.columns
        else (base_scores > 0.7)
    )

    # If grid event or spurious, reduce the anomaly score to avoid false priority
    filtered_scores = np.where(is_grid_event, base_scores * 0.35, base_scores)
    filtered_scores = np.where(is_spurious, filtered_scores * 0.40, filtered_scores)

    final_is_anomaly = base_is_anom & (~is_grid_event) & (~is_spurious)

    categories = []
    for is_anom, is_grid, is_spur in zip(base_is_anom, is_grid_event, is_spurious):
        if not is_anom:
            categories.append("NORMAL")
        elif is_grid:
            categories.append("GRID_DISTURBANCE")
        elif is_spur:
            categories.append("TELEMETRY_NOISE")
        else:
            categories.append("GENUINE_UC_ANOMALY")

    result = result.with_columns([
        pl.Series("IS_GRID_EVENT", is_grid_event),
        pl.Series("IS_SPURIOUS_GLITCH", is_spurious),
        pl.Series("FILTERED_ANOMALY_SCORE", np.round(filtered_scores, 4)),
        pl.Series("FINAL_IS_ANOMALY", final_is_anomaly),
        pl.Series("ANOMALY_CATEGORY", categories),
    ])

    return result
