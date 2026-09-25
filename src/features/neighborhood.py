"""Compute electrical neighborhood deviations relative to local transformer peers.

Grain: UC × CUTOFF_DATE.
Compares each UC's voltage and consumption metrics against the mean/median
of all other UCs connected to the EXACT same operational transformer/post
(POSTO_OPERACIONAL in electrical_hierarchy).

If the entire transformer experiences a voltage sag, it is a grid-level event.
If only ONE UC exhibits a severe voltage depression or phase divergence while
its neighbors remain nominal, this isolates a local anomaly (potential theft,
meter malfunction, or branch fault).
"""

from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl


def _resolve_uc_col(df: pl.DataFrame) -> str | None:
    """Returns the UC identifier column name, handling both old and new naming conventions."""
    for candidate in ("id__uc_id", "UC"):
        if candidate in df.columns:
            return candidate
    return None


def compute_neighborhood_features(
    uc_features_df: pl.DataFrame,
    hierarchy_df: pl.DataFrame,
    *,
    trafo_col: str = "POSTO_OPERACIONAL",
    uc_col: str = "UC",
) -> pl.DataFrame:
    """Compute local transformer neighborhood relative deviations.

    Args:
        uc_features_df: UC-level feature table.
        hierarchy_df: Electrical hierarchy table (containing UC and POSTO_OPERACIONAL).
        trafo_col: Transformer post column name.
        uc_col: Consumer unit column name (auto-detected if not in frames).

    Returns:
        DataFrame with columns:
        - UC / id__uc_id
        - TRAFO_ID
        - TRAFO_PEER_COUNT
        - VOLTAGE_DELTA_TO_PEERS (V_uc - V_peers_mean)
        - VOLTAGE_ZSCORE_IN_TRAFO
        - CONSUMPTION_RATIO_TO_PEER_MEDIAN
        - IS_ISOLATED_DEVIATION (Boolean: True if UC deviates while peers are normal)
    """
    if uc_features_df.is_empty() or hierarchy_df.is_empty():
        return pl.DataFrame(schema={
            "UC": pl.String,
            "id__uc_id": pl.String,
            "TRAFO_ID": pl.String,
            "TRAFO_PEER_COUNT": pl.Int32,
            "VOLTAGE_DELTA_TO_PEERS": pl.Float64,
            "VOLTAGE_ZSCORE_IN_TRAFO": pl.Float64,
            "CONSUMPTION_RATIO_TO_PEER_MEDIAN": pl.Float64,
            "IS_ISOLATED_DEVIATION": pl.Boolean,
        })

    feat_uc_col = _resolve_uc_col(uc_features_df) or uc_col
    hier_uc_col = _resolve_uc_col(hierarchy_df) or uc_col

    if feat_uc_col not in uc_features_df.columns or hier_uc_col not in hierarchy_df.columns:
        return pl.DataFrame()

    if trafo_col not in hierarchy_df.columns:
        return pl.DataFrame()

    # Join hierarchy to get trafo assignment
    trafo_mapping = hierarchy_df.select([hier_uc_col, trafo_col]).drop_nulls().unique(subset=[hier_uc_col])
    if hier_uc_col != feat_uc_col:
        trafo_mapping = trafo_mapping.rename({hier_uc_col: feat_uc_col})

    merged = uc_features_df.join(trafo_mapping, left_on=feat_uc_col, right_on=feat_uc_col, how="inner")

    if merged.is_empty():
        return pl.DataFrame()

    # Probe voltage and consumption columns (with bare and x__ prefix fallback)
    def _col(*names: str) -> str | None:
        for n in names:
            if n in merged.columns:
                return n
            xn = f"x__{n}"
            if xn in merged.columns:
                return xn
        return None

    v_col = _col("HIST_VOLTAGE_IMBALANCE_MAX", "U_L1_AVG_MEAN", "VOLTAGE_MEAN", "U_L1_MEDIAN")
    c_col = _col("HIST_FA_SUM_MEDIAN", "FA_INTERVAL_SUM_MEDIAN", "FA_INTERVAL_SUM")

    trafos_dict = merged.partition_by(trafo_col, as_dict=True)
    rows: list[dict[str, Any]] = []

    for trafo_key, t_df in trafos_dict.items():
        t_id = trafo_key[0] if isinstance(trafo_key, tuple) else trafo_key
        peer_count = t_df.height

        if v_col and v_col in t_df.columns:
            v_vals = t_df.get_column(v_col).to_numpy()
            v_valid = v_vals[~np.isnan(v_vals)]
            t_v_mean = float(np.mean(v_valid)) if len(v_valid) > 0 else 220.0
            t_v_std = float(np.std(v_valid)) if len(v_valid) > 1 else 1.0
        else:
            t_v_mean = 220.0
            t_v_std = 1.0

        if c_col and c_col in t_df.columns:
            c_vals = t_df.get_column(c_col).to_numpy()
            c_valid = c_vals[~np.isnan(c_vals)]
            t_c_median = float(np.median(c_valid)) if len(c_valid) > 0 else 1.0
        else:
            t_c_median = 1.0

        for row in t_df.iter_rows(named=True):
            uc_val = str(row[feat_uc_col])
            uc_v = float(row.get(v_col, t_v_mean) or t_v_mean)
            uc_c = float(row.get(c_col, t_c_median) or t_c_median)

            delta_v = uc_v - t_v_mean
            zscore_v = delta_v / (t_v_std + 1e-6)
            c_ratio = uc_c / (t_c_median + 1e-6)

            # An isolated deviation occurs when this UC is far from peers (> 2.5 std devs)
            # within a group of at least 3 peers on the transformer
            is_isolated = bool(peer_count >= 3 and abs(zscore_v) > 2.5)

            rows.append({
                "UC": uc_val,
                "id__uc_id": uc_val,
                "TRAFO_ID": str(t_id),
                "TRAFO_PEER_COUNT": int(peer_count),
                "VOLTAGE_DELTA_TO_PEERS": round(delta_v, 3),
                "VOLTAGE_ZSCORE_IN_TRAFO": round(zscore_v, 3),
                "CONSUMPTION_RATIO_TO_PEER_MEDIAN": round(c_ratio, 3),
                "IS_ISOLATED_DEVIATION": is_isolated,
            })

    return pl.DataFrame(rows, infer_schema_length=None, strict=False)
