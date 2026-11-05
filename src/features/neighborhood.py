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
    uc_features_df: pl.DataFrame | None = None,
    hierarchy_df: pl.DataFrame | None = None,
    *,
    trafo_col: str = "POSTO_OPERACIONAL",
    uc_col: str = "UC",
    **kwargs: Any,
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
    if uc_features_df is None and "day_features_df" in kwargs:
        uc_features_df = kwargs["day_features_df"]
    if uc_features_df is None:
        uc_features_df = pl.DataFrame()
    if hierarchy_df is None:
        hierarchy_df = pl.DataFrame()

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

    # Drop any leftover _right columns to prevent Polars duplicate errors
    right_cols = [c for c in uc_features_df.columns if c.endswith("_right")]
    if right_cols:
        uc_features_df = uc_features_df.drop(right_cols)

    feat_uc_col = _resolve_uc_col(uc_features_df) or uc_col
    hier_uc_col = _resolve_uc_col(hierarchy_df) or uc_col

    if feat_uc_col not in uc_features_df.columns or hier_uc_col not in hierarchy_df.columns:
        return pl.DataFrame()

    if trafo_col not in hierarchy_df.columns and trafo_col not in uc_features_df.columns:
        return pl.DataFrame()

    # Join hierarchy to get trafo assignment if not already present
    alt_trafo_col = "meta__posto" if "meta__posto" in uc_features_df.columns else None
    if trafo_col in uc_features_df.columns and uc_features_df.get_column(trafo_col).drop_nulls().len() > 0:
        merged = uc_features_df
    elif alt_trafo_col and uc_features_df.get_column(alt_trafo_col).drop_nulls().len() > 0:
        merged = uc_features_df.with_columns(pl.col(alt_trafo_col).alias(trafo_col)).filter(pl.col(trafo_col).is_not_null())
    else:
        trafo_mapping = (
            hierarchy_df.select([
                pl.col(hier_uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_UC"),
                pl.col(trafo_col),
            ])
            .drop_nulls()
            .filter(pl.col("_NORM_UC") != "")
            .unique(subset=["_NORM_UC"])
        )
        merged = (
            uc_features_df.with_columns(
                pl.col(feat_uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NORM_UC")
            )
            .join(trafo_mapping, on="_NORM_UC", how="inner")
            .drop("_NORM_UC")
        )

    if merged.is_empty():
        return pl.DataFrame()

    cutoff_col = next((c for c in ("meta__cutoff_date", "CUTOFF_DATE", "REPORT_DAY") if c in merged.columns), None)

    # Probe voltage and consumption columns (with bare and x__ prefix fallback)
    def _col(*names: str) -> str | None:
        for n in names:
            if n in merged.columns:
                return n
            xn = f"x__{n.lower()}"
            if xn in merged.columns:
                return xn
            xnu = f"x__{n}"
            if xnu in merged.columns:
                return xnu
        return None

    v_col = _col("U_L1_AVG_MEAN", "VOLTAGE_MEAN", "U_L1_MEDIAN", "HIST_VOLTAGE_IMBALANCE_MAX")
    c_col = _col("HIST_FA_SUM_MEDIAN", "FA_INTERVAL_SUM_MEDIAN", "FA_INTERVAL_SUM")

    is_imbalance = bool(v_col and "IMBALANCE" in v_col.upper())
    default_v = 0.0 if is_imbalance else 220.0
    default_std = 0.05 if is_imbalance else 1.0

    part_cols = [trafo_col, cutoff_col] if cutoff_col else [trafo_col]
    trafos_dict = merged.partition_by(part_cols, as_dict=True)
    rows: list[dict[str, Any]] = []

    for trafo_key, t_df in trafos_dict.items():
        if isinstance(trafo_key, tuple):
            t_id = trafo_key[0]
            c_dt = trafo_key[1] if len(trafo_key) > 1 else None
        else:
            t_id = trafo_key
            c_dt = None
        peer_count = t_df.height

        v_vals = t_df.get_column(v_col).to_numpy() if v_col and v_col in t_df.columns else np.array([])
        c_vals = t_df.get_column(c_col).to_numpy() if c_col and c_col in t_df.columns else np.array([])

        for idx, row in enumerate(t_df.iter_rows(named=True)):
            uc_val = str(row[feat_uc_col])
            raw_v = row.get(v_col)
            uc_v = float(raw_v) if raw_v is not None and not (isinstance(raw_v, float) and np.isnan(raw_v)) else default_v

            raw_c = row.get(c_col)
            uc_c = float(raw_c) if raw_c is not None and not (isinstance(raw_c, float) and np.isnan(raw_c)) else 0.0

            # Exclude self to get true peer statistics
            if len(v_vals) > 1:
                peer_v = np.delete(v_vals, idx)
                peer_v_valid = peer_v[~np.isnan(peer_v)]
                t_v_mean = float(np.mean(peer_v_valid)) if len(peer_v_valid) > 0 else default_v
                t_v_std = float(np.std(peer_v_valid)) if len(peer_v_valid) > 1 else default_std
            else:
                t_v_mean = uc_v
                t_v_std = default_std

            if len(c_vals) > 1:
                peer_c = np.delete(c_vals, idx)
                peer_c_valid = peer_c[~np.isnan(peer_c)]
                t_c_median = float(np.median(peer_c_valid)) if len(peer_c_valid) > 0 else 1.0
            else:
                t_c_median = max(uc_c, 1.0)

            delta_v = uc_v - t_v_mean
            zscore_v = delta_v / (t_v_std + 1e-6)
            c_ratio = uc_c / (t_c_median + 1e-6)

            # An isolated deviation occurs when this UC deviates significantly from transformer peers
            # (z-score > 2.5 or large absolute difference: > 15V for voltage, > 0.08 for imbalance ratio)
            # within a transformer group of at least 3 peers
            crit_delta = 0.08 if is_imbalance else 15.0
            is_isolated = bool(peer_count >= 3 and (abs(zscore_v) > 2.5 or abs(delta_v) > crit_delta))

            row_out: dict[str, Any] = {
                "UC": uc_val,
                "id__uc_id": uc_val,
                "TRAFO_ID": str(t_id),
                "TRAFO_PEER_COUNT": int(peer_count),
                "VOLTAGE_DELTA_TO_PEERS": round(delta_v, 3),
                "VOLTAGE_ZSCORE_IN_TRAFO": round(zscore_v, 3),
                "CONSUMPTION_RATIO_TO_PEER_MEDIAN": round(c_ratio, 3),
                "IS_ISOLATED_DEVIATION": is_isolated,
            }
            if cutoff_col and c_dt is not None:
                row_out[cutoff_col] = c_dt
            rows.append(row_out)

    return pl.DataFrame(rows, infer_schema_length=None, strict=False)
