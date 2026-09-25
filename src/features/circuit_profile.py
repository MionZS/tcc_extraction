"""Extract statistical signatures and electrical profiles of feeders and circuits.

Aggregates UC-level features and physical topology (electrical_hierarchy)
to characterize the feeder's electrical nature:
- Voltage drop dispersion and feeder profile (delta V)
- Circuit load factor and consumption entropy
- Customer mix (Residential vs Commercial/Industrial vs Rural)
- Solar / Distributed Generation (GD) penetration (reverse energy frequency)
- Global phase unbalance of the feeder
- Transformer density (mean UCs per operational post)
"""

from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl


def _resolve_uc_col(df: pl.DataFrame) -> str | None:
    """Returns the UC identifier column name, handling both old and new naming conventions."""
    for candidate in ("UC", "id__uc_id"):
        if candidate in df.columns:
            return candidate
    return None


def compute_circuit_profile(
    features_df: pl.DataFrame,
    hierarchy_df: pl.DataFrame,
    *,
    feeder_col: str = "ALIMENTADOR",
) -> pl.DataFrame:
    """Compute circuit-level electrical and demographic statistical signatures.

    Args:
        features_df: UC-level feature table (must have 'UC' or 'id__uc_id' and electrical features).
        hierarchy_df: Electrical hierarchy table (must have 'UC' and 'ALIMENTADOR').
        feeder_col: Column identifying the feeder in features_df. Auto-detected if not found.

    Returns:
        DataFrame with 1 row per feeder containing circuit statistical metrics.
    """
    if features_df.is_empty() or hierarchy_df.is_empty():
        return pl.DataFrame(schema={
            "ALIMENTADOR": pl.String,
            "CIRCUIT_UC_COUNT": pl.Int64,
            "CIRCUIT_TRAFO_COUNT": pl.Int64,
            "CIRCUIT_UCS_PER_TRAFO_MEAN": pl.Float64,
            "CIRCUIT_VOLTAGE_MEAN": pl.Float64,
            "CIRCUIT_VOLTAGE_STD": pl.Float64,
            "CIRCUIT_VOLTAGE_IMBALANCE_MEAN": pl.Float64,
            "CIRCUIT_LOAD_FACTOR_MEAN": pl.Float64,
            "CIRCUIT_ACTIVE_REVERSAL_RATIO": pl.Float64,
            "CIRCUIT_SOLAR_GD_PENETRATION": pl.Float64,
            "CIRCUIT_RESIDENTIAL_RATIO": pl.Float64,
            "CIRCUIT_COMMERCIAL_RATIO": pl.Float64,
            "CIRCUIT_INDUSTRIAL_RATIO": pl.Float64,
            "CIRCUIT_CONSUMPTION_ENTROPY": pl.Float64,
        })

    # Auto-detect the feeder column if needed
    if feeder_col not in features_df.columns:
        for candidate in ("meta__feeder", "ALIMENTADOR"):
            if candidate in features_df.columns:
                feeder_col = candidate
                break

    # Determine the UC key column in both DataFrames
    feat_uc_col = _resolve_uc_col(features_df)
    hier_uc_col = _resolve_uc_col(hierarchy_df)

    # If the feeder is already in features_df, we may not need the hierarchy join
    if feeder_col in features_df.columns and feat_uc_col is None:
        merged = features_df
    elif feat_uc_col is None or hier_uc_col is None:
        return pl.DataFrame()
    else:
        # Build hierarchy subset with UC + optional metadata columns
        cols_to_keep = [hier_uc_col]
        for c in ["ALIMENTADOR", "POSTO_OPERACIONAL", "CLASSE_PRINCIPAL", "POT_INST_KVA"]:
            if c in hierarchy_df.columns:
                cols_to_keep.append(c)

        hier_sub = hierarchy_df.select(cols_to_keep).unique(subset=[hier_uc_col])

        # Rename hierarchy UC col to match features UC col if they differ
        if hier_uc_col != feat_uc_col:
            hier_sub = hier_sub.rename({hier_uc_col: feat_uc_col})

        merged = features_df.join(
            hier_sub,
            left_on=feat_uc_col,
            right_on=feat_uc_col,
            how="inner",
        )

    if merged.is_empty() or feeder_col not in merged.columns:
        # Try auto-detecting feeder col in merged result
        for candidate in ("meta__feeder", "ALIMENTADOR"):
            if candidate in merged.columns:
                feeder_col = candidate
                break
    if merged.is_empty() or feeder_col not in merged.columns:
        return pl.DataFrame()

    feeders = merged.get_column(feeder_col).drop_nulls().unique().to_list()
    rows: list[dict[str, Any]] = []

    for f_name in feeders:
        f_df = merged.filter(pl.col(feeder_col) == f_name)
        n_ucs = f_df.height
        if n_ucs == 0:
            continue

        # Transformer count and density
        trafo_col = "POSTO_OPERACIONAL" if "POSTO_OPERACIONAL" in f_df.columns else None
        if trafo_col:
            trafos = f_df.get_column(trafo_col).drop_nulls().unique()
            n_trafos = trafos.len()
            ucs_per_trafo = float(n_ucs / n_trafos) if n_trafos > 0 else float(n_ucs)
        else:
            n_trafos = 1
            ucs_per_trafo = float(n_ucs)

        # Voltage statistics — probe bare and x__-prefixed names
        def _col(*names: str) -> str | None:
            """Return first matching column in f_df, checking x__-prefix fallback."""
            for n in names:
                if n in f_df.columns:
                    return n
                xn = f"x__{n}"
                if xn in f_df.columns:
                    return xn
            return None

        v_mean_col = _col("U_L1_AVG_MEAN", "VOLTAGE_MEAN")
        if v_mean_col and f_df.get_column(v_mean_col).drop_nulls().len() > 0:
            v_vals = f_df.get_column(v_mean_col).drop_nulls().to_numpy()
            v_mean = float(np.mean(v_vals))
            v_std = float(np.std(v_vals))
        else:
            v_mean = 220.0
            v_std = 0.0

        # Voltage imbalance
        v_imb_col = _col("VOLTAGE_IMBALANCE_MAX", "VOLTAGE_IMBALANCE_MEAN")
        if v_imb_col and f_df.get_column(v_imb_col).drop_nulls().len() > 0:
            v_imb_vals = f_df.get_column(v_imb_col).drop_nulls().to_numpy()
            v_imb_mean = float(np.nanmean(v_imb_vals))
        else:
            v_imb_mean = 0.0

        # Load factor
        lf_col = _col("LOAD_FACTOR_MEAN", "LOAD_FACTOR")
        if lf_col and f_df.get_column(lf_col).drop_nulls().len() > 0:
            lf_vals = f_df.get_column(lf_col).drop_nulls().to_numpy()
            lf_mean = float(np.nanmean(lf_vals))
        else:
            lf_mean = 0.5

        # Reverse active energy (Solar/Microgeneration indication)
        rev_col = _col("RA_REVERSAL_DAYS", "RA_REVERSAL_RATIO")
        if rev_col and f_df.get_column(rev_col).drop_nulls().len() > 0:
            rev_vals = f_df.get_column(rev_col).drop_nulls().to_numpy()
            rev_ratio = float(np.mean(rev_vals > 0.0))
        else:
            rev_ratio = 0.0

        # Customer class mix
        classe_col = "CLASSE_PRINCIPAL" if "CLASSE_PRINCIPAL" in f_df.columns else None
        if classe_col and f_df.get_column(classe_col).drop_nulls().len() > 0:
            classes = f_df.get_column(classe_col).to_list()
            res_ratio = sum(1 for c in classes if c and "RESIDEN" in str(c).upper()) / n_ucs
            com_ratio = sum(1 for c in classes if c and "COMERC" in str(c).upper()) / n_ucs
            ind_ratio = sum(1 for c in classes if c and "INDUST" in str(c).upper()) / n_ucs
        else:
            res_ratio, com_ratio, ind_ratio = 0.85, 0.12, 0.03

        # Consumption entropy across UCs
        fa_sum_col = _col("FA_INTERVAL_SUM_MEDIAN", "FA_INTERVAL_SUM")
        if fa_sum_col and f_df.get_column(fa_sum_col).drop_nulls().len() > 1:
            fa_vals = f_df.get_column(fa_sum_col).drop_nulls().to_numpy()
            fa_positive = fa_vals[fa_vals > 0]
            if len(fa_positive) > 0:
                p = fa_positive / np.sum(fa_positive)
                entropy = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12))
            else:
                entropy = 0.0
        else:
            entropy = 0.5

        rows.append({
            "ALIMENTADOR": str(f_name),
            "CIRCUIT_UC_COUNT": int(n_ucs),
            "CIRCUIT_TRAFO_COUNT": int(n_trafos),
            "CIRCUIT_UCS_PER_TRAFO_MEAN": round(ucs_per_trafo, 2),
            "CIRCUIT_VOLTAGE_MEAN": round(v_mean, 2),
            "CIRCUIT_VOLTAGE_STD": round(v_std, 3),
            "CIRCUIT_VOLTAGE_IMBALANCE_MEAN": round(v_imb_mean, 3),
            "CIRCUIT_LOAD_FACTOR_MEAN": round(lf_mean, 3),
            "CIRCUIT_ACTIVE_REVERSAL_RATIO": round(rev_ratio, 4),
            "CIRCUIT_SOLAR_GD_PENETRATION": round(rev_ratio, 4),
            "CIRCUIT_RESIDENTIAL_RATIO": round(res_ratio, 3),
            "CIRCUIT_COMMERCIAL_RATIO": round(com_ratio, 3),
            "CIRCUIT_INDUSTRIAL_RATIO": round(ind_ratio, 3),
            "CIRCUIT_CONSUMPTION_ENTROPY": round(entropy, 4),
        })

    return pl.DataFrame(rows, infer_schema_length=None, strict=False)
