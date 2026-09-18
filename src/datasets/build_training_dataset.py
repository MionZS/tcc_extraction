"""Build the model_input/training_dataset table.

Projects all features onto the unified sample unit:
    UC × CUTOFF_DATE

Applies the strict prefix convention:
- id__*     : Identifiers (never sent to estimator)
- meta__*   : Cutoff, split, version metadata
- x__*      : Permitted model features
- y__*      : Ground-truth target labels (never included in X)
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import polars as pl


def build_training_dataset(
    uc_window_features_df: pl.DataFrame,
    uc_context_df: pl.DataFrame | None = None,
    hierarchy_df: pl.DataFrame | None = None,
    labels_df: pl.DataFrame | None = None,
    *,
    split_policy: str = "train",
) -> pl.DataFrame:
    """Combine window features, context, hierarchy, and labels into a model-ready matrix.

    Args:
        uc_window_features_df: Rolling window features for UCs.
        uc_context_df: Optional UC context table.
        hierarchy_df: Optional electrical hierarchy table.
        labels_df: Optional ground-truth labels table.
        split_policy: Split label for this partition ("train", "validation", "test").

    Returns:
        A Polars DataFrame conforming to the id__*, meta__*, x__*, y__* convention.
    """
    if uc_window_features_df.is_empty():
        return pl.DataFrame()

    df = uc_window_features_df

    # ── Identifiers and Meta ──────────────────────────────────────────────
    exprs: list[pl.Expr] = [
        pl.col("UC").cast(pl.String).alias("id__uc_id"),
        pl.col("CUTOFF_DATE").cast(pl.Date).alias("meta__cutoff_date"),
        pl.col("WINDOW_DAYS").cast(pl.Int16).alias("meta__window_days"),
        pl.lit(split_policy).cast(pl.String).alias("meta__split"),
    ]

    # ── Feature mapping from window features ──────────────────────────────
    window_feature_cols = [
        "COVERAGE_RELIABLE_DAYS",
        "AVG_INTERVAL_COVERAGE",
        "HIST_FA_SUM_MEDIAN",
        "HIST_FA_SUM_MAD",
        "HIST_RA_REVERSAL_DAYS",
        "HIST_RA_REVERSAL_RATIO_MAX",
        "HIST_LOAD_FACTOR_MEAN",
        "HIST_VOLTAGE_IMBALANCE_MAX",
        "HIST_CURRENT_IMBALANCE_MAX",
        "TREND_FA_SLOPE",
        "METER_AGE_DAYS",
        "METER_CHANGED_30D",
        "METER_CHANGED_90D",
        "ALARM_COUNT_WINDOW",
        "AVG_ALARM_LATENCY_SEC",
    ]

    for col_name in window_feature_cols:
        if col_name in df.columns:
            exprs.append(pl.col(col_name).alias(f"x__{col_name.lower()}"))

    base_model_df = df.select(exprs)

    # ── Join Context Features (if provided) ───────────────────────────────
    if uc_context_df is not None and not uc_context_df.is_empty():
        ctx_cols = {c.upper(): c for c in uc_context_df.columns}
        if "UC" in ctx_cols:
            context_select: list[pl.Expr] = [pl.col(ctx_cols["UC"]).cast(pl.String).alias("id__uc_id")]
            for fld in ("TENSAO_BASE", "DEMANDA", "CONSUMO_ESTM", "CLASSE", "FASE_CIRCUITO"):
                if fld in ctx_cols:
                    context_select.append(pl.col(ctx_cols[fld]).alias(f"x__{fld.lower()}"))

            context_subset = uc_context_df.select(context_select).unique(subset=["id__uc_id"])
            base_model_df = base_model_df.join(context_subset, on="id__uc_id", how="left")

    # ── Join Electrical Hierarchy Context (if provided) ───────────────────
    if hierarchy_df is not None and not hierarchy_df.is_empty():
        hier_cols = {c.upper(): c for c in hierarchy_df.columns}
        if "UC" in hier_cols:
            hier_select: list[pl.Expr] = [pl.col(hier_cols["UC"]).cast(pl.String).alias("id__uc_id")]
            for fld in ("POT_INST_KVA", "TENSAO_ALIMENTADOR", "TENSAO_SE"):
                if fld in hier_cols:
                    hier_select.append(pl.col(hier_cols[fld]).alias(f"x__{fld.lower()}"))

            hier_subset = hierarchy_df.select(hier_select).unique(subset=["id__uc_id"])
            base_model_df = base_model_df.join(hier_subset, on="id__uc_id", how="left")

    # ── Join Labels (if provided) ─────────────────────────────────────────
    if labels_df is not None and not labels_df.is_empty():
        lbl_cols = {c.upper(): c for c in labels_df.columns}
        if "ENTITY_ID" in lbl_cols and "LABEL_VALUE" in lbl_cols:
            label_subset = labels_df.select([
                pl.col(lbl_cols["ENTITY_ID"]).cast(pl.String).alias("id__uc_id"),
                pl.col(lbl_cols["LABEL_VALUE"]).cast(pl.Boolean).alias("y__target_irregularity"),
            ]).unique(subset=["id__uc_id"])
            base_model_df = base_model_df.join(label_subset, on="id__uc_id", how="left")
    else:
        # Default null target column for unsupervised / semi-supervised workflows
        base_model_df = base_model_df.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("y__target_irregularity")
        )

    return base_model_df

