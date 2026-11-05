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
from typing import Any, Sequence

import polars as pl


def build_training_dataset(
    uc_window_features_df: pl.DataFrame | None = None,
    uc_context_df: pl.DataFrame | None = None,
    hierarchy_df: pl.DataFrame | None = None,
    labels_df: pl.DataFrame | None = None,
    *,
    split_policy: str = "train",
    **kwargs: Any,
) -> pl.DataFrame:
    """Combine window features, context, hierarchy, and labels into a model-ready matrix."""
    if uc_window_features_df is None and "window_features_df" in kwargs:
        uc_window_features_df = kwargs["window_features_df"]
    if uc_window_features_df is None and "day_features_df" in kwargs:
        uc_window_features_df = kwargs["day_features_df"]

    if uc_window_features_df is None or uc_window_features_df.is_empty():
        return pl.DataFrame()

    df = uc_window_features_df

    # ── Identifiers and Meta ──────────────────────────────────────────────
    feature_set = kwargs.get("feature_set", "all").lower()

    exprs: list[pl.Expr] = [
        pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("id__uc_id"),
        pl.col("CUTOFF_DATE").cast(pl.Date).alias("meta__cutoff_date"),
        pl.col("WINDOW_DAYS").cast(pl.Int16).alias("meta__window_days"),
        pl.lit(split_policy).cast(pl.String).alias("meta__split"),
    ]

    # Data quality metadata indicator
    if "AVG_INTERVAL_COVERAGE" in df.columns:
        exprs.append(pl.col("AVG_INTERVAL_COVERAGE").cast(pl.Float64).alias("meta__data_quality"))
    elif "COVERAGE_RELIABLE_DAYS" in df.columns:
        exprs.append((pl.col("COVERAGE_RELIABLE_DAYS") / 30.0).clip(0.0, 1.0).cast(pl.Float64).alias("meta__data_quality"))
    else:
        exprs.append(pl.lit(1.0).alias("meta__data_quality"))

    # ── Feature mapping from window features ──────────────────────────────
    electrical_features = [
        "HIST_FA_SUM_MEDIAN",
        "HIST_FA_SUM_MAD",
        "HIST_RA_REVERSAL_DAYS",
        "HIST_RA_REVERSAL_RATIO_MAX",
        "HIST_LOAD_FACTOR_MEAN",
        "HIST_VOLTAGE_IMBALANCE_MAX",
        "HIST_CURRENT_IMBALANCE_MAX",
        "TREND_FA_SLOPE",
    ]
    communication_features = [
        "COVERAGE_RELIABLE_DAYS",
        "AVG_INTERVAL_COVERAGE",
        "ALARM_COUNT_WINDOW",
        "AVG_ALARM_LATENCY_SEC",
    ]
    tamper_features = [
        "ALARM_COUNT_WINDOW",
        "METER_CHANGED_30D",
        "METER_CHANGED_90D",
        "METER_AGE_DAYS",
        "TREND_FA_SLOPE",
    ]

    if feature_set == "electrical":
        target_cols = electrical_features
    elif feature_set == "communication":
        target_cols = communication_features
    elif feature_set == "tamper":
        target_cols = tamper_features
    else:  # "all"
        target_cols = [
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

    for col_name in target_cols:
        if col_name in df.columns:
            exprs.append(pl.col(col_name).alias(f"x__{col_name.lower()}"))

    # Preserve optional curated metadata and normalized ratios if produced by retroactive cleaner
    for extra_meta in ("IS_CLEAN_FOR_TRAIN", "TELECOM_STATUS", "RELIABLE_DAY_RATIO"):
        if extra_meta in df.columns:
            exprs.append(pl.col(extra_meta).alias(f"meta__{extra_meta.lower()}"))
        elif f"meta__{extra_meta.lower()}" in df.columns:
            exprs.append(pl.col(f"meta__{extra_meta.lower()}"))

    for extra_feat in ("FA_CV_WINDOW", "ZERO_CONSUMPTION_FLAG"):
        if extra_feat in df.columns:
            exprs.append(pl.col(extra_feat).alias(f"x__{extra_feat.lower()}"))
        elif f"x__{extra_feat.lower()}" in df.columns:
            exprs.append(pl.col(f"x__{extra_feat.lower()}"))

    base_model_df = df.select(exprs)

    # ── Reconcile NIO -> UC if id__uc_id contains unmapped NIO identifiers ─
    if hierarchy_df is not None and not hierarchy_df.is_empty():
        h_cols = {c.upper(): c for c in hierarchy_df.columns}
        if "UC" in h_cols and "NIO" in h_cols:
            known_ucs = set(
                hierarchy_df.get_column(h_cols["UC"])
                .drop_nulls()
                .cast(pl.String)
                .str.strip_chars()
                .str.replace(r"^0+", "")
                .to_list()
            )
            sample_ids = set(base_model_df.get_column("id__uc_id").drop_nulls().to_list())
            direct_overlap = len(sample_ids & known_ucs)
            if sample_ids and direct_overlap < len(sample_ids) * 0.5:
                nio_bridge = (
                    hierarchy_df.select([
                        pl.col(h_cols["NIO"]).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_BRIDGE_NIO"),
                        pl.col(h_cols["UC"]).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_BRIDGE_UC"),
                    ])
                    .drop_nulls()
                    .filter((pl.col("_BRIDGE_NIO") != "") & (pl.col("_BRIDGE_UC") != ""))
                    .unique(subset=["_BRIDGE_NIO"], keep="last")
                )
                base_model_df = (
                    base_model_df.join(nio_bridge, left_on="id__uc_id", right_on="_BRIDGE_NIO", how="left")
                    .with_columns(pl.coalesce([pl.col("_BRIDGE_UC"), pl.col("id__uc_id")]).alias("id__uc_id"))
                    .drop("_BRIDGE_UC")
                    .unique(subset=["id__uc_id", "meta__cutoff_date"], keep="last")
                )

    # ── Join Context Features (if provided) ───────────────────────────────
    if uc_context_df is not None and not uc_context_df.is_empty():
        ctx_cols = {c.upper(): c for c in uc_context_df.columns}
        if "UC" in ctx_cols:
            context_select: list[pl.Expr] = [
                pl.col(ctx_cols["UC"]).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("id__uc_id")
            ]
            for fld in ("TENSAO_BASE", "DEMANDA", "CONSUMO_ESTM", "CLASSE", "FASE_CIRCUITO"):
                if fld in ctx_cols:
                    context_select.append(pl.col(ctx_cols[fld]).alias(f"x__{fld.lower()}"))

            context_subset = uc_context_df.select(context_select).unique(subset=["id__uc_id"])
            base_model_df = base_model_df.join(context_subset, on="id__uc_id", how="left")

    # ── Join Electrical Hierarchy Context (if provided) ───────────────────
    if hierarchy_df is not None and not hierarchy_df.is_empty():
        hier_cols = {c.upper(): c for c in hierarchy_df.columns}
        if "UC" in hier_cols:
            hier_select: list[pl.Expr] = [
                pl.col(hier_cols["UC"]).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("id__uc_id")
            ]
            # Numeric features for ML
            for fld in ("POT_INST_KVA", "TENSAO_ALIMENTADOR", "TENSAO_SE"):
                if fld in hier_cols:
                    hier_select.append(pl.col(hier_cols[fld]).alias(f"x__{fld.lower()}"))
            # Meta columns for grouping / routing — never sent to estimator
            if "ALIMENTADOR" in hier_cols:
                hier_select.append(pl.col(hier_cols["ALIMENTADOR"]).cast(pl.String).alias("meta__feeder"))
            if "POSTO_OPERACIONAL" in hier_cols:
                hier_select.append(pl.col(hier_cols["POSTO_OPERACIONAL"]).cast(pl.String).alias("meta__posto"))
            if "SUBESTACAO" in hier_cols:
                hier_select.append(pl.col(hier_cols["SUBESTACAO"]).cast(pl.String).alias("meta__subestacao"))

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

