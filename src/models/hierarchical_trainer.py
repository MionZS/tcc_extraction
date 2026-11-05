"""Hierarchical & Multi-Level Model Training Orchestrator.

Orchestrates training and evaluation across the electrical hierarchy:
- Feeder-level models (Local circuit experts)
- Municipality-level models (Regional experts)
- Global Copel model (Distributional anchor)

Integrates:
- CompositeAnomalyDetector (Multi-technique committee)
- MoERouter (Circuit-aware gating weights)
- SpuriousFilter (Transformer peer glitch elimination)
- Versioned Model Registry & Persistence
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
import joblib
import numpy as np
import polars as pl

from src.features.circuit_profile import compute_circuit_profile
from src.features.neighborhood import compute_neighborhood_features
from src.models.composite import CompositeAnomalyDetector
from src.models.moe_router import MoERouter
from src.models.spurious_filter import filter_spurious_anomalies


class HierarchicalModelOrchestrator:
    """Trains, saves, and scores models across Feeder, Municipality, and Global tiers."""

    def __init__(
        self,
        *,
        output_dir: Path | str = "output",
        version: str = "v1",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.version = version
        self.contamination = contamination
        self.random_state = random_state

        self.models_dir = self.output_dir / "models" / self.version
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.router = MoERouter()

    def train_and_score(
        self,
        training_df: pl.DataFrame,
        hierarchy_df: pl.DataFrame | None = None,
        *,
        target_feeder: str | None = None,
    ) -> pl.DataFrame:
        """Run hierarchical training, MoE consensus routing, and spurious filtering.

        Args:
            training_df: Complete training dataset (UC × cutoff_date).
            hierarchy_df: Optional electrical hierarchy table.
            target_feeder: Optional specific feeder to isolate and score.

        Returns:
            Scored DataFrame with individual model scores, consensus score,
            and spurious/grid event classification.
        """
        if training_df.is_empty():
            return pl.DataFrame()

        # Extract numeric feature columns
        feature_cols = [c for c in training_df.columns if c.startswith("x__") and training_df[c].dtype.is_numeric()]
        if not feature_cols:
            raise ValueError("No numeric features found with 'x__' prefix.")

        # 1. Global Model (fit strictly on clean telemetry windows when Quality Gate is present)
        has_clean_gate = "meta__is_clean_for_train" in training_df.columns
        clean_global_df = (
            training_df.filter(pl.col("meta__is_clean_for_train").fill_null(True))
            if has_clean_gate
            else training_df
        )
        if clean_global_df.height < 30:
            clean_global_df = training_df

        print(
            f"[*] Training Global Model on {clean_global_df.height:,} clean samples "
            f"(scoring {training_df.height:,} total)..."
        )
        X_global_fit = clean_global_df.select(feature_cols).to_numpy()
        X_global_all = training_df.select(feature_cols).to_numpy()
        global_detector = CompositeAnomalyDetector(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        global_detector.fit(X_global_fit)
        joblib.dump(global_detector, self.models_dir / "global_composite_model.joblib")

        global_preds = global_detector.predict_composite(X_global_all, raw_dataframe=training_df)
        global_scores = global_preds["consensus_score"]

        # 2. Feeder and Regional Models
        # Enrich dataset with hierarchy if columns not present or mostly null
        df = training_df.clone()

        # Drop any leftover _right columns to prevent Polars duplicate errors
        right_cols = [c for c in df.columns if c.endswith("_right")]
        if right_cols:
            df = df.drop(right_cols)

        if "meta__feeder" in df.columns:
            if "ALIMENTADOR" in df.columns:
                df = df.with_columns(pl.coalesce([pl.col("ALIMENTADOR"), pl.col("meta__feeder")]).alias("ALIMENTADOR"))
            else:
                df = df.with_columns(pl.col("meta__feeder").alias("ALIMENTADOR"))

        if hierarchy_df is not None and not hierarchy_df.is_empty():
            uc_col = "id__uc_id" if "id__uc_id" in df.columns else "UC"
            hier_uc_col = "UC" if "UC" in hierarchy_df.columns else "id__uc_id"

            df = df.with_columns(
                pl.col(uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias(uc_col)
            )
            h_norm = hierarchy_df.with_columns(
                pl.col(hier_uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_H_UC")
            ).filter(pl.col("_H_UC").is_not_null() & (pl.col("_H_UC") != ""))

            # Reconcile NIO -> UC if df holds NIOs in uc_col (legacy datasets)
            if "NIO" in h_norm.columns:
                known_ucs = set(h_norm.get_column("_H_UC").to_list())
                df_ids = set(df.get_column(uc_col).drop_nulls().to_list())
                if df_ids and len(df_ids & known_ucs) < len(df_ids) * 0.5:
                    nio_bridge = (
                        h_norm.select([
                            pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_H_NIO"),
                            pl.col("_H_UC"),
                        ])
                        .filter(pl.col("_H_NIO").is_not_null() & (pl.col("_H_NIO") != ""))
                        .unique(subset=["_H_NIO"], keep="last")
                    )
                    df = (
                        df.join(nio_bridge, left_on=uc_col, right_on="_H_NIO", how="left")
                        .with_columns(pl.coalesce([pl.col("_H_UC"), pl.col(uc_col)]).alias(uc_col))
                        .drop("_H_UC")
                    )

            hier_enrich_cols = [c for c in ["ALIMENTADOR", "MUNICIPIO_SE", "POSTO_OPERACIONAL"] if c in h_norm.columns]
            if hier_enrich_cols:
                h_sub = (
                    h_norm.select(["_H_UC"] + [pl.col(c).alias(f"_H_{c}") for c in hier_enrich_cols])
                    .unique(subset=["_H_UC"], keep="first")
                )
                df = df.join(h_sub, left_on=uc_col, right_on="_H_UC", how="left")
                for c in hier_enrich_cols:
                    if c in df.columns:
                        df = df.with_columns(pl.coalesce([pl.col(c), pl.col(f"_H_{c}")]).alias(c)).drop(f"_H_{c}")
                    else:
                        df = df.rename({f"_H_{c}": c})

        has_feeder_col = "ALIMENTADOR" in df.columns
        feeders = (
            [target_feeder] if target_feeder
            else (df.get_column("ALIMENTADOR").drop_nulls().unique().to_list() if has_feeder_col else [])
        )
        feeders = [f for f in feeders if f and str(f).strip() != ""]

        feeder_scores_map: dict[str, np.ndarray] = {}
        moe_weights_map: dict[str, dict[str, float]] = {}

        # Precompute circuit profiles if hierarchy available
        circuit_profiles_df = (
            compute_circuit_profile(df, hierarchy_df)
            if hierarchy_df is not None and not hierarchy_df.is_empty()
            else pl.DataFrame()
        )

        for f_name in feeders:
            if not f_name or str(f_name).strip() == "":
                continue
            f_mask = (df.get_column("ALIMENTADOR") == f_name).fill_null(False).to_numpy()
            f_count = int(f_mask.sum())
            if f_count < 30:
                continue

            feeder_sub_df = df.filter(pl.col("ALIMENTADOR") == f_name)
            feeder_fit_df = (
                feeder_sub_df.filter(pl.col("meta__is_clean_for_train").fill_null(True))
                if has_clean_gate
                else feeder_sub_df
            )
            if feeder_fit_df.height < 30:
                feeder_fit_df = feeder_sub_df

            print(
                f"[*] Training Feeder Specialist for '{f_name}' "
                f"({feeder_fit_df.height:,} clean / {f_count:,} total)..."
            )
            X_feeder_fit = feeder_fit_df.select(feature_cols).to_numpy()
            X_feeder_all = feeder_sub_df.select(feature_cols).to_numpy()
            feeder_detector = CompositeAnomalyDetector(
                contamination=self.contamination,
                random_state=self.random_state,
            )
            feeder_detector.fit(X_feeder_fit)

            safe_name = str(f_name).replace(" ", "_").replace("/", "_")
            joblib.dump(feeder_detector, self.models_dir / f"feeder_{safe_name}_composite_model.joblib")

            f_preds = feeder_detector.predict_composite(X_feeder_all, raw_dataframe=feeder_sub_df)
            feeder_scores_map[f_name] = f_preds["consensus_score"]

            # Compute circuit profile entry for gating
            c_prof = None
            if not circuit_profiles_df.is_empty():
                cp_row = circuit_profiles_df.filter(pl.col("ALIMENTADOR") == f_name)
                if not cp_row.is_empty():
                    c_prof = cp_row.to_dicts()[0]

            moe_weights_map[f_name] = self.router.compute_gating_weights(
                feeder_sample_count=f_count,
                circuit_profile=c_prof,
            )

        # 3. Combine with MoE Router
        final_scores = np.copy(global_scores)
        if has_feeder_col and feeder_scores_map:
            for f_name, f_scores in feeder_scores_map.items():
                f_mask = (df.get_column("ALIMENTADOR") == f_name).fill_null(False).to_numpy()
                g_sub_scores = global_scores[f_mask]
                weights = moe_weights_map.get(f_name, {"w_feeder": 0.6, "w_global": 0.4})

                combined_f = self.router.combine_predictions(
                    scores_feeder=f_scores,
                    scores_mun=None,
                    scores_global=g_sub_scores,
                    weights=weights,
                )
                final_scores[f_mask] = combined_f

        # 4. Calibrated thresholding per electrical feeder partition (on clean baseline windows)
        clean_mask_arr = (
            df.get_column("meta__is_clean_for_train").fill_null(True).to_numpy()
            if has_clean_gate and "meta__is_clean_for_train" in df.columns
            else np.ones(len(df), dtype=bool)
        )
        ref_global_scores = final_scores[clean_mask_arr] if int(clean_mask_arr.sum()) >= 30 else final_scores
        global_threshold = float(np.percentile(ref_global_scores, 100 * (1 - self.contamination)))
        is_anomaly_arr = np.zeros(len(df), dtype=bool)

        if has_feeder_col and feeders:
            for f_name in feeders:
                f_mask = (df.get_column("ALIMENTADOR") == f_name).fill_null(False).to_numpy()
                f_clean_mask = f_mask & clean_mask_arr
                f_cnt = int(f_mask.sum())
                if f_cnt >= 30:
                    ref_f_scores = final_scores[f_clean_mask] if int(f_clean_mask.sum()) >= 30 else final_scores[f_mask]
                    f_tau = float(np.percentile(ref_f_scores, 100 * (1 - self.contamination)))
                    is_anomaly_arr[f_mask] = (final_scores[f_mask] >= f_tau) & clean_mask_arr[f_mask]
                else:
                    is_anomaly_arr[f_mask] = (final_scores[f_mask] >= global_threshold) & clean_mask_arr[f_mask]

            # Any UCs without a matched feeder use the global threshold
            feeder_list = df.get_column("ALIMENTADOR").fill_null("").to_list()
            unassigned_mask = ~np.isin(feeder_list, feeders)
            if np.any(unassigned_mask):
                is_anomaly_arr[unassigned_mask] = (
                    (final_scores[unassigned_mask] >= global_threshold) & clean_mask_arr[unassigned_mask]
                )
        else:
            is_anomaly_arr = (final_scores >= global_threshold) & clean_mask_arr

        # Build output table
        id_cols = [c for c in ["id__uc_id", "meta__cutoff_date", "meta__window_days", "ALIMENTADOR"] if c in df.columns]
        result = df.select(id_cols) if id_cols else pl.DataFrame()

        result = result.with_columns([
            pl.Series("global_score", np.round(global_scores, 4)),
            pl.Series("anomaly_score", np.round(final_scores, 4)),
            pl.Series("is_anomaly", is_anomaly_arr),
            pl.Series("global_is_anomaly", final_scores >= global_threshold),
        ])

        # 5. Apply Spurious Measurement & Grid Disturbance Filter
        neighborhood_df = None
        if hierarchy_df is not None and not hierarchy_df.is_empty():
            neighborhood_df = compute_neighborhood_features(df, hierarchy_df)

        result_filtered = filter_spurious_anomalies(result, neighborhood_df=neighborhood_df)

        # Route incomplete RF Mesh telemetry windows to explicit TELECOM_QUARANTINE bucket
        if has_clean_gate and "meta__is_clean_for_train" in df.columns:
            clean_series = df.get_column("meta__is_clean_for_train").fill_null(True)
            result_filtered = result_filtered.with_columns([
                pl.Series("_IS_CLEAN", clean_series),
            ]).with_columns([
                pl.when(~pl.col("_IS_CLEAN"))
                .then(pl.lit(False))
                .otherwise(pl.col("FINAL_IS_ANOMALY"))
                .alias("FINAL_IS_ANOMALY"),
                pl.when(~pl.col("_IS_CLEAN"))
                .then(pl.lit("TELECOM_QUARANTINE"))
                .otherwise(pl.col("ANOMALY_CATEGORY"))
                .alias("ANOMALY_CATEGORY"),
            ]).drop("_IS_CLEAN")

        # 6. Save Scores
        scores_dir = (
            self.output_dir / "hierarchical_scores" / self.version
            if self.output_dir.name == "output"
            else self.output_dir / "output" / "hierarchical_scores" / self.version
        )
        scores_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_parquet = scores_dir / f"hierarchical_scores_{timestamp}.parquet"
        out_csv = scores_dir / f"hierarchical_scores_{timestamp}.csv"

        result_filtered.write_parquet(out_parquet)
        result_filtered.write_csv(out_csv, separator=";")
        print(f"[+] Saved hierarchical anomaly scores to {out_parquet.name}")

        return result_filtered


# Alias for backward compatibility
HierarchicalModelTrainer = HierarchicalModelOrchestrator
