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

        # 1. Global Model
        print(f"[*] Training Global Model across all {training_df.height:,} samples...")
        X_global = training_df.select(feature_cols).to_numpy()
        global_detector = CompositeAnomalyDetector(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        global_detector.fit(X_global)
        joblib.dump(global_detector, self.models_dir / "global_composite_model.joblib")

        global_preds = global_detector.predict_composite(X_global, raw_dataframe=training_df)
        global_scores = global_preds["consensus_score"]

        # 2. Feeder and Regional Models
        # Enrich dataset with hierarchy if columns not present
        df = training_df.clone()
        if hierarchy_df is not None and not hierarchy_df.is_empty():
            h_sub = hierarchy_df.select([
                c for c in ["UC", "ALIMENTADOR", "MUNICIPIO_SE", "POSTO_OPERACIONAL"]
                if c in hierarchy_df.columns
            ]).unique(subset=["UC"])
            uc_col = "id__uc_id" if "id__uc_id" in df.columns else "UC"
            df = df.join(h_sub, left_on=uc_col, right_on="UC", how="left")

        has_feeder_col = "ALIMENTADOR" in df.columns
        feeders = (
            [target_feeder] if target_feeder
            else (df.get_column("ALIMENTADOR").drop_nulls().unique().to_list() if has_feeder_col else [])
        )

        feeder_scores_map: dict[str, np.ndarray] = {}
        moe_weights_map: dict[str, dict[str, float]] = {}

        # Precompute circuit profiles if hierarchy available
        circuit_profiles_df = (
            compute_circuit_profile(df, hierarchy_df)
            if hierarchy_df is not None and not hierarchy_df.is_empty()
            else pl.DataFrame()
        )

        for f_name in feeders:
            if not f_name:
                continue
            f_mask = (df.get_column("ALIMENTADOR") == f_name).to_numpy()
            f_count = int(f_mask.sum())
            if f_count < 30:
                continue

            print(f"[*] Training Feeder Specialist for '{f_name}' ({f_count:,} UCs)...")
            X_feeder = df.filter(pl.col("ALIMENTADOR") == f_name).select(feature_cols).to_numpy()
            feeder_detector = CompositeAnomalyDetector(
                contamination=self.contamination,
                random_state=self.random_state,
            )
            feeder_detector.fit(X_feeder)

            safe_name = str(f_name).replace(" ", "_").replace("/", "_")
            joblib.dump(feeder_detector, self.models_dir / f"feeder_{safe_name}_composite_model.joblib")

            f_preds = feeder_detector.predict_composite(X_feeder, raw_dataframe=df.filter(pl.col("ALIMENTADOR") == f_name))
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
                f_mask = (df.get_column("ALIMENTADOR") == f_name).to_numpy()
                g_sub_scores = global_scores[f_mask]
                weights = moe_weights_map.get(f_name, {"w_feeder": 0.6, "w_global": 0.4})

                combined_f = self.router.combine_predictions(
                    scores_feeder=f_scores,
                    scores_mun=None,
                    scores_global=g_sub_scores,
                    weights=weights,
                )
                final_scores[f_mask] = combined_f

        # 4. Build output table
        id_cols = [c for c in ["id__uc_id", "meta__cutoff_date", "meta__window_days", "ALIMENTADOR"] if c in df.columns]
        result = df.select(id_cols) if id_cols else pl.DataFrame()

        result = result.with_columns([
            pl.Series("global_score", np.round(global_scores, 4)),
            pl.Series("anomaly_score", np.round(final_scores, 4)),
            pl.Series("is_anomaly", final_scores >= float(np.percentile(final_scores, 100 * (1 - self.contamination)))),
        ])

        # 5. Apply Spurious Measurement & Grid Disturbance Filter
        neighborhood_df = None
        if hierarchy_df is not None and not hierarchy_df.is_empty():
            neighborhood_df = compute_neighborhood_features(df, hierarchy_df)

        result_filtered = filter_spurious_anomalies(result, neighborhood_df=neighborhood_df)

        # 6. Save Scores
        scores_dir = self.output_dir / "outputs" / "hierarchical_scores" / self.version
        scores_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_parquet = scores_dir / f"hierarchical_scores_{timestamp}.parquet"
        out_csv = scores_dir / f"hierarchical_scores_{timestamp}.csv"

        result_filtered.write_parquet(out_parquet)
        result_filtered.write_csv(out_csv, separator=";")
        print(f"[+] Saved hierarchical anomaly scores to {out_parquet.name}")

        return result_filtered
