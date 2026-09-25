"""Dataset Splitter Module.

Enforces strict separation of Train, Validation, and Test datasets.
Supports:
1. Feeder Similarity Clustering: Groups feeders with similar electrical & demographic
   signatures (via `circuit_profile`).
2. Feeder-Level Group Isolation: If a group has >= 3 similar feeders, reserves 1 feeder
   exclusively for Validation and 1 for Test, using the remaining for Training.
3. Temporal / Stratified Fallback: For small feeder counts, performs an out-of-time (OOT)
   or stratified split to guarantee zero data leakage between splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.features.circuit_profile import compute_circuit_profile


@dataclass
class DatasetSplitResult:
    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    split_summary: dict[str, Any]


class DatasetSplitter:
    """Intelligent Dataset Splitter for Feeder-Level and UC-Level ML Modeling."""

    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.20,
        min_cluster_feeders: int = 3,
        random_state: int = 42,
    ) -> None:
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_cluster_feeders = min_cluster_feeders
        self.random_state = random_state

    def group_feeders_by_similarity(
        self,
        features_df: pl.DataFrame,
        hierarchy_df: pl.DataFrame,
        n_clusters: int | None = None,
    ) -> dict[str, list[str]]:
        """Groups feeders by electrical signature similarity.

        Returns:
            Dictionary mapping cluster ID string to list of feeder names.
        """
        profile_df = compute_circuit_profile(features_df, hierarchy_df)
        if profile_df.is_empty() or "ALIMENTADOR" not in profile_df.columns:
            return {}

        feeders = profile_df.get_column("ALIMENTADOR").to_list()
        if len(feeders) <= 1:
            return {"cluster_0": feeders}

        num_cols = [c for c in profile_df.columns if c != "ALIMENTADOR" and profile_df[c].dtype.is_numeric()]
        if not num_cols:
            return {"cluster_0": feeders}

        X = profile_df.select(num_cols).to_numpy()
        # Impute missing values if any
        X = np.nan_to_num(X, nan=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_clusters is None:
            # Rule of thumb: 1 cluster per 3-5 feeders, max len(feeders)//2
            n_clusters = max(1, min(len(feeders) // self.min_cluster_feeders, 5))

        if n_clusters == 1 or len(feeders) <= n_clusters:
            return {"cluster_0": feeders}

        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X_scaled)

        grouped: dict[str, list[str]] = {}
        for f_name, label in zip(feeders, labels):
            cid = f"cluster_{label}"
            grouped.setdefault(cid, []).append(f_name)

        return grouped

    def split(
        self,
        df: pl.DataFrame,
        hierarchy_df: pl.DataFrame | None = None,
        feeder_col: str = "meta__feeder",
        date_col: str = "meta__cutoff_date",
    ) -> DatasetSplitResult:
        """Splits `df` into Train, Validation, and Test sets based on feeder similarity & temporal rules.

        Feeder identification priority:
        1. ``meta__feeder`` column (injected by run_araucaria_full pipeline).
        2. ``feeder_col`` argument (custom column name).
        3. ``ALIMENTADOR`` column (raw GEO join residual).
        4. Join from ``hierarchy_df`` via UC key.

        Args:
            df: Input feature dataset.
            hierarchy_df: Optional electrical hierarchy for feeder mapping if not in df.
            feeder_col: Column specifying feeder name (default: ``meta__feeder``).
            date_col: Column specifying cutoff date for temporal fallback split.

        Returns:
            DatasetSplitResult containing train_df, val_df, test_df, and summary dict.
        """
        if df.is_empty():
            return DatasetSplitResult(
                train_df=df, val_df=df, test_df=df, split_summary={"status": "empty_input"}
            )

        # Check for feeder column availability
        actual_feeder_col = feeder_col if feeder_col in df.columns else (
            "ALIMENTADOR" if "ALIMENTADOR" in df.columns else None
        )

        # Join hierarchy if actual_feeder_col is not present but hierarchy is provided
        working_df = df
        if not actual_feeder_col and hierarchy_df is not None:
            # Detect UC key in both frames (support legacy "UC" and new "id__uc_id")
            feat_uc_col = next((c for c in ("UC", "id__uc_id") if c in df.columns), None)
            hier_uc_col = next((c for c in ("UC", "id__uc_id") if c in hierarchy_df.columns), None)
            if feat_uc_col and hier_uc_col and "ALIMENTADOR" in hierarchy_df.columns:
                h_sub = hierarchy_df.select([hier_uc_col, "ALIMENTADOR"]).unique(subset=[hier_uc_col])
                if hier_uc_col != feat_uc_col:
                    h_sub = h_sub.rename({hier_uc_col: feat_uc_col})
                working_df = df.join(h_sub, left_on=feat_uc_col, right_on=feat_uc_col, how="left")
                actual_feeder_col = "ALIMENTADOR"

        feeders = (
            working_df.get_column(actual_feeder_col).drop_nulls().unique().to_list()
            if actual_feeder_col and actual_feeder_col in working_df.columns
            else []
        )

        train_dfs: list[pl.DataFrame] = []
        val_dfs: list[pl.DataFrame] = []
        test_dfs: list[pl.DataFrame] = []

        feeder_assignments: dict[str, str] = {}

        if len(feeders) >= self.min_cluster_feeders and hierarchy_df is not None:
            # Group feeders by similarity
            grouped_clusters = self.group_feeders_by_similarity(working_df, hierarchy_df)

            for cid, cluster_feeders in grouped_clusters.items():
                c_df = working_df.filter(pl.col(actual_feeder_col).is_in(cluster_feeders))
                if len(cluster_feeders) >= self.min_cluster_feeders:
                    # Deterministic shuffle of cluster feeders
                    np.random.seed(self.random_state)
                    shuffled_feeders = list(cluster_feeders)
                    np.random.shuffle(shuffled_feeders)

                    # Reserve 1 for test, 1 for val, rest for train
                    test_f = [shuffled_feeders[0]]
                    val_f = [shuffled_feeders[1]]
                    train_f = shuffled_feeders[2:]

                    for f in test_f:
                        feeder_assignments[f] = "test"
                    for f in val_f:
                        feeder_assignments[f] = "val"
                    for f in train_f:
                        feeder_assignments[f] = "train"

                    test_dfs.append(c_df.filter(pl.col(actual_feeder_col).is_in(test_f)))
                    val_dfs.append(c_df.filter(pl.col(actual_feeder_col).is_in(val_f)))
                    train_dfs.append(c_df.filter(pl.col(actual_feeder_col).is_in(train_f)))
                else:
                    # Fallback to temporal or stratified split within this small cluster
                    tr, va, te = self._temporal_or_random_split(c_df, date_col)
                    train_dfs.append(tr)
                    val_dfs.append(va)
                    test_dfs.append(te)
                    for f in cluster_feeders:
                        feeder_assignments[f] = "mixed_split"
        else:
            # Global fallback temporal or random split
            tr, va, te = self._temporal_or_random_split(working_df, date_col)
            train_dfs.append(tr)
            val_dfs.append(va)
            test_dfs.append(te)

        res_train = pl.concat(train_dfs) if train_dfs else pl.DataFrame(schema=df.schema)
        res_val = pl.concat(val_dfs) if val_dfs else pl.DataFrame(schema=df.schema)
        res_test = pl.concat(test_dfs) if test_dfs else pl.DataFrame(schema=df.schema)

        total_rows = working_df.height
        summary = {
            "total_rows": total_rows,
            "train_rows": res_train.height,
            "train_ratio": round(res_train.height / total_rows, 4) if total_rows > 0 else 0.0,
            "val_rows": res_val.height,
            "val_ratio": round(res_val.height / total_rows, 4) if total_rows > 0 else 0.0,
            "test_rows": res_test.height,
            "test_ratio": round(res_test.height / total_rows, 4) if total_rows > 0 else 0.0,
            "feeder_assignments": feeder_assignments,
        }

        return DatasetSplitResult(
            train_df=res_train,
            val_df=res_val,
            test_df=res_test,
            split_summary=summary,
        )

    def _temporal_or_random_split(
        self, df: pl.DataFrame, date_col: str
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Splits `df` temporally if `date_col` exists, otherwise randomly."""
        n = df.height
        if n == 0:
            return df, df, df

        if date_col in df.columns and df[date_col].drop_nulls().len() > 0:
            sorted_df = df.sort(date_col)
            val_idx = int(n * (1.0 - self.val_ratio - self.test_ratio))
            test_idx = int(n * (1.0 - self.test_ratio))

            train_df = sorted_df.slice(0, val_idx)
            val_df = sorted_df.slice(val_idx, test_idx - val_idx)
            test_df = sorted_df.slice(test_idx, n - test_idx)
            return train_df, val_df, test_df
        else:
            # Shuffled random split
            shuffled = df.sample(fraction=1.0, shuffle=True, seed=self.random_state)
            val_idx = int(n * (1.0 - self.val_ratio - self.test_ratio))
            test_idx = int(n * (1.0 - self.test_ratio))

            train_df = shuffled.slice(0, val_idx)
            val_df = shuffled.slice(val_idx, test_idx - val_idx)
            test_df = shuffled.slice(test_idx, n - test_idx)
            return train_df, val_df, test_df
