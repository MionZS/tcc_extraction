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
from datetime import date
from typing import Any
import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.features.circuit_profile import compute_circuit_profile

# Período Canônico Padrão do TCC Copel Araucária
CANONICAL_TRAIN_START = date(2026, 8, 1)
CANONICAL_TRAIN_END = date(2026, 10, 31)


@dataclass
class DatasetSplitResult:
    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    split_summary: dict[str, Any]


class DatasetSplitter:
    """Intelligent Dataset Splitter for Feeder-Level and UC-Level ML Modeling.

    Enforces the canonical 3-month training window: Agosto, Setembro e Outubro/2026.
    Subsequent months (Novembro/2026 onwards) are strictly reserved for Validation and Test
    without data leakage.
    """

    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.20,
        min_cluster_feeders: int = 3,
        canonical_train_start: date = CANONICAL_TRAIN_START,
        canonical_train_end: date = CANONICAL_TRAIN_END,
        random_state: int = 42,
    ) -> None:
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_cluster_feeders = min_cluster_feeders
        self.canonical_train_start = canonical_train_start
        self.canonical_train_end = canonical_train_end
        self.random_state = random_state

    def _extract_date_series(self, df: pl.DataFrame, date_col: str | None = None) -> pl.Series | None:
        """Extracts a normalized pl.Date series from df using candidate column names."""
        candidates = [date_col] if date_col else []
        candidates.extend(["meta__cutoff_date", "CUTOFF_DATE", "REPORT_DAY", "DATE", "report_day"])

        for c in candidates:
            if not c or c not in df.columns:
                continue
            col = df.get_column(c)
            if col.dtype == pl.Date:
                return col
            if col.dtype == pl.Datetime:
                return col.dt.date()
            if col.dtype == pl.String:
                try_iso = col.str.to_date("%Y-%m-%d", strict=False)
                if try_iso.drop_nulls().len() > 0:
                    return try_iso
                try_num = col.str.to_date("%Y%m%d", strict=False)
                if try_num.drop_nulls().len() > 0:
                    return try_num
            if col.dtype.is_numeric():
                try_num = col.cast(pl.String).str.to_date("%Y%m%d", strict=False)
                if try_num.drop_nulls().len() > 0:
                    return try_num

        # Check year/month/day columns
        if "meta__report_year" in df.columns and "meta__report_month" in df.columns and "meta__report_day" in df.columns:
            try:
                date_str = (
                    df["meta__report_year"].cast(pl.String)
                    + "-"
                    + df["meta__report_month"].cast(pl.String).str.zfill(2)
                    + "-"
                    + df["meta__report_day"].cast(pl.String).str.zfill(2)
                )
                return date_str.str.to_date("%Y-%m-%d", strict=False)
            except Exception:
                pass

        return None

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

        # Check for feeder column availability and populate from hierarchy if missing or mostly null
        working_df = df
        actual_feeder_col = feeder_col if feeder_col in working_df.columns else (
            "ALIMENTADOR" if "ALIMENTADOR" in working_df.columns else None
        )

        needs_feeder_enrichment = False
        if actual_feeder_col:
            valid_cnt = working_df.filter(
                pl.col(actual_feeder_col).is_not_null() & (pl.col(actual_feeder_col).cast(pl.String).str.strip_chars() != "")
            ).height
            if valid_cnt < working_df.height * 0.5:
                needs_feeder_enrichment = True
        else:
            needs_feeder_enrichment = True

        if needs_feeder_enrichment and hierarchy_df is not None and not hierarchy_df.is_empty():
            feat_uc_col = next((c for c in ("id__uc_id", "UC", "uc_id") if c in working_df.columns), None)
            hier_uc_col = next((c for c in ("UC", "id__uc_id", "uc_id") if c in hierarchy_df.columns), None)
            hier_f_col = next((c for c in ("ALIMENTADOR", "meta__feeder", "feeder") if c in hierarchy_df.columns), None)

            if feat_uc_col and hier_uc_col and hier_f_col:
                h_sub = (
                    hierarchy_df.select([hier_uc_col, hier_f_col])
                    .drop_nulls()
                    .with_columns([
                        pl.col(hier_uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_JOIN_UC"),
                        pl.col(hier_f_col).cast(pl.String).str.strip_chars().alias("_ENRICHED_FEEDER"),
                    ])
                    .filter(pl.col("_JOIN_UC") != "")
                    .unique(subset=["_JOIN_UC"])
                )

                working_df = (
                    working_df.with_columns(
                        pl.col(feat_uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_JOIN_UC")
                    )
                    .join(h_sub, on="_JOIN_UC", how="left")
                    .drop("_JOIN_UC")
                )

                if actual_feeder_col and actual_feeder_col in working_df.columns:
                    working_df = working_df.with_columns(
                        pl.coalesce([pl.col(actual_feeder_col), pl.col("_ENRICHED_FEEDER")]).alias("meta__feeder")
                    ).drop("_ENRICHED_FEEDER")
                else:
                    working_df = working_df.rename({"_ENRICHED_FEEDER": "meta__feeder"})
                actual_feeder_col = "meta__feeder"

        feeders = (
            working_df.get_column(actual_feeder_col).drop_nulls().unique().to_list()
            if actual_feeder_col and actual_feeder_col in working_df.columns
            else []
        )

        train_dfs: list[pl.DataFrame] = []
        val_dfs: list[pl.DataFrame] = []
        test_dfs: list[pl.DataFrame] = []

        feeder_assignments: dict[str, str] = {}

        # ── 1. CANONICAL TEMPORAL OUT-OF-TIME (OOT) SPLIT ────────────────────────
        # Treino Canônico Oficial: Agosto, Setembro e Outubro/2026 (01/08 a 31/10/2026)
        # Validação & Teste: Meses seguintes (> 31/10/2026, ex: Novembro em diante)
        date_series = self._extract_date_series(working_df, date_col)
        has_subsequent_dates = False
        if date_series is not None and date_series.drop_nulls().len() > 0:
            max_d = date_series.drop_nulls().max()
            if max_d is not None and max_d > self.canonical_train_end:
                has_subsequent_dates = True

        if has_subsequent_dates and date_series is not None:
            valid_dates_mask = date_series.is_not_null()
            train_mask = valid_dates_mask & (date_series >= self.canonical_train_start) & (date_series <= self.canonical_train_end)
            future_mask = valid_dates_mask & (date_series > self.canonical_train_end)

            res_train = working_df.filter(train_mask)
            future_df = working_df.filter(future_mask)

            n_fut = future_df.height
            if n_fut > 0:
                fut_dates = self._extract_date_series(future_df, date_col)
                sorted_fut = future_df.sort(fut_dates) if fut_dates is not None else future_df
                val_fraction = self.val_ratio / (self.val_ratio + self.test_ratio)
                val_n = max(1, int(round(n_fut * val_fraction))) if n_fut > 1 else 1
                res_val = sorted_fut.slice(0, val_n)
                res_test = sorted_fut.slice(val_n) if n_fut > 1 else sorted_fut
            else:
                res_val = pl.DataFrame(schema=working_df.schema)
                res_test = pl.DataFrame(schema=working_df.schema)

            total_rows = working_df.height
            summary = {
                "total_rows": total_rows,
                "train_rows": res_train.height,
                "train_ratio": round(res_train.height / total_rows, 4) if total_rows > 0 else 0.0,
                "val_rows": res_val.height,
                "val_ratio": round(res_val.height / total_rows, 4) if total_rows > 0 else 0.0,
                "test_rows": res_test.height,
                "test_ratio": round(res_test.height / total_rows, 4) if total_rows > 0 else 0.0,
                "feeder_assignments": {"all_feeders": "train_in_aug_oct"},
                "strategy": "canonical_3months_oot",
                "train_period": f"{self.canonical_train_start.strftime('%d/%m/%Y')} a {self.canonical_train_end.strftime('%d/%m/%Y')} (Agosto, Setembro e Outubro/2026)",
                "subsequent_period": f"Posterior a {self.canonical_train_end.strftime('%d/%m/%Y')} (Novembro/2026 em diante)",
                "has_subsequent_data": True,
            }

            return DatasetSplitResult(
                train_df=res_train,
                val_df=res_val,
                test_df=res_test,
                split_summary=summary,
            )

        # ── 1B. INTRA-CANONICAL MULTI-WINDOW OOT SPLIT (Before November Arrives) ─
        # When >= 3 decoupled cutoffs exist inside [01/08/2026, 31/10/2026] (e.g. Aug, Sep, Oct)
        # and November is not yet extracted, keep early cutoffs (e.g. Aug + Sep) in Train across
        # ALL feeders and reserve the latest cutoff(s) (e.g. Oct) strictly Out-of-Time for Val/Test.
        if date_series is not None:
            in_canon = date_series.drop_nulls().filter(
                (date_series.drop_nulls() >= self.canonical_train_start)
                & (date_series.drop_nulls() <= self.canonical_train_end)
            )
            unique_canon_dates = sorted(in_canon.unique().to_list())
            if len(unique_canon_dates) >= 3:
                holdout_cutoff = unique_canon_dates[-1]
                valid_dates_mask = date_series.is_not_null()
                train_mask = (
                    valid_dates_mask
                    & (date_series >= self.canonical_train_start)
                    & (date_series < holdout_cutoff)
                )
                holdout_mask = valid_dates_mask & (date_series >= holdout_cutoff)

                res_train = working_df.filter(train_mask)
                holdout_df = working_df.filter(holdout_mask)
                n_hold = holdout_df.height
                val_fraction = self.val_ratio / (self.val_ratio + self.test_ratio)
                val_n = max(1, int(round(n_hold * val_fraction))) if n_hold > 1 else 1
                res_val = holdout_df.slice(0, val_n)
                res_test = holdout_df.slice(val_n) if n_hold > 1 else holdout_df

                total_rows = working_df.height
                summary = {
                    "total_rows": total_rows,
                    "train_rows": res_train.height,
                    "train_ratio": round(res_train.height / total_rows, 4) if total_rows > 0 else 0.0,
                    "val_rows": res_val.height,
                    "val_ratio": round(res_val.height / total_rows, 4) if total_rows > 0 else 0.0,
                    "test_rows": res_test.height,
                    "test_ratio": round(res_test.height / total_rows, 4) if total_rows > 0 else 0.0,
                    "feeder_assignments": {"all_feeders": "multi_window_oot"},
                    "strategy": "intra_canonical_multi_window_oot",
                    "train_period": f"{unique_canon_dates[0].strftime('%d/%m/%Y')} a {unique_canon_dates[-2].strftime('%d/%m/%Y')} ({len(unique_canon_dates) - 1} janelas desacopladas)",
                    "subsequent_period": f"Janela OOT {holdout_cutoff.strftime('%d/%m/%Y')} (até chegada de Novembro/2026)",
                    "has_subsequent_data": False,
                }
                return DatasetSplitResult(
                    train_df=res_train,
                    val_df=res_val,
                    test_df=res_test,
                    split_summary=summary,
                )

        # ── 2. SPLIT POR SIMILARIDADE DE ALIMENTADORES (Fallback) ────────────────
        if actual_feeder_col and actual_feeder_col in working_df.columns and len(feeders) >= 2:
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
                    # Allocate cluster feeders across roles if not done yet
                    assigned_roles = set(feeder_assignments.values())
                    for f in cluster_feeders:
                        if "test" not in assigned_roles:
                            feeder_assignments[f] = "test"
                            test_dfs.append(c_df.filter(pl.col(actual_feeder_col) == f))
                            assigned_roles.add("test")
                        elif "val" not in assigned_roles:
                            feeder_assignments[f] = "val"
                            val_dfs.append(c_df.filter(pl.col(actual_feeder_col) == f))
                            assigned_roles.add("val")
                        else:
                            feeder_assignments[f] = "train"
                            train_dfs.append(c_df.filter(pl.col(actual_feeder_col) == f))
        else:
            # Global fallback temporal or random split
            tr, va, te = self._temporal_or_random_split(working_df, date_col)
            train_dfs.append(tr)
            val_dfs.append(va)
            test_dfs.append(te)

        # CRITICAL RECOVERY: Ensure 100% of rows from working_df are retained across splits
        assigned_train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame(schema=working_df.schema)
        assigned_val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame(schema=working_df.schema)
        assigned_test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame(schema=working_df.schema)

        total_assigned = assigned_train_df.height + assigned_val_df.height + assigned_test_df.height
        if total_assigned < working_df.height:
            uc_key = next((c for c in ("id__uc_id", "UC") if c in working_df.columns), None)
            if uc_key:
                assigned_keys = set(assigned_train_df[uc_key].to_list()) | set(assigned_val_df[uc_key].to_list()) | set(assigned_test_df[uc_key].to_list())
                unassigned_df = working_df.filter(~pl.col(uc_key).is_in(assigned_keys))
            else:
                unassigned_df = working_df.slice(total_assigned)

            if unassigned_df.height > 0:
                un_tr, un_va, un_te = self._temporal_or_random_split(unassigned_df, date_col)
                train_dfs.append(un_tr)
                val_dfs.append(un_va)
                test_dfs.append(un_te)

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
            "strategy": "canonical_feeder_clustering",
            "train_period": f"{self.canonical_train_start.strftime('%d/%m/%Y')} a {self.canonical_train_end.strftime('%d/%m/%Y')} (Agosto, Setembro e Outubro/2026)",
            "subsequent_period": "Aguardando extração de Novembro/2026 para Validação/Teste 100% OOT",
            "has_subsequent_data": False,
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
            val_idx = int(round(n * (1.0 - self.val_ratio - self.test_ratio)))
            test_idx = int(round(n * (1.0 - self.test_ratio)))

            train_df = sorted_df.slice(0, val_idx)
            val_df = sorted_df.slice(val_idx, test_idx - val_idx)
            test_df = sorted_df.slice(test_idx, n - test_idx)
            return train_df, val_df, test_df
        else:
            # Shuffled random split
            shuffled = df.sample(fraction=1.0, shuffle=True, seed=self.random_state)
            val_idx = int(round(n * (1.0 - self.val_ratio - self.test_ratio)))
            test_idx = int(round(n * (1.0 - self.test_ratio)))

            train_df = shuffled.slice(0, val_idx)
            val_df = shuffled.slice(val_idx, test_idx - val_idx)
            test_df = shuffled.slice(test_idx, n - test_idx)
            return train_df, val_df, test_df
