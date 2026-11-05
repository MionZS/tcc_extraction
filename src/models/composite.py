"""Composite Multi-Technique Anomaly Detection Engine.

Combines complementary anomaly detection algorithms to avoid relying on a single technique:
1. Isolation Forest (Hyperplane isolation of multidimensional outliers)
2. Local Outlier Factor (LOF - Local density estimation for varying consumption clusters)
3. ECOD / Robust Mahalanobis Distance (Tail probability and covariance displacement)
4. Robust PCA / Truncated SVD (Projection reconstruction residual)
5. Domain / Physics Rules (PRODIST Module 8 critical voltage, chronic unmetered reverse power)

Outputs a normalized consensus score [0.0, 1.0] along with the breakdown of individual votes.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize score array to [0.0, 1.0] range with min and max."""
    if len(scores) == 0:
        return np.array([], dtype=float)
    s_min = float(np.nanmin(scores))
    s_max = float(np.nanmax(scores))
    if s_max <= s_min:
        return np.zeros_like(scores, dtype=float)
    return (scores - s_min) / (s_max - s_min)


class RobustPCADetector(BaseEstimator, OutlierMixin):
    """Reconstruction error anomaly detector using Truncated SVD."""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> RobustPCADetector:
        k = min(self.n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
        # If matrix is constant or has near zero variance, TruncatedSVD exp_var / full_var triggers divide-by-zero
        var_sum = float(np.sum(np.var(X, axis=0))) if X.size > 0 else 0.0
        if var_sum < 1e-9:
            self.svd_ = None
            return self
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.svd_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.svd_ is None:
            return np.zeros(len(X))
        X_projected = self.svd_.transform(X)
        X_reconstructed = self.svd_.inverse_transform(X_projected)
        # Residual Euclidean reconstruction error
        residuals = np.linalg.norm(X - X_reconstructed, axis=1)
        return residuals


class ECODDetector(BaseEstimator, OutlierMixin):
    """Empirical Cumulative Distribution Functions Outlier Detection (ECOD).

    Measures tail probabilities along all features independently.
    Fast, parameter-free, and highly interpretable.
    """

    def __init__(self):
        self.ecdf_samples_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> ECODDetector:
        self.ecdf_samples_ = np.sort(X, axis=0)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.ecdf_samples_ is None:
            return np.zeros(len(X))
        n_samples = len(self.ecdf_samples_)
        # Search position in sorted empirical distributions
        scores = np.zeros(len(X))
        for col_idx in range(X.shape[1]):
            col_vals = self.ecdf_samples_[:, col_idx]
            left_prob = (np.searchsorted(col_vals, X[:, col_idx], side="right") + 1.0) / (n_samples + 2.0)
            right_prob = (n_samples - np.searchsorted(col_vals, X[:, col_idx], side="left") + 1.0) / (n_samples + 2.0)
            tail_prob = np.minimum(left_prob, right_prob)
            scores += -np.log(tail_prob + 1e-12)
        return scores


class CompositeAnomalyDetector:
    """Ensemble of Isolation Forest, LOF, ECOD, Robust PCA, and Physical Rules."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        random_state: int = 42,
        weights: dict[str, float] | None = None,
    ):
        # Enforce contamination in valid scikit-learn range (0.0, 0.5]
        self.contamination = float(np.clip(contamination, 0.001, 0.5))
        self.random_state = random_state
        self.weights = weights or {
            "iso_forest": 0.30,
            "lof": 0.25,
            "ecod": 0.25,
            "pca": 0.20,
        }

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()

        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.ecod = ECODDetector()
        self.pca = RobustPCADetector(n_components=3, random_state=random_state)

        self.fitted_ = False

    def fit(self, X: np.ndarray) -> CompositeAnomalyDetector:
        """Fit all models on feature matrix X."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.fit_transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        # Dynamically bound LOF n_neighbors to avoid UserWarning when n_samples <= n_neighbors
        n_samples = len(X_scaled)
        effective_neighbors = min(20, max(1, n_samples - 1))
        self.lof.set_params(n_neighbors=effective_neighbors)

        self.iso_forest.fit(X_scaled)
        self.lof.fit(X_scaled)
        self.ecod.fit(X_scaled)
        self.pca.fit(X_scaled)

        self.fitted_ = True
        return self

    def predict_composite(
        self,
        X: np.ndarray,
        feature_names: Sequence[str] | None = None,
        raw_dataframe: pl.DataFrame | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute individual scores, physical violations, and consensus score.

        Returns dict containing:
        - 'consensus_score': array of float in [0.0, 1.0]
        - 'iso_score': array of float in [0.0, 1.0]
        - 'lof_score': array of float in [0.0, 1.0]
        - 'ecod_score': array of float in [0.0, 1.0]
        - 'pca_score': array of float in [0.0, 1.0]
        - 'rule_violation_flag': array of bool
        - 'is_anomaly': array of bool
        """
        if not self.fitted_:
            raise RuntimeError("CompositeAnomalyDetector must be fitted before predict.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.transform(X_clean)

        # 1. Isolation Forest (higher negative score_samples = more anomalous)
        s_iso = _min_max_normalize(-self.iso_forest.score_samples(X_scaled))

        # 2. LOF (higher negative score_samples = more anomalous)
        s_lof = _min_max_normalize(-self.lof.score_samples(X_scaled))

        # 3. ECOD
        s_ecod = _min_max_normalize(self.ecod.score_samples(X_scaled))

        # 4. PCA Reconstruction
        s_pca = _min_max_normalize(self.pca.score_samples(X_scaled))

        # 5. Physics / PRODIST rules
        rule_flags = np.zeros(len(X), dtype=bool)
        if raw_dataframe is not None:
            # Exclude metadata and static nominal ratings (e.g. TENSAO_BASE=127V, TENSAO_ALIMENTADOR=13.8kV, TENSAO_SE=69kV)
            # These are ratings/cadastre, NOT telemetry measurements.
            rating_cols = {
                "TENSAO_BASE", "X__TENSAO_BASE",
                "TENSAO_ALIMENTADOR", "X__TENSAO_ALIMENTADOR",
                "TENSAO_SE", "X__TENSAO_SE",
                "TENSAO_SEC", "X__TENSAO_SEC",
                "NOMINAL_VOLTAGE", "X__NOMINAL_VOLTAGE",
            }

            # Rule A: Critical voltage level violation (< 0.70 pu or > 1.25 pu)
            # Only test actual measurement series (e.g. U_L1_AVG_MEAN, VOLTAGE_MEAN, U_L1_MEDIAN, etc.)
            v_meas_cols = [
                c for c in raw_dataframe.columns
                if c.upper() not in rating_cols
                and ("VOLTAGE" in c.upper() or "U_L1" in c.upper() or "TENSAO" in c.upper())
                and not any(x in c.upper() for x in ("IQR", "RAMP", "IMBALANCE", "DELTA", "STD", "AUTOCORR", "RATIO", "COUNT"))
            ]

            # Determine nominal base voltage if available (defaults: 220V or 127V)
            v_base_arr = np.full(len(X), 220.0, dtype=float)
            for b_col in ("x__tensao_base", "TENSAO_BASE"):
                if b_col in raw_dataframe.columns and raw_dataframe[b_col].dtype.is_numeric():
                    b_vals = raw_dataframe[b_col].fill_null(220.0).to_numpy()
                    valid_b = (b_vals > 50.0) & (b_vals < 500.0)
                    v_base_arr[valid_b] = b_vals[valid_b]
                    break

            for vc in v_meas_cols:
                if raw_dataframe[vc].dtype.is_numeric():
                    valid_mask = raw_dataframe[vc].is_not_null().to_numpy()
                    v_arr = raw_dataframe[vc].to_numpy()
                    # Only apply per-unit sag/swell check if values resemble actual line volts (> 50V)
                    plausible_meas = (v_arr > 50.0) & valid_mask
                    if np.any(plausible_meas):
                        v_pu = np.where(plausible_meas, v_arr / np.maximum(v_base_arr, 50.0), 1.0)
                        crit_sag_swell = plausible_meas & ((v_pu < 0.70) | (v_pu > 1.25))
                        rule_flags |= crit_sag_swell

            # Rule B: Extreme voltage imbalance (> 20% imbalance)
            v_imb_cols = [c for c in raw_dataframe.columns if "VOLTAGE_IMBALANCE" in c.upper() or "IMBALANCE_MAX" in c.upper()]
            for vic in v_imb_cols:
                if raw_dataframe[vic].dtype.is_numeric():
                    imb_arr = raw_dataframe[vic].fill_null(0.0).to_numpy()
                    # Imbalance ratios > 0.20 indicate severe unbalance / open phase
                    rule_flags |= (imb_arr > 0.20)

            # Rule C: Severe unmetered active energy reversal without GD/solar profile
            # Only test ratio/fraction columns (0.0 to 1.0), never integer day counts (HIST_RA_REVERSAL_DAYS)
            rev_cols = [
                c for c in raw_dataframe.columns
                if "REVERSAL" in c.upper()
                and any(k in c.upper() for k in ("RATIO", "PROP", "PERCENT"))
                and not any(k in c.upper() for k in ("DAY", "COUNT", "DELTA"))
            ]
            for rc in rev_cols:
                if raw_dataframe[rc].dtype.is_numeric():
                    r_arr = raw_dataframe[rc].fill_null(0.0).to_numpy()
                    rule_flags |= (r_arr > 0.85)

        # Weighted consensus
        consensus = (
            self.weights.get("iso_forest", 0.3) * s_iso
            + self.weights.get("lof", 0.25) * s_lof
            + self.weights.get("ecod", 0.25) * s_ecod
            + self.weights.get("pca", 0.2) * s_pca
        )

        # Physics rule bonus: physical violation elevates the consensus score
        consensus = np.clip(consensus + np.where(rule_flags, 0.20, 0.0), 0.0, 1.0)

        # Thresholding at (1.0 - contamination) percentile strictly preserves requested contamination rate
        threshold = float(np.percentile(consensus, 100 * (1.0 - self.contamination)))
        is_anomaly = consensus >= threshold

        return {
            "consensus_score": np.round(consensus, 4),
            "iso_score": np.round(s_iso, 4),
            "lof_score": np.round(s_lof, 4),
            "ecod_score": np.round(s_ecod, 4),
            "pca_score": np.round(s_pca, 4),
            "rule_violation_flag": rule_flags,
            "is_anomaly": is_anomaly,
        }

    def score_samples(self, X: np.ndarray, raw_dataframe: pl.DataFrame | None = None) -> np.ndarray:
        """Return consensus anomaly scores in [0.0, 1.0]."""
        res = self.predict_composite(X, raw_dataframe=raw_dataframe)
        return res["consensus_score"]

    def predict(self, X: np.ndarray, raw_dataframe: pl.DataFrame | None = None) -> np.ndarray:
        """Return boolean anomaly predictions."""
        res = self.predict_composite(X, raw_dataframe=raw_dataframe)
        return res["is_anomaly"]


# Alias for backward compatibility
CompositeAnomalyModel = CompositeAnomalyDetector
