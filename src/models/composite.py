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
    """Normalize score array to [0.0, 1.0] range with 1st and 99th percentile clipping."""
    if len(scores) == 0:
        return np.array([], dtype=float)
    p1 = np.nanpercentile(scores, 1)
    p99 = np.nanpercentile(scores, 99)
    if p99 <= p1:
        return np.zeros_like(scores, dtype=float)
    clipped = np.clip(scores, p1, p99)
    return (clipped - p1) / (p99 - p1)


class RobustPCADetector(BaseEstimator, OutlierMixin):
    """Reconstruction error anomaly detector using Truncated SVD."""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> RobustPCADetector:
        k = min(self.n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
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
            # Rule A: Critical voltage sag / swell (< 180V or > 255V for nominal 220V)
            # Only test absolute voltage levels (mean, median, p05, p95), excluding IQR, ramp, imbalance, std, delta
            v_cols = [
                c for c in raw_dataframe.columns
                if ("VOLTAGE" in c.upper() or "U_L1" in c.upper() or "TENSAO" in c.upper())
                and not any(x in c.upper() for x in ("IQR", "RAMP", "IMBALANCE", "DELTA", "STD", "AUTOCORR"))
            ]
            for vc in v_cols:
                if raw_dataframe[vc].dtype.is_numeric():
                    v_arr = raw_dataframe[vc].fill_null(220.0).to_numpy()
                    rule_flags |= (v_arr < 180.0) | (v_arr > 255.0)

            # Rule B: Severe active reversal without justification
            rev_cols = [c for c in raw_dataframe.columns if "REVERSAL" in c.upper()]
            for rc in rev_cols:
                if raw_dataframe[rc].dtype.is_numeric():
                    r_arr = raw_dataframe[rc].fill_null(0.0).to_numpy()
                    rule_flags |= (r_arr > 0.8)

        # Weighted consensus
        consensus = (
            self.weights.get("iso_forest", 0.3) * s_iso
            + self.weights.get("lof", 0.25) * s_lof
            + self.weights.get("ecod", 0.25) * s_ecod
            + self.weights.get("pca", 0.2) * s_pca
        )

        # Physics rule bonus: physical violation elevates the consensus score
        consensus = np.clip(consensus + np.where(rule_flags, 0.20, 0.0), 0.0, 1.0)

        # Thresholding at (1.0 - contamination) percentile
        threshold = float(np.percentile(consensus, 100 * (1.0 - self.contamination)))
        is_anomaly = (consensus >= threshold) | rule_flags

        return {
            "consensus_score": np.round(consensus, 4),
            "iso_score": np.round(s_iso, 4),
            "lof_score": np.round(s_lof, 4),
            "ecod_score": np.round(s_ecod, 4),
            "pca_score": np.round(s_pca, 4),
            "rule_violation_flag": rule_flags,
            "is_anomaly": is_anomaly,
        }
