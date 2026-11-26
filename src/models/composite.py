"""Composite Multi-Technique Anomaly Detector (PyOD-Grade Pure Electrical Ensemble).

Combines six complementary unsupervised anomaly detection families aligned with PyOD
(Zhao et al., JMLR 2019; Li et al., ICDM 2020 / TKDE 2022):
1. Isolation Forest (IForest — Global isolation trees & feature interactions)
2. Local Outlier Factor (LOF — Local k-NN density deviation within consumption tiers)
3. ECOD (Empirical Cumulative Distribution Outlier Detection — Non-parametric left/right tail probabilities)
4. COPOD (Copula-Based Outlier Detection — Empirical copula with skewness-corrected tail selection)
5. HBOS (Histogram-Based Outlier Score — Dynamic-bin univariate log-density estimation)
6. Robust PCA Reconstruction Error (Subspace correlation & linear manifold violation)
7. PRODIST Module 8 Physical Guardrails (Critical voltage sag/swell, phase unbalance, reverse active flow)

Also enforces strict subspace separation via `select_pure_electrical_features(...)`:
communication telemetry coverage (`x__avg_interval_coverage`, `x__coverage_reliable_days`) and
hardware lifecycle indicators (`x__meter_age_days`, `x__meter_changed_30d`) are strictly excluded
from the electrical anomaly feature matrix so packet loss never contaminates electrical anomaly scores.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import polars as pl
from scipy.stats import skew as scipy_skew
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import pyod  # type: ignore[import-untyped]
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False


# Substrings that belong to Communication or Hardware Lifecycle — NEVER to pure electrical anomaly scoring
NON_ELECTRICAL_FEATURE_TOKENS: tuple[str, ...] = (
    "coverage",
    "reliable_day",
    "null_ratio",
    "latency",
    "alarm",
    "meter_age",
    "meter_changed",
    "n_meters",
    "peer_median_coverage",
    "coverage_deficit",
)


def select_pure_electrical_features(df_or_cols: pl.DataFrame | Sequence[str]) -> list[str]:
    """Select strictly electrical features (Consumption, Voltage, Current, Load Factor, Imbalance, Reversal).

    Excludes all communication/coverage, alarm-latency, and meter-lifecycle variables so that
    unsupervised electrical anomaly detectors evaluate exclusively physical load and grid signals.
    """
    if isinstance(df_or_cols, pl.DataFrame):
        candidate_cols = [
            c for c in df_or_cols.columns
            if c.startswith("x__") and df_or_cols[c].dtype.is_numeric()
        ]
    else:
        candidate_cols = [c for c in df_or_cols if c.startswith("x__")]

    pure_elec = [
        c for c in candidate_cols
        if not any(tok in c.lower() for tok in NON_ELECTRICAL_FEATURE_TOKENS)
    ]
    return pure_elec if pure_elec else list(candidate_cols)


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize arbitrary score array to [0.0, 1.0] safely using batch min-max."""
    if len(scores) == 0:
        return scores
    s = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    if s_max - s_min < 1e-12:
        return np.zeros_like(s, dtype=float)
    return np.clip((s - s_min) / (s_max - s_min), 0.0, 1.0)


def _compute_norm_params(raw_scores: np.ndarray) -> dict[str, Any]:
    """Compute frozen training-set reference statistics for stable out-of-sample normalization."""
    s = np.nan_to_num(np.asarray(raw_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(s) == 0:
        return {
            "min": 0.0,
            "max": 1.0,
            "p01": 0.0,
            "p99": 1.0,
            "median": 0.0,
            "iqr": 1.0,
            "sorted_raw": np.array([0.0], dtype=float),
        }
    p01 = float(np.percentile(s, 1.0))
    p25 = float(np.percentile(s, 25.0))
    p50 = float(np.percentile(s, 50.0))
    p75 = float(np.percentile(s, 75.0))
    p99 = float(np.percentile(s, 99.0))
    p999 = float(np.percentile(s, 99.9))
    return {
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "p01": p01,
        "p99": p99,
        "p999": p999,
        "median": p50,
        "iqr": max(p75 - p25, 1e-9),
        "sorted_raw": np.sort(s),
    }


def _normalize_with_params(
    raw_scores: np.ndarray,
    params: dict[str, Any] | None,
    mode: str = "frozen_quantile",
) -> np.ndarray:
    """Normalize raw detector scores to [0.0, 1.0] without batch-to-batch drift."""
    s = np.nan_to_num(np.asarray(raw_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(s) == 0:
        return s
    if params is None or mode == "batch_minmax":
        return _min_max_normalize(s)

    if mode == "frozen_ecdf":
        sorted_ref = params["sorted_raw"]
        n_ref = max(1, len(sorted_ref))
        ranks = np.searchsorted(sorted_ref, s, side="right")
        return np.clip(ranks.astype(float) / float(n_ref), 0.0, 1.0)

    # Default: "frozen_quantile" (anchored at training P01 and P99, fallback to P99.9 then min/max if degenerate)
    lo = float(params["p01"])
    hi = float(params["p99"])
    if hi - lo < 1e-12:
        hi = float(params.get("p999", params["max"]))
    if hi - lo < 1e-12:
        lo = float(params["min"])
        hi = float(params["max"])
    if hi - lo < 1e-12:
        return np.zeros_like(s, dtype=float)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


class ECODDetector:
    """Empirical Cumulative Distribution-based Outlier Detection (Li et al., TKDE 2022).

    Computes non-parametric left- and right-tail empirical probabilities per feature
    and aggregates negative log-probabilities across dimensions.
    """

    def __init__(self) -> None:
        self.sorted_features_: list[np.ndarray] = []
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> "ECODDetector":
        self.n_samples_, self.n_features_ = X.shape
        self.sorted_features_ = [np.sort(X[:, j]) for j in range(self.n_features_)]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return positive anomaly scores (higher = more extreme in tails)."""
        n_eval = X.shape[0]
        total_scores = np.zeros(n_eval, dtype=float)
        n_train = max(1, self.n_samples_)

        for j in range(self.n_features_):
            col_sorted = self.sorted_features_[j]
            vals = X[:, j]

            left_counts = np.searchsorted(col_sorted, vals, side="right")
            p_left = np.clip((left_counts + 1.0) / (n_train + 2.0), 1e-9, 1.0)

            right_counts = n_train - np.searchsorted(col_sorted, vals, side="left")
            p_right = np.clip((right_counts + 1.0) / (n_train + 2.0), 1e-9, 1.0)

            tail_p = np.minimum(p_left, p_right)
            total_scores += -np.log(tail_p)

        return total_scores / max(1, self.n_features_)


class COPODDetector:
    """Copula-Based Outlier Detection (Li et al., IEEE ICDM 2020 — PyOD Algorithm).

    Constructs an empirical copula across all electrical dimensions and combines:
    - Left-tail empirical copula log-probability (-log U_l)
    - Right-tail empirical copula log-probability (-log U_r)
    - Skewness-corrected tail log-probability (selecting left tail when skewness < 0
      and right tail when skewness >= 0, which is ideal for right-skewed kWh and imbalance distributions).
    """

    def __init__(self) -> None:
        self.sorted_features_: list[np.ndarray] = []
        self.skewness_: np.ndarray = np.array([], dtype=float)
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> "COPODDetector":
        self.n_samples_, self.n_features_ = X.shape
        self.sorted_features_ = [np.sort(X[:, j]) for j in range(self.n_features_)]
        skews = np.zeros(self.n_features_, dtype=float)
        for j in range(self.n_features_):
            col = X[:, j]
            if np.std(col) > 1e-9:
                sk = float(scipy_skew(col, bias=False))
                skews[j] = 0.0 if np.isnan(sk) else sk
        self.skewness_ = skews
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return COPOD empirical copula outlier scores (higher = more anomalous)."""
        n_eval = X.shape[0]
        n_train = max(1, self.n_samples_)
        o_left = np.zeros(n_eval, dtype=float)
        o_right = np.zeros(n_eval, dtype=float)
        o_skew = np.zeros(n_eval, dtype=float)

        for j in range(self.n_features_):
            col_sorted = self.sorted_features_[j]
            vals = X[:, j]

            u_l = np.clip((np.searchsorted(col_sorted, vals, side="right") + 1.0) / (n_train + 2.0), 1e-9, 1.0)
            u_r = np.clip(((n_train - np.searchsorted(col_sorted, vals, side="left")) + 1.0) / (n_train + 2.0), 1e-9, 1.0)

            neg_log_l = -np.log(u_l)
            neg_log_r = -np.log(u_r)

            o_left += neg_log_l
            o_right += neg_log_r
            # Skewness correction: positive skew -> right tail carries extreme anomalies; negative skew -> left tail
            if self.skewness_[j] >= 0.0:
                o_skew += neg_log_r
            else:
                o_skew += neg_log_l

        denom = float(max(1, self.n_features_))
        return np.maximum(np.maximum(o_left, o_right), o_skew) / denom


class HBOSDetector:
    """Histogram-Based Outlier Score (Goldstein & Dengel, KI 2012 — PyOD Algorithm).

    Builds Laplace-smoothed univariate histograms per electrical dimension and scores
    samples by their inverse log-density. Fast and highly sensitive to rare modes
    (e.g., zero-consumption plateaus or reverse-energy spikes).
    """

    def __init__(self, n_bins: int = 20, alpha: float = 0.01) -> None:
        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_edges_: list[np.ndarray] = []
        self.log_inv_densities_: list[np.ndarray] = []
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> "HBOSDetector":
        n_samples, self.n_features_ = X.shape
        n_b = max(5, min(self.n_bins, int(np.sqrt(max(25, n_samples)))))
        self.bin_edges_ = []
        self.log_inv_densities_ = []

        for j in range(self.n_features_):
            col = X[:, j]
            c_min, c_max = float(np.min(col)), float(np.max(col))
            if c_max - c_min < 1e-9:
                edges = np.array([c_min - 0.5, c_max + 0.5], dtype=float)
                inv_d = np.array([0.0], dtype=float)
            else:
                hist, edges = np.histogram(col, bins=n_b, density=False)
                prob = (hist.astype(float) + self.alpha) / (float(n_samples) + self.alpha * n_b)
                inv_d = -np.log(np.clip(prob, 1e-9, 1.0))
            self.bin_edges_.append(edges)
            self.log_inv_densities_.append(inv_d)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        n_eval = X.shape[0]
        scores = np.zeros(n_eval, dtype=float)
        for j in range(self.n_features_):
            edges = self.bin_edges_[j]
            inv_d = self.log_inv_densities_[j]
            if len(inv_d) <= 1:
                continue
            idx = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1, 0, len(inv_d) - 1)
            scores += inv_d[idx]
        return scores / float(max(1, self.n_features_))


class RobustPCADetector:
    """PCA Reconstruction Error detector for subspace correlation anomalies."""

    def __init__(self, variance_retained: float = 0.85, random_state: int = 42) -> None:
        self.variance_retained = variance_retained
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None

    def fit(self, X: np.ndarray) -> "RobustPCADetector":
        n_samples, n_features = X.shape
        if n_features <= 1 or n_samples <= 2:
            self.scaler_ = None
            self.pca_ = None
            return self

        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X)
        n_comp = max(1, min(n_samples - 1, n_features - 1, int(np.ceil(n_features * 0.6))))
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X_std)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return L2 reconstruction error per sample in standardized feature space."""
        if self.pca_ is None:
            return np.linalg.norm(X, axis=1)
        X_eval = self.scaler_.transform(X) if self.scaler_ is not None else X
        X_proj = self.pca_.transform(X_eval)
        X_rec = self.pca_.inverse_transform(X_proj)
        return np.mean((X_eval - X_rec) ** 2, axis=1)


class CompositeAnomalyDetector:
    """Multi-paradigm PyOD-grade ensemble for Smart Meter Electrical Anomaly Detection.

    Combines 6 complementary detectors on pure electrical features:
    - `iso_forest` (0.22): Global nonlinear feature interactions
    - `lof`        (0.20): Local k-NN density deviation within similar consumption profiles
    - `ecod`       (0.18): Empirical left/right tail probability (Li et al. 2022)
    - `copod`      (0.18): Empirical copula with skewness correction (Li et al. 2020)
    - `hbos`       (0.10): Histogram inverse log-density (Goldstein & Dengel 2012)
    - `pca`        (0.12): Linear subspace reconstruction error
    """

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        n_estimators: int = 150,
        lof_neighbors: int = 20,
        weights: dict[str, float] | None = None,
        normalization_mode: str = "frozen_quantile",
        freeze_threshold: bool = True,
        strict_voltage_imbalance_rule: bool = False,
        random_state: int = 42,
    ):
        self.contamination = float(np.clip(contamination, 0.001, 0.45))
        self.n_estimators = n_estimators
        self.lof_neighbors = lof_neighbors
        self.normalization_mode = normalization_mode
        self.freeze_threshold = freeze_threshold
        self.strict_voltage_imbalance_rule = strict_voltage_imbalance_rule
        self.random_state = random_state

        self.weights = weights or {
            "iso_forest": 0.22,
            "lof": 0.20,
            "ecod": 0.18,
            "copod": 0.18,
            "hbos": 0.10,
            "pca": 0.12,
        }

        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = RobustScaler()

        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.ecod = ECODDetector()
        self.copod = COPODDetector()
        self.hbos = HBOSDetector()
        self.pca = RobustPCADetector(random_state=self.random_state)
        self.is_fitted_: bool = False
        self.pyod_backend_active_: bool = HAS_PYOD
        self.norm_params_: dict[str, dict[str, Any]] = {}
        self.frozen_threshold_: float | None = None

    def _compute_raw_detector_scores(self, X_scaled: np.ndarray) -> dict[str, np.ndarray]:
        """Compute unnormalized anomaly scores (higher = more anomalous) for each sub-detector."""
        copod_det = getattr(self, "copod", None)
        if copod_det is None:
            copod_det = COPODDetector().fit(X_scaled)
            self.copod = copod_det

        hbos_det = getattr(self, "hbos", None)
        if hbos_det is None:
            hbos_det = HBOSDetector().fit(X_scaled)
            self.hbos = hbos_det

        return {
            "iso_forest": -self.iso_forest.score_samples(X_scaled),
            # Log-compress raw LOF to prevent k-distance=0 tie spikes (1e10) from crushing P01-P99
            "lof": np.log1p(np.maximum(-self.lof.score_samples(X_scaled) - 1.0, 0.0)),
            "ecod": self.ecod.score_samples(X_scaled),
            "copod": copod_det.score_samples(X_scaled),
            "hbos": hbos_det.score_samples(X_scaled),
            "pca": self.pca.score_samples(X_scaled),
        }

    def _combine_weighted_scores(self, norm_scores: dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted ensemble consensus before PRODIST rule boost."""
        w_iso = self.weights.get("iso_forest", 0.22)
        w_lof = self.weights.get("lof", 0.20)
        w_ecod = self.weights.get("ecod", 0.18)
        w_copod = self.weights.get("copod", 0.18)
        w_hbos = self.weights.get("hbos", 0.10)
        w_pca = self.weights.get("pca", 0.12)
        w_sum = max(1e-9, w_iso + w_lof + w_ecod + w_copod + w_hbos + w_pca)

        return (
            w_iso * norm_scores["iso_forest"]
            + w_lof * norm_scores["lof"]
            + w_ecod * norm_scores["ecod"]
            + w_copod * norm_scores["copod"]
            + w_hbos * norm_scores["hbos"]
            + w_pca * norm_scores["pca"]
        ) / w_sum

    def fit(self, X: np.ndarray) -> "CompositeAnomalyDetector":
        """Fit all 6 sub-detectors on clean electrical feature matrix X and freeze reference stats."""
        if len(X) == 0:
            raise ValueError("Cannot fit CompositeAnomalyDetector on empty array.")

        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        n_samples = len(X_scaled)
        effective_neighbors = max(2, min(self.lof_neighbors, n_samples - 1))
        self.lof.set_params(n_neighbors=effective_neighbors)

        self.iso_forest.fit(X_scaled)
        self.lof.fit(X_scaled)
        self.ecod.fit(X_scaled)
        self.copod.fit(X_scaled)
        self.hbos.fit(X_scaled)
        self.pca.fit(X_scaled)

        # Compute and freeze training normalization parameters and consensus threshold
        raw_train = self._compute_raw_detector_scores(X_scaled)
        self.norm_params_ = {name: _compute_norm_params(arr) for name, arr in raw_train.items()}

        mode = getattr(self, "normalization_mode", "frozen_quantile")
        norm_train = {
            name: _normalize_with_params(arr, self.norm_params_[name], mode)
            for name, arr in raw_train.items()
        }
        train_consensus = np.clip(self._combine_weighted_scores(norm_train), 0.0, 1.0)
        tau = float(np.percentile(train_consensus, 100.0 * (1.0 - self.contamination)))
        self.frozen_threshold_ = max(tau, 1e-6) if float(np.max(train_consensus)) <= 1e-6 else tau

        self.is_fitted_ = True
        return self

    def predict_composite(
        self,
        X: np.ndarray,
        raw_dataframe: pl.DataFrame | None = None,
        *,
        use_frozen_threshold: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute individual detector scores, physical rule flags, and consensus output."""
        if len(X) == 0:
            return {
                "consensus_score": np.array([], dtype=float),
                "iso_score": np.array([], dtype=float),
                "lof_score": np.array([], dtype=float),
                "ecod_score": np.array([], dtype=float),
                "copod_score": np.array([], dtype=float),
                "hbos_score": np.array([], dtype=float),
                "pca_score": np.array([], dtype=float),
                "raw_iso_score": np.array([], dtype=float),
                "raw_lof_score": np.array([], dtype=float),
                "raw_ecod_score": np.array([], dtype=float),
                "raw_copod_score": np.array([], dtype=float),
                "raw_hbos_score": np.array([], dtype=float),
                "raw_pca_score": np.array([], dtype=float),
                "rule_violation_flag": np.array([], dtype=bool),
                "is_anomaly": np.array([], dtype=bool),
            }

        if not self.is_fitted_:
            raise RuntimeError("CompositeAnomalyDetector must be fitted before calling predict_composite.")

        if np.isnan(X).any() and np.all(np.isnan(X), axis=0).any():
            valid_cols = ~np.all(np.isnan(X), axis=0)
            X_clean = np.zeros_like(X, dtype=float)
            if valid_cols.any():
                X_clean[:, valid_cols] = self.imputer.transform(X)[:, : int(valid_cols.sum())]
        else:
            X_clean = self.imputer.transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.transform(X_clean)

        raw_scores = self._compute_raw_detector_scores(X_scaled)
        norm_params = getattr(self, "norm_params_", {})
        mode = getattr(self, "normalization_mode", "frozen_quantile")

        s_iso = _normalize_with_params(raw_scores["iso_forest"], norm_params.get("iso_forest"), mode)
        s_lof = _normalize_with_params(raw_scores["lof"], norm_params.get("lof"), mode)
        s_ecod = _normalize_with_params(raw_scores["ecod"], norm_params.get("ecod"), mode)
        s_copod = _normalize_with_params(raw_scores["copod"], norm_params.get("copod"), mode)
        s_hbos = _normalize_with_params(raw_scores["hbos"], norm_params.get("hbos"), mode)
        s_pca = _normalize_with_params(raw_scores["pca"], norm_params.get("pca"), mode)

        # 7. Physics / PRODIST rules (Strictly electrical measurements)
        rule_flags = np.zeros(len(X), dtype=bool)
        if raw_dataframe is not None:
            rating_cols = {
                "TENSAO_BASE", "X__TENSAO_BASE",
                "TENSAO_ALIMENTADOR", "X__TENSAO_ALIMENTADOR",
                "TENSAO_SE", "X__TENSAO_SE",
                "TENSAO_SEC", "X__TENSAO_SEC",
                "NOMINAL_VOLTAGE", "X__NOMINAL_VOLTAGE",
            }

            v_meas_cols = [
                c for c in raw_dataframe.columns
                if c.upper() not in rating_cols
                and ("VOLTAGE" in c.upper() or "U_L1" in c.upper() or "TENSAO" in c.upper())
                and not any(x in c.upper() for x in ("IQR", "RAMP", "IMBALANCE", "DELTA", "STD", "AUTOCORR", "RATIO", "COUNT"))
            ]

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
                    plausible_meas = (v_arr > 50.0) & valid_mask
                    if np.any(plausible_meas):
                        v_pu = np.where(plausible_meas, v_arr / np.maximum(v_base_arr, 50.0), 1.0)
                        crit_sag_swell = plausible_meas & ((v_pu < 0.70) | (v_pu > 1.25))
                        rule_flags |= crit_sag_swell

            strict_v_rule = bool(getattr(self, "strict_voltage_imbalance_rule", False))
            if strict_v_rule:
                v_imb_cols = [c for c in raw_dataframe.columns if "VOLTAGE_IMBALANCE" in c.upper()]
            else:
                v_imb_cols = [c for c in raw_dataframe.columns if "VOLTAGE_IMBALANCE" in c.upper() or "IMBALANCE_MAX" in c.upper()]
            for vic in v_imb_cols:
                if raw_dataframe[vic].dtype.is_numeric():
                    imb_arr = raw_dataframe[vic].fill_null(0.0).to_numpy()
                    rule_flags |= (imb_arr > 20.0) if np.nanmax(imb_arr) > 2.0 else (imb_arr > 0.20)

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

        norm_scores = {
            "iso_forest": s_iso,
            "lof": s_lof,
            "ecod": s_ecod,
            "copod": s_copod,
            "hbos": s_hbos,
            "pca": s_pca,
        }
        consensus = self._combine_weighted_scores(norm_scores)
        consensus = np.clip(consensus + np.where(rule_flags, 0.15, 0.0), 0.0, 1.0)

        use_frozen = getattr(self, "freeze_threshold", True) if use_frozen_threshold is None else use_frozen_threshold
        frozen_tau = getattr(self, "frozen_threshold_", None)
        if use_frozen and frozen_tau is not None:
            threshold = float(frozen_tau)
        else:
            threshold = float(np.percentile(consensus, 100.0 * (1.0 - self.contamination)))
        self.last_threshold_applied_ = threshold

        is_anomaly = consensus >= threshold

        return {
            "consensus_score": np.round(consensus, 4),
            "iso_score": np.round(s_iso, 4),
            "lof_score": np.round(s_lof, 4),
            "ecod_score": np.round(s_ecod, 4),
            "copod_score": np.round(s_copod, 4),
            "hbos_score": np.round(s_hbos, 4),
            "pca_score": np.round(s_pca, 4),
            "raw_iso_score": raw_scores["iso_forest"],
            "raw_lof_score": raw_scores["lof"],
            "raw_ecod_score": raw_scores["ecod"],
            "raw_copod_score": raw_scores["copod"],
            "raw_hbos_score": raw_scores["hbos"],
            "raw_pca_score": raw_scores["pca"],
            "rule_violation_flag": rule_flags,
            "is_anomaly": is_anomaly,
            "threshold_applied": np.full(len(X), threshold, dtype=float),
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
