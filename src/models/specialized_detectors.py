"""Specialized Multi-View Anomaly Detection Engine for Smart Meters.

Deconstructs monolithic anomaly detection into specialized, orthogonal models:
1. Communication Anomaly Model (S_comm): Availability, cadence, gaps, packet loss, telemetry alarms.
2. Electrical Anomaly Model (S_elec): Physical load, energy, voltage/current imbalance, PRODIST rules.
   - Guarded by data_quality gating: low completeness yields is_reliable=False rather than fake electrical score.
3. Tamper / Irregularity Model (S_tamper): Hardware intrusion, cover open, strong magnetic field, bypass.
4. Behavioral Profile Clusterer: Unsupervised PCA + Clustering (e.g. residential vs commercial vs constant load).
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.models.composite import (
    CompositeAnomalyDetector,
    ECODDetector,
    _compute_norm_params,
    _min_max_normalize,
    _normalize_with_params,
)


class CommunicationAnomalyDetector(BaseEstimator):
    """Detects telemetry anomalies, packet drop bursts, and chronic modem failures.

    Input features:
    - completeness ratios (1d, 7d, 30d)
    - gap statistics (count, max_gap_min, mean_gap_min, p95_gap_min)
    - consecutive missing intervals
    - nocturnal gap ratio
    - communication alarms (iommsOffline, poorLinkQuality, etc.) and latencies
    """

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        normalization_mode: str = "frozen_quantile",
        random_state: int = 42,
    ):
        self.contamination = float(np.clip(contamination, 0.001, 0.5))
        self.normalization_mode = normalization_mode
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.ecod = ECODDetector()
        self.norm_params_: dict[str, dict[str, Any]] = {}
        self.frozen_threshold_: float | None = None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> CommunicationAnomalyDetector:
        """Fit detector on communication feature matrix X and freeze normalization stats."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.fit_transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.iso_forest.fit(X_scaled)
        self.ecod.fit(X_scaled)

        raw_iso = -self.iso_forest.score_samples(X_scaled)
        raw_ecod = self.ecod.score_samples(X_scaled)
        self.norm_params_ = {
            "iso": _compute_norm_params(raw_iso),
            "ecod": _compute_norm_params(raw_ecod),
        }
        mode = getattr(self, "normalization_mode", "frozen_quantile")
        s_iso = _normalize_with_params(raw_iso, self.norm_params_["iso"], mode)
        s_ecod = _normalize_with_params(raw_ecod, self.norm_params_["ecod"], mode)
        s_comm_train = 0.5 * s_iso + 0.5 * s_ecod
        self.frozen_threshold_ = float(np.percentile(s_comm_train, 100.0 * (1.0 - self.contamination)))
        self.fitted_ = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute communication anomaly score S_comm in [0.0, 1.0]."""
        if not self.fitted_:
            raise RuntimeError("CommunicationAnomalyDetector must be fitted before predict.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.transform(X_clean)

        raw_iso = -self.iso_forest.score_samples(X_scaled)
        raw_ecod = self.ecod.score_samples(X_scaled)
        norm_params = getattr(self, "norm_params_", {})
        mode = getattr(self, "normalization_mode", "frozen_quantile")

        s_iso = _normalize_with_params(raw_iso, norm_params.get("iso"), mode)
        s_ecod = _normalize_with_params(raw_ecod, norm_params.get("ecod"), mode)

        s_comm = 0.5 * s_iso + 0.5 * s_ecod
        return np.round(s_comm, 4)


class ElectricalAnomalyDetector(BaseEstimator):
    """Detects genuine physical consumption and power quality anomalies.

    Excludes all communication missingness from input features X.
    Applies strict data quality gating: if data_quality < min_quality,
    the score is flagged as unreliable rather than fabricated.
    """

    def __init__(
        self,
        *,
        min_quality_threshold: float = 0.70,
        min_reliable_days: int | None = None,
        contamination: float = 0.05,
        normalization_mode: str = "frozen_quantile",
        freeze_threshold: bool = True,
        strict_voltage_imbalance_rule: bool = False,
        random_state: int = 42,
    ):
        self.min_quality_threshold = min_quality_threshold
        self.min_reliable_days = min_reliable_days
        self.contamination = contamination
        self.normalization_mode = normalization_mode
        self.freeze_threshold = freeze_threshold
        self.strict_voltage_imbalance_rule = strict_voltage_imbalance_rule
        self.random_state = random_state
        self.detector = CompositeAnomalyDetector(
            contamination=contamination,
            normalization_mode=normalization_mode,
            freeze_threshold=freeze_threshold,
            strict_voltage_imbalance_rule=strict_voltage_imbalance_rule,
            random_state=random_state,
        )
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> ElectricalAnomalyDetector:
        """Fit electrical detector strictly on physical features."""
        self.detector.fit(X)
        self.fitted_ = True
        return self

    def predict_electrical(
        self,
        X: np.ndarray,
        data_quality: np.ndarray | Sequence[float],
        raw_dataframe: pl.DataFrame | None = None,
        reliable_days: np.ndarray | Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Compute electrical anomaly score S_elec conditioned on data quality.

        Returns dict containing:
        - s_elec: Anomaly score in [0.0, 1.0]
        - is_reliable: Boolean flag indicating if data quality was sufficient
        - is_electrical_anomaly: Boolean flag (anomaly and reliable)
        - rule_violation_flag: Physical PRODIST rule flag
        - breakdown: Individual sub-detector normalized scores
        """
        if not self.fitted_:
            raise RuntimeError("ElectricalAnomalyDetector must be fitted before predict.")

        dq_arr = np.asarray(data_quality, dtype=float)
        is_reliable = dq_arr >= self.min_quality_threshold
        min_rel_d = getattr(self, "min_reliable_days", None)
        if min_rel_d is not None:
            if reliable_days is not None:
                rd_arr = np.asarray(reliable_days, dtype=float)
                is_reliable = is_reliable & (rd_arr >= float(min_rel_d))
            elif raw_dataframe is not None and "x__coverage_reliable_days" in raw_dataframe.columns:
                rd_arr = raw_dataframe["x__coverage_reliable_days"].fill_null(0).to_numpy()
                is_reliable = is_reliable & (rd_arr >= float(min_rel_d))

        res = self.detector.predict_composite(X, raw_dataframe=raw_dataframe)
        raw_scores = res["consensus_score"]

        # Only reliable observations can be declared candidate electrical anomalies
        is_anomaly = res["is_anomaly"] & is_reliable

        return {
            "s_elec": raw_scores,
            "data_quality": dq_arr,
            "is_reliable": is_reliable,
            "is_electrical_anomaly": is_anomaly,
            "rule_violation_flag": res["rule_violation_flag"],
            "breakdown": res,
        }


class TamperAnomalyDetector:
    """Computes Tamper / Hardware Intrusion Score S_tamper.

    Combines:
    - Tamper alarm occurrences (Strong Magnetic Field, Cover Open, Bypass)
    - Electrical consumption collapse (severe negative slope after tamper event)
    """

    def __init__(self, *, alarm_weight: float = 0.70, slope_weight: float = 0.30):
        self.alarm_weight = alarm_weight
        self.slope_weight = slope_weight

    def score(
        self,
        *,
        tamper_alarm_counts: np.ndarray | Sequence[int],
        fa_trend_slopes: np.ndarray | Sequence[float] | None = None,
        meter_changed_30d: np.ndarray | Sequence[bool] | None = None,
    ) -> np.ndarray:
        """Compute S_tamper in [0.0, 1.0]."""
        alarms = np.asarray(tamper_alarm_counts, dtype=float)
        n = len(alarms)

        # Non-linear saturation for alarm counts (1 alarm is notable, >= 3 is severe)
        s_alarms = np.clip(alarms / 3.0, 0.0, 1.0)

        # Drop penalty: negative slope indicates post-tamper drop in billed energy
        if fa_trend_slopes is not None:
            slopes = np.asarray(fa_trend_slopes, dtype=float)
            s_drop = np.clip(-slopes / 5.0, 0.0, 1.0)
        else:
            s_drop = np.zeros(n)

        # Meter replacement context
        meter_flag = np.asarray(meter_changed_30d, dtype=float) if meter_changed_30d is not None else np.zeros(n)

        s_tamper = self.alarm_weight * s_alarms + self.slope_weight * s_drop
        # Elevate if meter was recently replaced or has multiple alarms
        s_tamper = np.clip(s_tamper + 0.15 * (s_alarms > 0.0) * meter_flag, 0.0, 1.0)
        return np.round(s_tamper, 4)


class BehavioralClusterer:
    """Profiles electrical consumption patterns into baseline archetypes.

    Extracts consumption habits (e.g. nocturnal residential, diurnal commercial,
    steady baseload, solar prosumer) using PCA + KMeans.
    """

    def __init__(self, n_clusters: int = 4, n_pca_components: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.random_state = random_state

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_pca_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> BehavioralClusterer:
        """Fit clustering pipeline without predicting."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.fit_transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        k = min(self.n_pca_components, X.shape[1])
        self.pca.set_params(n_components=k)
        X_pca = self.pca.fit_transform(X_scaled)

        self.kmeans.fit(X_pca)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels using frozen imputer, scaler, PCA, and KMeans."""
        if not self.fitted_:
            raise RuntimeError("BehavioralClusterer must be fitted before predict.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.transform(X_clean)
        X_pca = self.pca.transform(X_scaled)
        return self.kmeans.predict(X_pca)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering pipeline and return cluster labels."""
        return self.fit(X).predict(X)


class MultiViewAnomalyPipeline:
    """End-to-End Orchestrator for Specialized Anomaly Scoring.

    Integrates S_comm, S_elec, S_tamper, Data Quality, and Behavioral Clusters
    into a unified actionable diagnostic matrix. Supports strict separation between
    `.fit(...)` on clean training windows and frozen `.score(...)` on OOT cohorts.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        min_quality_threshold: float = 0.70,
        min_reliable_days: int | None = None,
        normalization_mode: str = "frozen_quantile",
        freeze_threshold: bool = True,
        strict_voltage_imbalance_rule: bool = False,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.min_quality_threshold = min_quality_threshold
        self.min_reliable_days = min_reliable_days
        self.normalization_mode = normalization_mode
        self.freeze_threshold = freeze_threshold
        self.strict_voltage_imbalance_rule = strict_voltage_imbalance_rule
        self.random_state = random_state

        self.comm_detector = CommunicationAnomalyDetector(
            contamination=contamination,
            normalization_mode=normalization_mode,
            random_state=random_state,
        )
        self.elec_detector = ElectricalAnomalyDetector(
            min_quality_threshold=min_quality_threshold,
            min_reliable_days=min_reliable_days,
            contamination=contamination,
            normalization_mode=normalization_mode,
            freeze_threshold=freeze_threshold,
            strict_voltage_imbalance_rule=strict_voltage_imbalance_rule,
            random_state=random_state,
        )
        self.tamper_detector = TamperAnomalyDetector()
        self.clusterer = BehavioralClusterer(n_clusters=4, random_state=random_state)
        self.fitted_ = False

    def fit(
        self,
        *,
        X_comm: np.ndarray,
        X_elec: np.ndarray,
    ) -> MultiViewAnomalyPipeline:
        """Fit all specialized models and freeze preprocessing/normalization artifacts on training data."""
        self.comm_detector.fit(X_comm)
        self.elec_detector.fit(X_elec)
        self.clusterer.fit(X_elec)
        self.fitted_ = True
        return self

    def score(
        self,
        *,
        X_comm: np.ndarray,
        X_elec: np.ndarray,
        data_quality: np.ndarray | Sequence[float],
        tamper_alarms: np.ndarray | Sequence[int],
        fa_trend_slopes: np.ndarray | Sequence[float] | None = None,
        meter_changed_30d: np.ndarray | Sequence[bool] | None = None,
        raw_dataframe: pl.DataFrame | None = None,
        uc_ids: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Run frozen out-of-sample inference across all specialized views."""
        if not self.fitted_:
            raise RuntimeError("MultiViewAnomalyPipeline must be fitted before calling score().")

        s_comm = self.comm_detector.score_samples(X_comm)
        elec_res = self.elec_detector.predict_electrical(
            X_elec, data_quality=data_quality, raw_dataframe=raw_dataframe
        )
        s_elec = elec_res["s_elec"]
        is_reliable_elec = elec_res["is_reliable"]
        is_elec_anomaly = elec_res["is_electrical_anomaly"]
        breakdown = elec_res.get("breakdown", {})

        s_tamper = self.tamper_detector.score(
            tamper_alarm_counts=tamper_alarms,
            fa_trend_slopes=fa_trend_slopes,
            meter_changed_30d=meter_changed_30d,
        )

        cluster_ids = self.clusterer.predict(X_elec)

        # Diagnostic interpretation without causal confirmation claims from telemetry
        diagnoses = []
        n_samples = len(s_comm)
        for i in range(n_samples):
            sc = float(s_comm[i])
            se = float(s_elec[i])
            st = float(s_tamper[i])
            rel = bool(is_reliable_elec[i])
            elec_anom = bool(is_elec_anomaly[i])

            if st >= 0.70:
                diag = "INVESTIGACAO_FRAUDE_OU_TAMPER"
            elif not rel and sc >= 0.70:
                diag = "FALHA_CRITICA_TELEMETRIA_DADO_INSUFICIENTE"
            elif not rel:
                diag = "DADO_ELETRICO_INSUFICIENTE_COMUNICACAO_DEGRADADA"
            elif elec_anom or se >= 0.70:
                diag = "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT"
            elif sc >= 0.70:
                diag = "INSTABILIDADE_COMUNICACAO_OPERACAO_NORMAL"
            else:
                diag = "OPERACAO_REGULAR"
            diagnoses.append(diag)

        # Combined actionable score: use electrical consensus on reliable UCs, tamper fallback on high tamper
        combined_score = np.where(is_reliable_elec, np.maximum(s_elec, 0.85 * s_tamper), s_comm)
        combined_is_anom = is_elec_anomaly | (is_reliable_elec & (s_tamper >= 0.70))

        cols_to_attach: dict[str, Any] = {
            "s_comm": s_comm,
            "s_elec": s_elec,
            "s_tamper": s_tamper,
            "data_quality": np.asarray(data_quality, dtype=float),
            "is_reliable_elec": is_reliable_elec,
            "is_elec_anomaly": is_elec_anomaly,
            "cluster_id": cluster_ids,
            "operational_diagnosis": diagnoses,
            "ANOMALY_SCORE": np.round(combined_score, 4),
            "IS_ANOMALY": combined_is_anom,
        }
        for b_key in ("iso_score", "lof_score", "ecod_score", "copod_score", "hbos_score", "pca_score", "rule_violation_flag"):
            if b_key in breakdown:
                cols_to_attach[b_key] = breakdown[b_key]

        if raw_dataframe is not None:
            out_df = raw_dataframe.clone()
            existing = set(out_df.columns)
            to_drop = [k for k in cols_to_attach if k in existing]
            if to_drop:
                out_df = out_df.drop(to_drop)
            out_df = out_df.with_columns([
                pl.Series(name=k, values=v) for k, v in cols_to_attach.items()
            ])
            if uc_ids is not None and "id__uc_id" not in out_df.columns:
                out_df = out_df.with_columns(pl.Series(name="id__uc_id", values=list(uc_ids)))
            return out_df

        if uc_ids is not None:
            cols_to_attach["id__uc_id"] = list(uc_ids)
        return pl.DataFrame(cols_to_attach)

    def fit_and_score(
        self,
        *,
        X_comm: np.ndarray,
        X_elec: np.ndarray,
        data_quality: np.ndarray | Sequence[float],
        tamper_alarms: np.ndarray | Sequence[int],
        fa_trend_slopes: np.ndarray | Sequence[float] | None = None,
        meter_changed_30d: np.ndarray | Sequence[bool] | None = None,
        raw_dataframe: pl.DataFrame | None = None,
        uc_ids: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Fit on the provided matrices and immediately score them (for single-cohort exploratory runs)."""
        self.fit(X_comm=X_comm, X_elec=X_elec)
        return self.score(
            X_comm=X_comm,
            X_elec=X_elec,
            data_quality=data_quality,
            tamper_alarms=tamper_alarms,
            fa_trend_slopes=fa_trend_slopes,
            meter_changed_30d=meter_changed_30d,
            raw_dataframe=raw_dataframe,
            uc_ids=uc_ids,
        )
