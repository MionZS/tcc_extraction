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

from src.models.composite import CompositeAnomalyDetector, ECODDetector, _min_max_normalize


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
        random_state: int = 42,
    ):
        self.contamination = float(np.clip(contamination, 0.001, 0.5))
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
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> CommunicationAnomalyDetector:
        """Fit detector on communication feature matrix X."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.fit_transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.iso_forest.fit(X_scaled)
        self.ecod.fit(X_scaled)
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

        s_iso = _min_max_normalize(-self.iso_forest.score_samples(X_scaled))
        s_ecod = _min_max_normalize(self.ecod.score_samples(X_scaled))

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
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.min_quality_threshold = min_quality_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.detector = CompositeAnomalyDetector(
            contamination=contamination,
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
    ) -> dict[str, np.ndarray]:
        """Compute electrical anomaly score S_elec conditioned on data quality.

        Returns dict containing:
        - s_elec: Anomaly score in [0.0, 1.0]
        - is_reliable: Boolean flag indicating if data quality was sufficient
        - is_electrical_anomaly: Boolean flag (anomaly and reliable)
        - consensus_breakdown: Dict of raw detector outputs
        """
        if not self.fitted_:
            raise RuntimeError("ElectricalAnomalyDetector must be fitted before predict.")

        dq_arr = np.asarray(data_quality, dtype=float)
        is_reliable = dq_arr >= self.min_quality_threshold

        res = self.detector.predict_composite(X, raw_dataframe=raw_dataframe)
        raw_scores = res["consensus_score"]

        # Only reliable observations can be declared definitive electrical anomalies
        is_anomaly = res["is_anomaly"] & is_reliable

        return {
            "s_elec": raw_scores,
            "data_quality": dq_arr,
            "is_reliable": is_reliable,
            "is_electrical_anomaly": is_anomaly,
            "rule_violation_flag": res["rule_violation_flag"],
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

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering pipeline and return cluster labels."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_clean = self.imputer.fit_transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        k = min(self.n_pca_components, X.shape[1])
        self.pca.set_params(n_components=k)
        X_pca = self.pca.fit_transform(X_scaled)

        labels = self.kmeans.fit_predict(X_pca)
        self.fitted_ = True
        return labels


class MultiViewAnomalyPipeline:
    """End-to-End Orchestrator for Specialized Anomaly Scoring.

    Integrates S_comm, S_elec, S_tamper, Data Quality, and Behavioral Clusters
    into a unified actionable diagnostic matrix.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        min_quality_threshold: float = 0.70,
        random_state: int = 42,
    ):
        self.comm_detector = CommunicationAnomalyDetector(
            contamination=contamination,
            random_state=random_state,
        )
        self.elec_detector = ElectricalAnomalyDetector(
            min_quality_threshold=min_quality_threshold,
            contamination=contamination,
            random_state=random_state,
        )
        self.tamper_detector = TamperAnomalyDetector()
        self.clusterer = BehavioralClusterer(n_clusters=4, random_state=random_state)

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
        """Execute all specialized models and assemble the decomposed score table."""
        # 1. Communication score
        self.comm_detector.fit(X_comm)
        s_comm = self.comm_detector.score_samples(X_comm)

        # 2. Electrical score
        self.elec_detector.fit(X_elec)
        elec_res = self.elec_detector.predict_electrical(
            X_elec, data_quality=data_quality, raw_dataframe=raw_dataframe
        )
        s_elec = elec_res["s_elec"]
        is_reliable_elec = elec_res["is_reliable"]
        is_elec_anomaly = elec_res["is_electrical_anomaly"]

        # 3. Tamper score
        s_tamper = self.tamper_detector.score(
            tamper_alarm_counts=tamper_alarms,
            fa_trend_slopes=fa_trend_slopes,
            meter_changed_30d=meter_changed_30d,
        )

        # 4. Behavioral clusters
        cluster_ids = self.clusterer.fit_predict(X_elec)

        # 5. Diagnostic interpretation
        diagnoses = []
        n_samples = len(s_comm)
        for i in range(n_samples):
            sc = s_comm[i]
            se = s_elec[i]
            st = s_tamper[i]
            rel = is_reliable_elec[i]

            if st >= 0.70:
                diag = "INVESTIGACAO_FRAUDE_OU_TAMPER"
            elif not rel and sc >= 0.70:
                diag = "FALHA_CRITICA_TELEMETRIA_DADO_INSUFICIENTE"
            elif not rel:
                diag = "DADO_ELETRICO_INSUFICIENTE_COMUNICACAO_DEGRADADA"
            elif se >= 0.70:
                diag = "ANOMALIA_ELETRICA_CONFIRMADA"
            elif sc >= 0.70:
                diag = "INSTABILIDADE_COMUNICACAO_OPERACAO_NORMAL"
            else:
                diag = "OPERACAO_REGULAR"
            diagnoses.append(diag)

        res_dict: dict[str, Any] = {
            "s_comm": s_comm,
            "s_elec": s_elec,
            "s_tamper": s_tamper,
            "data_quality": np.asarray(data_quality, dtype=float),
            "is_reliable_elec": is_reliable_elec,
            "is_elec_anomaly": is_elec_anomaly,
            "cluster_id": cluster_ids,
            "operational_diagnosis": diagnoses,
        }
        if uc_ids is not None:
            res_dict["id__uc_id"] = list(uc_ids)

        return pl.DataFrame(res_dict)
