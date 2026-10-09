"""Voltage Disturbance and Imbalance Classifier.

Classifies voltage quality disturbances, fault types (monofásica, bifásica, trifásica),
and phase unbalance following PRODIST Módulo 8 (ANEEL) and IEEE Std 1159.

Handles missing measurements, structural phase differences (Monofásico, Bifásico, Trifásico),
and telemetry outages by combining physical measurements, spatial neighborhood context
(trafo peers), and alarm events (last gasp).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier


class VoltageDisturbanceType(str, Enum):
    """Categories of voltage disturbances and faults."""

    NORMAL = "NORMAL"
    SUBTENSAO = "SUBTENSAO"
    SOBRETENSAO = "SOBRETENSAO"
    FALTA_MONOFASICA = "FALTA_MONOFASICA"
    FALTA_BIFASICA = "FALTA_BIFASICA"
    INTERRUPCAO_TOTAL = "INTERRUPCAO_TOTAL"
    DESBALANCEAMENTO_CRITICO = "DESBALANCEAMENTO_CRITICO"
    DESCONEXAO_TELEMETRIA = "DESCONEXAO_TELEMETRIA"


@dataclass(frozen=True)
class ClassificationResult:
    """Output container for voltage classification predictions."""

    predictions_df: pl.DataFrame
    feature_names: list[str]


def extract_voltage_classifier_features(
    instantaneous_df: pl.DataFrame,
    context_df: pl.DataFrame | None = None,
    neighborhood_df: pl.DataFrame | None = None,
    alarms_df: pl.DataFrame | None = None,
    *,
    tensao_base_default: float = 127.0,
) -> pl.DataFrame:
    """Extract normalized features and missing data indicators for voltage classification.

    Handles the three types of missing data:
    1. Structural Nulls: Phase physically missing (e.g., L3 in bifásico/monofásico).
    2. Zero Voltages: Telemetry alive, phase voltage measured 0.0V.
    3. Complete Telemetry Absence: No packets received (differentiates blackout vs comm failure
       using trafo peer coverage and last gasp alarms).

    Args:
        instantaneous_df: Telemetry dataframe with U_L1, U_L2, U_L3 or U_L1_AVG...
        context_df: UC context dataframe with UC, FASE_CIRCUITO, TENSAO_BASE.
        neighborhood_df: Trafo neighborhood features (NEIGHBOR_COVERAGE_RATIO, etc.).
        alarms_df: Alarm events dataframe (power outage / last gasp alarms).
        tensao_base_default: Fallback nominal voltage in Volts.

    Returns:
        Polars DataFrame with features for classification.
    """
    if instantaneous_df.is_empty():
        return instantaneous_df

    df = instantaneous_df.clone()

    # Join context if available
    if context_df is not None and not context_df.is_empty() and "UC" in context_df.columns:
        context_cols = [c for c in ["UC", "FASE_CIRCUITO", "TENSAO_BASE"] if c in context_df.columns]
        df = df.join(context_df.select(context_cols), on="UC", how="left")

    # Determine base voltage column or fallback
    if "TENSAO_BASE" not in df.columns:
        df = df.with_columns(pl.lit(tensao_base_default).alias("TENSAO_BASE"))
    else:
        df = df.with_columns(pl.col("TENSAO_BASE").fill_null(tensao_base_default))

    # Identify phase voltage columns
    u_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in df.columns]
    if not u_cols:
        u_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in df.columns]

    # Map standard phase names U_L1, U_L2, U_L3
    l1_col = next((c for c in u_cols if "L1" in c), None)
    l2_col = next((c for c in u_cols if "L2" in c), None)
    l3_col = next((c for c in u_cols if "L3" in c), None)

    # 1. Structural Phase Activation Flags based on FASE_CIRCUITO
    # Monofásico -> L1 active; Bifásico -> L1, L2 active; Trifásico -> L1, L2, L3 active
    phase_type = pl.col("FASE_CIRCUITO").fill_null("TRIFASICO").str.to_uppercase()
    df = df.with_columns([
        pl.lit(True).alias("IS_ACTIVE_L1"),
        pl.when(phase_type.str.contains("BIFASICO") | phase_type.str.contains("TRIFASICO"))
        .then(True).otherwise(False).alias("IS_ACTIVE_L2"),
        pl.when(phase_type.str.contains("TRIFASICO"))
        .then(True).otherwise(False).alias("IS_ACTIVE_L3"),
    ])

    # 2. Per-Unit Voltages (PU) and Missing/Zero Indicators
    v_base = pl.col("TENSAO_BASE")

    l1_val = pl.col(l1_col) if l1_col else pl.lit(None, dtype=pl.Float64)
    l2_val = pl.col(l2_col) if l2_col else pl.lit(None, dtype=pl.Float64)
    l3_val = pl.col(l3_col) if l3_col else pl.lit(None, dtype=pl.Float64)

    df = df.with_columns([
        (l1_val / v_base).alias("U_L1_PU"),
        (l2_val / v_base).alias("U_L2_PU"),
        (l3_val / v_base).alias("U_L3_PU"),

        # Missing indicator (telemetry missing for an expected active phase)
        (l1_val.is_null() & pl.col("IS_ACTIVE_L1")).alias("IS_NULL_L1"),
        (l2_val.is_null() & pl.col("IS_ACTIVE_L2")).alias("IS_NULL_L2"),
        (l3_val.is_null() & pl.col("IS_ACTIVE_L3")).alias("IS_NULL_L3"),

        # Zero indicator (telemetry received, but phase voltage = 0.0)
        (l1_val.fill_null(999.0) <= 5.0).alias("IS_ZERO_L1"),
        (l2_val.fill_null(999.0) <= 5.0).alias("IS_ZERO_L2"),
        (l3_val.fill_null(999.0) <= 5.0).alias("IS_ZERO_L3"),
    ])

    # 3. Dynamic Voltage Imbalance Ratio for Active Non-Null Phases
    # PRODIST M8: Imbalance = max(|V_k - V_avg|) / V_avg
    active_pu_exprs = [
        pl.when(pl.col("IS_ACTIVE_L1") & pl.col("U_L1_PU").is_not_null()).then(pl.col("U_L1_PU")),
        pl.when(pl.col("IS_ACTIVE_L2") & pl.col("U_L2_PU").is_not_null()).then(pl.col("U_L2_PU")),
        pl.when(pl.col("IS_ACTIVE_L3") & pl.col("U_L3_PU").is_not_null()).then(pl.col("U_L3_PU")),
    ]

    count_active_valid = sum(
        pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"U_L{i}_PU").is_not_null()).then(1).otherwise(0)
        for i in (1, 2, 3)
    )

    sum_active_pu = sum(
        pl.col(f"U_L{i}_PU").fill_null(0.0)
        for i in (1, 2, 3)
    )

    v_avg_pu = pl.when(count_active_valid > 0).then(sum_active_pu / count_active_valid).otherwise(0.0)
    
    # Calculate max deviation from mean across active phases
    dev_l1 = (pl.col("U_L1_PU") - v_avg_pu).abs()
    dev_l2 = (pl.col("U_L2_PU") - v_avg_pu).abs()
    dev_l3 = (pl.col("U_L3_PU") - v_avg_pu).abs()

    max_dev = pl.max_horizontal([
        pl.when(pl.col("IS_ACTIVE_L1") & pl.col("U_L1_PU").is_not_null()).then(dev_l1).otherwise(0.0),
        pl.when(pl.col("IS_ACTIVE_L2") & pl.col("U_L2_PU").is_not_null()).then(dev_l2).otherwise(0.0),
        pl.when(pl.col("IS_ACTIVE_L3") & pl.col("U_L3_PU").is_not_null()).then(dev_l3).otherwise(0.0),
    ])

    v_imb_ratio = pl.when((v_avg_pu > 0.05) & (count_active_valid >= 2)).then(max_dev / v_avg_pu).otherwise(0.0)

    v_min_pu = pl.min_horizontal([
        pl.when(pl.col("IS_ACTIVE_L1")).then(pl.col("U_L1_PU").fill_null(0.0)).otherwise(99.0),
        pl.when(pl.col("IS_ACTIVE_L2")).then(pl.col("U_L2_PU").fill_null(0.0)).otherwise(99.0),
        pl.when(pl.col("IS_ACTIVE_L3")).then(pl.col("U_L3_PU").fill_null(0.0)).otherwise(99.0),
    ])

    v_max_pu = pl.max_horizontal([
        pl.when(pl.col("IS_ACTIVE_L1")).then(pl.col("U_L1_PU").fill_null(0.0)).otherwise(0.0),
        pl.when(pl.col("IS_ACTIVE_L2")).then(pl.col("U_L2_PU").fill_null(0.0)).otherwise(0.0),
        pl.when(pl.col("IS_ACTIVE_L3")).then(pl.col("U_L3_PU").fill_null(0.0)).otherwise(0.0),
    ])

    df = df.with_columns([
        v_avg_pu.alias("V_AVG_PU"),
        v_min_pu.alias("V_MIN_PU"),
        v_max_pu.alias("V_MAX_PU"),
        v_imb_ratio.alias("VOLTAGE_IMBALANCE_RATIO"),
        count_active_valid.alias("ACTIVE_VALID_PHASES_COUNT"),
    ])

    # 4. Integrate Spatial Neighborhood Context (Trafo Peer Coverage)
    if neighborhood_df is not None and not neighborhood_df.is_empty() and "UC" in neighborhood_df.columns:
        neigh_cols = [c for c in ["UC", "NEIGHBOR_COVERAGE_RATIO", "TRAFO_PEER_COUNT"] if c in neighborhood_df.columns]
        df = df.join(neighborhood_df.select(neigh_cols), on="UC", how="left")
    
    if "NEIGHBOR_COVERAGE_RATIO" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("NEIGHBOR_COVERAGE_RATIO"))
    else:
        df = df.with_columns(pl.col("NEIGHBOR_COVERAGE_RATIO").fill_null(1.0))

    # 5. Integrate Power Failure Alarms (Last Gasp)
    if alarms_df is not None and not alarms_df.is_empty() and "NIO" in alarms_df.columns:
        alarm_nios = set(
            alarms_df.filter(pl.col("CONTENT").str.to_uppercase().str.contains("FALTA|POWER|LAST_GASP"))
            .get_column("NIO")
            .to_list()
        )
        if "NIO" in df.columns:
            df = df.with_columns(pl.col("NIO").is_in(alarm_nios).alias("HAS_POWER_ALARM"))
        else:
            df = df.with_columns(pl.lit(False).alias("HAS_POWER_ALARM"))
    else:
        df = df.with_columns(pl.lit(False).alias("HAS_POWER_ALARM"))

    return df


class RuleBasedVoltageClassifier:
    """Heuristic Baseline Classifier following PRODIST Módulo 8 & IEEE 1159 rules."""

    @staticmethod
    def classify(df: pl.DataFrame) -> pl.DataFrame:
        """Assign baseline disturbance label using deterministic physical rules."""
        if df.is_empty():
            return df

        # Count zero phases among active phases
        zero_phase_count = sum(
            pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"IS_ZERO_L{i}")).then(1).otherwise(0)
            for i in (1, 2, 3)
        )

        null_phase_count = sum(
            pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"IS_NULL_L{i}")).then(1).otherwise(0)
            for i in (1, 2, 3)
        )

        active_phase_count = sum(
            pl.when(pl.col(f"IS_ACTIVE_L{i}")).then(1).otherwise(0)
            for i in (1, 2, 3)
        )

        # Rule evaluation
        is_total_interruption = (
            (zero_phase_count == active_phase_count)
            | (
                (null_phase_count == active_phase_count)
                & ((pl.col("NEIGHBOR_COVERAGE_RATIO") <= 0.3) | pl.col("HAS_POWER_ALARM"))
            )
        )

        is_monofasica = (zero_phase_count == 1) & (active_phase_count >= 2)
        is_bifasica = (zero_phase_count == 2) & (active_phase_count == 3)

        is_subtensao = (pl.col("V_MIN_PU") > 0.05) & (pl.col("V_MIN_PU") < 0.90) & (~is_monofasica) & (~is_bifasica)
        is_sobretensao = (pl.col("V_MAX_PU") > 1.05)
        is_desbalanceamento = (pl.col("VOLTAGE_IMBALANCE_RATIO") >= 0.02) & (pl.col("V_MIN_PU") >= 0.90)

        is_telemetry_outage = (
            (null_phase_count == active_phase_count)
            & (pl.col("NEIGHBOR_COVERAGE_RATIO") > 0.5)
            & (~pl.col("HAS_POWER_ALARM"))
        )

        label_expr = (
            pl.when(is_total_interruption)
            .then(pl.lit(VoltageDisturbanceType.INTERRUPCAO_TOTAL.value))
            .when(is_monofasica)
            .then(pl.lit(VoltageDisturbanceType.FALTA_MONOFASICA.value))
            .when(is_bifasica)
            .then(pl.lit(VoltageDisturbanceType.FALTA_BIFASICA.value))
            .when(is_subtensao)
            .then(pl.lit(VoltageDisturbanceType.SUBTENSAO.value))
            .when(is_sobretensao)
            .then(pl.lit(VoltageDisturbanceType.SOBRETENSAO.value))
            .when(is_desbalanceamento)
            .then(pl.lit(VoltageDisturbanceType.DESBALANCEAMENTO_CRITICO.value))
            .when(is_telemetry_outage)
            .then(pl.lit(VoltageDisturbanceType.DESCONEXAO_TELEMETRIA.value))
            .otherwise(pl.lit(VoltageDisturbanceType.NORMAL.value))
        )

        return df.with_columns(label_expr.alias("BASELINE_VOLTAGE_LABEL"))


class MLVoltageDisturbanceClassifier:
    """Supervised ML Classifier (HistGradientBoosting) for Voltage Disturbances.

    Trained on extracted voltage features (including missing value indicators & spatial peer context).
    Capable of handling missing data natively and predicting class probabilities.
    """

    FEATURE_COLS = [
        "U_L1_PU",
        "U_L2_PU",
        "U_L3_PU",
        "V_MIN_PU",
        "V_MAX_PU",
        "V_AVG_PU",
        "VOLTAGE_IMBALANCE_RATIO",
        "IS_NULL_L1",
        "IS_NULL_L2",
        "IS_NULL_L3",
        "IS_ZERO_L1",
        "IS_ZERO_L2",
        "IS_ZERO_L3",
        "IS_ACTIVE_L1",
        "IS_ACTIVE_L2",
        "IS_ACTIVE_L3",
        "NEIGHBOR_COVERAGE_RATIO",
        "HAS_POWER_ALARM",
    ]

    def __init__(self, *, random_state: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            categorical_features=[7, 8, 9, 10, 11, 12, 13, 14, 15, 17],
            random_state=random_state,
        )
        self.classes_: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        features_df: pl.DataFrame,
        target_column: str = "BASELINE_VOLTAGE_LABEL",
    ) -> MLVoltageDisturbanceClassifier:
        """Fit the ML model using labeled data (from baseline or ground truth)."""
        valid_df = features_df.filter(pl.col(target_column).is_not_null())
        if valid_df.height < 10:
            raise ValueError("Insufficient training samples for ML voltage classifier.")

        X = valid_df.select([c for c in self.FEATURE_COLS if c in valid_df.columns]).to_numpy()
        y = valid_df.get_column(target_column).to_numpy()

        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict(self, features_df: pl.DataFrame) -> ClassificationResult:
        """Predict disturbance class and class probabilities for input features.

        Returns:
            ClassificationResult containing Polars DataFrame with:
            - PREDICTED_DISTURBANCE_CLASS (String)
            - PROB_NORMAL, PROB_SUBTENSAO, PROB_INTERRUPCAO_TOTAL, etc. (Float)
            - CONFIDENCE_SCORE (Float)
        """
        if not self.is_fitted or self.classes_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        if features_df.is_empty():
            return ClassificationResult(predictions_df=features_df, feature_names=self.FEATURE_COLS)

        X = features_df.select([c for c in self.FEATURE_COLS if c in features_df.columns]).to_numpy()
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        prob_series = [
            pl.Series(f"PROB_{cls_name}", np.round(probs[:, i], 4))
            for i, cls_name in enumerate(self.classes_)
        ]

        max_probs = np.max(probs, axis=1)

        result_df = features_df.with_columns([
            pl.Series("PREDICTED_DISTURBANCE_CLASS", preds),
            pl.Series("DISTURBANCE_CONFIDENCE", np.round(max_probs, 4)),
            *prob_series,
        ])

        return ClassificationResult(predictions_df=result_df, feature_names=self.FEATURE_COLS)
