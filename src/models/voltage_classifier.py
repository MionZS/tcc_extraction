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
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


class VoltageDisturbanceType(str, Enum):
    """Categories of voltage disturbances and faults per PRODIST Módulo 8 and IEEE Std 1159."""

    NORMAL = "NORMAL"
    SUBTENSAO = "SUBTENSAO"
    SOBRETENSAO = "SOBRETENSAO"
    FALTA_MONOFASICA = "FALTA_MONOFASICA"
    FALTA_BIFASICA = "FALTA_BIFASICA"
    INTERRUPCAO_TOTAL = "INTERRUPCAO_TOTAL"
    DESBALANCEAMENTO_CRITICO = "DESBALANCEAMENTO_CRITICO"
    DESCONEXAO_TELEMETRIA = "DESCONEXAO_TELEMETRIA"


# PRODIST Módulo 8 (Seção 8.1 - Tensão em Regime Permanente) & IEEE 1159 thresholds in p.u.
PU_SUBTENSAO_THRESHOLD = 0.90  # Limite inferior adequado PRODIST / IEEE Sag threshold
PU_SOBRETENSAO_THRESHOLD = 1.05  # Limite superior adequado PRODIST / IEEE Swell threshold
PU_INTERRUPCAO_MAX_VOLTS = 5.0  # Tensão abaixo de 5V é considerada interrupção/falta completa (<= 0.04 p.u.)
PU_INTERRUPCAO_THRESHOLD = 0.05  # Tensão p.u. indicando interrupção de fase
PRODIST_DESBALANCEAMENTO_CRITICO = 0.03  # PRODIST Módulo 8 Seção 8.2: FD% crítico > 3.0%
PRODIST_DESBALANCEAMENTO_ALERTA = 0.02  # Limite contínuo de referência: 2.0%


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
    history_df: pl.DataFrame | None = None,
    *,
    tensao_base_default: float = 127.0,
) -> pl.DataFrame:
    """Extract normalized features and missing data indicators for voltage classification.

    Handles structural missingness, zero measurements, and telemetry outage:
    1. Structural Nulls: Phase physically absent for the UC (e.g., L3 in Monofásico/Bifásico).
    2. Zero Voltages: Telemetry active, but measured voltage <= 5.0V.
    3. Complete Telemetry Absence: No packet received. Differentiates grid blackout from
       meter communication outage using neighborhood trafo peer coverage and last gasp alarms.
    4. PRODIST Imbalance: Correct PVUR computation restricted exclusively to active valid phases.

    Args:
        instantaneous_df: Telemetry dataframe with U_L1, U_L2, U_L3 or U_L1_AVG...
        context_df: UC context dataframe with UC, FASE_CIRCUITO, TENSAO_BASE.
        neighborhood_df: Trafo neighborhood features (NEIGHBOR_COVERAGE_RATIO, etc.).
        alarms_df: Alarm events dataframe (power outage / last gasp alarms).
        history_df: Optional historical baseline dataframe with UC, V_HIST_MEAN_PU.
        tensao_base_default: Fallback nominal voltage in Volts (e.g. 127.0 or 220.0).

    Returns:
        Polars DataFrame with features ready for rule-based or ML classification.
    """
    if instantaneous_df.is_empty():
        return instantaneous_df

    df = instantaneous_df.clone()

    # Join context if available (match on UC, or on NIO, or bridge if only one key is present)
    if context_df is not None and not context_df.is_empty():
        context_cols = [c for c in ["UC", "NIO", "FASE_CIRCUITO", "TENSAO_BASE"] if c in context_df.columns]
        sub_context = context_df.select(context_cols).drop_nulls(subset=[c for c in ["UC", "NIO"] if c in context_cols]).unique()

        if "UC" in df.columns and "UC" in sub_context.columns:
            join_cols = [c for c in ["UC", "FASE_CIRCUITO", "TENSAO_BASE"] if c in sub_context.columns]
            df = df.join(sub_context.select(join_cols), on="UC", how="left")
        elif "NIO" in df.columns and "NIO" in sub_context.columns:
            join_cols = [c for c in ["NIO", "UC", "FASE_CIRCUITO", "TENSAO_BASE"] if c in sub_context.columns]
            df = df.join(sub_context.select(join_cols), on="NIO", how="left")
        elif "NIO" in df.columns and "UC" in sub_context.columns and "NIO" not in sub_context.columns:
            # If sub_context only has UC, check if UC values correspond to NIOs
            sub_renamed = sub_context.rename({"UC": "NIO"})
            join_cols = [c for c in ["NIO", "FASE_CIRCUITO", "TENSAO_BASE"] if c in sub_renamed.columns]
            df = df.join(sub_renamed.select(join_cols), on="NIO", how="left")

    # Determine base voltage column or fallback
    if "TENSAO_BASE" not in df.columns:
        df = df.with_columns(pl.lit(tensao_base_default).alias("TENSAO_BASE"))
    else:
        df = df.with_columns(pl.col("TENSAO_BASE").fill_null(tensao_base_default))

    # Identify phase voltage columns
    u_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in df.columns]
    if not u_cols:
        u_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in df.columns]

    l1_col = next((c for c in u_cols if "L1" in c), None)
    l2_col = next((c for c in u_cols if "L2" in c), None)
    l3_col = next((c for c in u_cols if "L3" in c), None)

    # 1. Structural Phase Activation Flags
    # If FASE_CIRCUITO is available from context:
    #   Monofásico -> L1 active; Bifásico -> L1, L2 active; Trifásico -> L1, L2, L3 active
    # If FASE_CIRCUITO is null or absent, do NOT invent data: infer physical presence
    # from available telemetry columns and observed values (e.g., presence of non-null L2/L3 readings).
    has_l2_phys = (l2_col is not None) and (df.select(pl.col(l2_col).is_not_null().any()).item() is True)
    has_l3_phys = (l3_col is not None) and (df.select(pl.col(l3_col).is_not_null().any()).item() is True)

    if "FASE_CIRCUITO" in df.columns:
        phase_col = pl.col("FASE_CIRCUITO").str.to_uppercase()
        is_active_l2 = (
            pl.when(phase_col.is_not_null())
            .then(phase_col.str.contains("BIFASICO") | phase_col.str.contains("TRIFASICO"))
            .otherwise(pl.lit(has_l2_phys))
            .alias("IS_ACTIVE_L2")
        )
        is_active_l3 = (
            pl.when(phase_col.is_not_null())
            .then(phase_col.str.contains("TRIFASICO"))
            .otherwise(pl.lit(has_l3_phys))
            .alias("IS_ACTIVE_L3")
        )
    else:
        is_active_l2 = pl.lit(has_l2_phys).alias("IS_ACTIVE_L2")
        is_active_l3 = pl.lit(has_l3_phys).alias("IS_ACTIVE_L3")

    df = df.with_columns([
        pl.lit(True).alias("IS_ACTIVE_L1"),
        is_active_l2,
        is_active_l3,
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

        # Zero indicator (telemetry received, but phase voltage <= PU_INTERRUPCAO_MAX_VOLTS)
        (l1_val.fill_null(999.0) <= PU_INTERRUPCAO_MAX_VOLTS).alias("IS_ZERO_L1"),
        (l2_val.fill_null(999.0) <= PU_INTERRUPCAO_MAX_VOLTS).alias("IS_ZERO_L2"),
        (l3_val.fill_null(999.0) <= PU_INTERRUPCAO_MAX_VOLTS).alias("IS_ZERO_L3"),
    ])

    # 3. Dynamic Aggregations & Imbalance Ratio exclusively over active valid phases
    count_active_valid = sum(
        pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"U_L{i}_PU").is_not_null()).then(1).otherwise(0)
        for i in (1, 2, 3)
    )

    count_active_total = sum(
        pl.when(pl.col(f"IS_ACTIVE_L{i}")).then(1).otherwise(0)
        for i in (1, 2, 3)
    )

    # Sum only active and non-null phases
    sum_active_pu = sum(
        pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"U_L{i}_PU").is_not_null())
        .then(pl.col(f"U_L{i}_PU"))
        .otherwise(0.0)
        for i in (1, 2, 3)
    )

    v_avg_pu = pl.when(count_active_valid > 0).then(sum_active_pu / count_active_valid).otherwise(0.0)

    # Compute deviance strictly for active valid phases
    dev_l1 = (pl.col("U_L1_PU") - v_avg_pu).abs()
    dev_l2 = (pl.col("U_L2_PU") - v_avg_pu).abs()
    dev_l3 = (pl.col("U_L3_PU") - v_avg_pu).abs()

    max_dev = pl.max_horizontal([
        pl.when(pl.col("IS_ACTIVE_L1") & pl.col("U_L1_PU").is_not_null()).then(dev_l1).otherwise(0.0),
        pl.when(pl.col("IS_ACTIVE_L2") & pl.col("U_L2_PU").is_not_null()).then(dev_l2).otherwise(0.0),
        pl.when(pl.col("IS_ACTIVE_L3") & pl.col("U_L3_PU").is_not_null()).then(dev_l3).otherwise(0.0),
    ])

    # PVUR: max(|V_k - V_avg|) / V_avg for polyphase circuits (bifásico / trifásico)
    v_imb_ratio = pl.when(
        (v_avg_pu > PU_INTERRUPCAO_THRESHOLD) & (count_active_valid >= 2)
    ).then(max_dev / v_avg_pu).otherwise(0.0)

    # Clean Min & Max over active valid phases without forcing zero on nulls
    active_valid_l1 = pl.when(pl.col("IS_ACTIVE_L1") & pl.col("U_L1_PU").is_not_null()).then(pl.col("U_L1_PU"))
    active_valid_l2 = pl.when(pl.col("IS_ACTIVE_L2") & pl.col("U_L2_PU").is_not_null()).then(pl.col("U_L2_PU"))
    active_valid_l3 = pl.when(pl.col("IS_ACTIVE_L3") & pl.col("U_L3_PU").is_not_null()).then(pl.col("U_L3_PU"))

    v_min_pu = pl.min_horizontal([active_valid_l1, active_valid_l2, active_valid_l3])
    v_max_pu = pl.max_horizontal([active_valid_l1, active_valid_l2, active_valid_l3])

    # Spread between max and min phase
    v_spread_pu = pl.when(count_active_valid >= 2).then(v_max_pu - v_min_pu).otherwise(0.0)

    # Ratios of zero and null phases
    zero_phase_count = sum(
        pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"IS_ZERO_L{i}")).then(1).otherwise(0)
        for i in (1, 2, 3)
    )
    null_phase_count = sum(
        pl.when(pl.col(f"IS_ACTIVE_L{i}") & pl.col(f"IS_NULL_L{i}")).then(1).otherwise(0)
        for i in (1, 2, 3)
    )

    ratio_zero_phases = pl.when(count_active_total > 0).then(zero_phase_count / count_active_total).otherwise(0.0)
    ratio_null_phases = pl.when(count_active_total > 0).then(null_phase_count / count_active_total).otherwise(0.0)

    df = df.with_columns([
        v_avg_pu.alias("V_AVG_PU"),
        v_min_pu.alias("V_MIN_PU"),
        v_max_pu.alias("V_MAX_PU"),
        v_spread_pu.alias("V_SPREAD_PU"),
        v_imb_ratio.alias("VOLTAGE_IMBALANCE_RATIO"),
        count_active_valid.alias("ACTIVE_VALID_PHASES_COUNT"),
        count_active_total.alias("ACTIVE_TOTAL_PHASES_COUNT"),
        ratio_zero_phases.alias("RATIO_ZERO_PHASES"),
        ratio_null_phases.alias("RATIO_NULL_PHASES"),
    ])

    # 4. Integrate Spatial Neighborhood Context (Trafo Peer Coverage)
    if neighborhood_df is not None and not neighborhood_df.is_empty():
        if "UC" in df.columns and "UC" in neighborhood_df.columns:
            neigh_cols = [c for c in ["UC", "NEIGHBOR_COVERAGE_RATIO", "TRAFO_PEER_COUNT"] if c in neighborhood_df.columns]
            df = df.join(neighborhood_df.select(neigh_cols), on="UC", how="left")
        elif "NIO" in df.columns and "NIO" in neighborhood_df.columns:
            neigh_cols = [c for c in ["NIO", "NEIGHBOR_COVERAGE_RATIO", "TRAFO_PEER_COUNT"] if c in neighborhood_df.columns]
            df = df.join(neighborhood_df.select(neigh_cols), on="NIO", how="left")
        elif "NIO" in df.columns and "UC" in neighborhood_df.columns and "NIO" not in neighborhood_df.columns:
            neigh_cols = [c for c in ["UC", "NEIGHBOR_COVERAGE_RATIO", "TRAFO_PEER_COUNT"] if c in neighborhood_df.columns]
            df = df.join(neighborhood_df.select(neigh_cols).rename({"UC": "NIO"}), on="NIO", how="left")

    if "NEIGHBOR_COVERAGE_RATIO" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("NEIGHBOR_COVERAGE_RATIO"))
    else:
        df = df.with_columns(pl.col("NEIGHBOR_COVERAGE_RATIO").fill_null(1.0))

    # 5. Integrate Power Failure Alarms (Last Gasp)
    if alarms_df is not None and not alarms_df.is_empty():
        alarm_mask = pl.col("CONTENT").str.to_uppercase().str.contains("FALTA|POWER|LAST_GASP|QUEDA|OUTAGE") if "CONTENT" in alarms_df.columns else pl.lit(True)
        filtered_alarms = alarms_df.filter(alarm_mask)
        alarm_keys = set()
        if "NIO" in filtered_alarms.columns:
            alarm_keys.update(filtered_alarms.get_column("NIO").drop_nulls().to_list())
        if "UC" in filtered_alarms.columns:
            alarm_keys.update(filtered_alarms.get_column("UC").drop_nulls().to_list())

        if "NIO" in df.columns and "UC" in df.columns:
            df = df.with_columns((pl.col("NIO").is_in(alarm_keys) | pl.col("UC").is_in(alarm_keys)).alias("HAS_POWER_ALARM"))
        elif "NIO" in df.columns:
            df = df.with_columns(pl.col("NIO").is_in(alarm_keys).alias("HAS_POWER_ALARM"))
        elif "UC" in df.columns:
            df = df.with_columns(pl.col("UC").is_in(alarm_keys).alias("HAS_POWER_ALARM"))
        else:
            df = df.with_columns(pl.lit(False).alias("HAS_POWER_ALARM"))
    else:
        df = df.with_columns(pl.lit(False).alias("HAS_POWER_ALARM"))

    # 6. Integrate Temporal Baseline Context (Delta vs Historic Voltage)
    if history_df is not None and not history_df.is_empty():
        if "UC" in df.columns and "UC" in history_df.columns and "V_HIST_MEAN_PU" in history_df.columns:
            df = df.join(history_df.select(["UC", "V_HIST_MEAN_PU"]), on="UC", how="left")
            df = df.with_columns((pl.col("V_AVG_PU") - pl.col("V_HIST_MEAN_PU")).alias("HIST_DELTA_V_PU"))
        elif "NIO" in df.columns and "NIO" in history_df.columns and "V_HIST_MEAN_PU" in history_df.columns:
            df = df.join(history_df.select(["NIO", "V_HIST_MEAN_PU"]), on="NIO", how="left")
            df = df.with_columns((pl.col("V_AVG_PU") - pl.col("V_HIST_MEAN_PU")).alias("HIST_DELTA_V_PU"))
        else:
            df = df.with_columns(pl.lit(0.0).alias("HIST_DELTA_V_PU"))
    else:
        df = df.with_columns(pl.lit(0.0).alias("HIST_DELTA_V_PU"))

    return df


class RuleBasedVoltageClassifier:
    """Deterministic Baseline Classifier following PRODIST Módulo 8 & IEEE 1159."""

    @staticmethod
    def classify(df: pl.DataFrame) -> pl.DataFrame:
        """Assign baseline disturbance label using physical and normative criteria."""
        if df.is_empty():
            return df

        # Phase counts
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

        # 1. Interrupção Total: Todas as fases zeradas, ou falta de telemetria com contexto de apagão/alarme
        is_total_interruption = (
            (zero_phase_count == active_phase_count)
            | (
                (null_phase_count == active_phase_count)
                & ((pl.col("NEIGHBOR_COVERAGE_RATIO") <= 0.3) | pl.col("HAS_POWER_ALARM"))
            )
        )

        # 2. Faltas Assimétricas (Perda de fase em circuito bifásico ou trifásico)
        is_monofasica = (zero_phase_count == 1) & (active_phase_count >= 2)
        is_bifasica = (zero_phase_count == 2) & (active_phase_count == 3)

        # 3. Desconexão de Telemetria (ausência de dados sem evidência de apagão)
        is_telemetry_outage = (
            (null_phase_count == active_phase_count)
            & (pl.col("NEIGHBOR_COVERAGE_RATIO") > 0.3)
            & (~pl.col("HAS_POWER_ALARM"))
        )

        # 4. Perturbações PRODIST Módulo 8 (Subtensão, Sobretensão e Desbalanceamento)
        # Afundamento/Subtensão: V_MIN_PU abaixo de 0.90 p.u. (mas acima do limiar de falta de fase)
        is_subtensao = (
            (pl.col("V_MIN_PU") > PU_INTERRUPCAO_THRESHOLD)
            & (pl.col("V_MIN_PU") < PU_SUBTENSAO_THRESHOLD)
            & (~is_monofasica)
            & (~is_bifasica)
            & (~is_total_interruption)
        )

        # Sobretensão / Elevação: V_MAX_PU acima de 1.05 p.u.
        is_sobretensao = (
            (pl.col("V_MAX_PU") > PU_SOBRETENSAO_THRESHOLD)
            & (~is_total_interruption)
            & (~is_monofasica)
            & (~is_bifasica)
        )

        # Desbalanceamento Crítico: FD >= 3.0% (PRODIST Módulo 8 Seção 8.2) e tensão em níveis operacionais
        is_desbalanceamento = (
            (pl.col("VOLTAGE_IMBALANCE_RATIO") >= PRODIST_DESBALANCEAMENTO_CRITICO)
            & (pl.col("V_MIN_PU") >= PU_SUBTENSAO_THRESHOLD)
            & (~is_monofasica)
            & (~is_bifasica)
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


def group_train_test_split(
    df: pl.DataFrame,
    *,
    group_col: str = "UC",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split DataFrame into train and test sets grouped by UC to prevent data leakage."""
    if df.is_empty():
        return df, df

    rng = np.random.default_rng(random_state)
    unique_groups = df.get_column(group_col).drop_nulls().unique().to_list()
    rng.shuffle(unique_groups)

    n_test = max(1, int(len(unique_groups) * test_size))
    test_groups = set(unique_groups[:n_test])

    train_df = df.filter(~pl.col(group_col).is_in(test_groups))
    test_df = df.filter(pl.col(group_col).is_in(test_groups))

    return train_df, test_df


class MLVoltageDisturbanceClassifier:
    """Supervised ML Classifier (HistGradientBoosting) for Voltage Disturbances.

    Trained on extracted voltage features (including missing value indicators & spatial peer context).
    Natively handles missing/NaN values and class imbalance with balanced sample weights.
    """

    FEATURE_COLS = [
        "U_L1_PU",
        "U_L2_PU",
        "U_L3_PU",
        "V_MIN_PU",
        "V_MAX_PU",
        "V_AVG_PU",
        "V_SPREAD_PU",
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
        "RATIO_ZERO_PHASES",
        "RATIO_NULL_PHASES",
        "NEIGHBOR_COVERAGE_RATIO",
        "HAS_POWER_ALARM",
        "HIST_DELTA_V_PU",
    ]

    def __init__(
        self,
        *,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        early_stopping: bool | str = "auto",
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.classes_: np.ndarray | None = None
        self.is_fitted: bool = False
        self.model: HistGradientBoostingClassifier | None = None

    def _prepare_matrix(self, features_df: pl.DataFrame) -> np.ndarray:
        """Convert Polars DataFrame to a clean float32 matrix with exact column order."""
        missing = [c for c in self.FEATURE_COLS if c not in features_df.columns]
        if missing:
            raise KeyError(f"Missing required feature columns for voltage classification: {missing}")

        exprs = []
        for c in self.FEATURE_COLS:
            dtype = features_df.schema[c]
            if dtype == pl.Boolean:
                exprs.append(pl.col(c).cast(pl.Float32).alias(c))
            else:
                exprs.append(pl.col(c).cast(pl.Float32).alias(c))

        return features_df.select(exprs).to_numpy()

    def fit(
        self,
        features_df: pl.DataFrame,
        target_column: str = "BASELINE_VOLTAGE_LABEL",
    ) -> MLVoltageDisturbanceClassifier:
        """Fit the ML model using labeled data (from baseline rules or field events)."""
        valid_df = features_df.filter(pl.col(target_column).is_not_null())
        if valid_df.height < 10:
            raise ValueError(f"Insufficient training samples ({valid_df.height}) for ML voltage classifier.")

        X = self._prepare_matrix(valid_df)
        y = valid_df.get_column(target_column).to_numpy()

        n_samples = len(y)
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        # Determine adaptive early stopping
        if self.early_stopping == "auto":
            use_early_stopping = n_samples >= 200
        else:
            use_early_stopping = bool(self.early_stopping)

        # Determine safe min_samples_leaf
        effective_min_leaf = min(self.min_samples_leaf, max(1, n_samples // (n_classes * 2)))

        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            min_samples_leaf=effective_min_leaf,
            class_weight="balanced",
            early_stopping=use_early_stopping,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=self.random_state,
        )

        self.model.fit(X, y)
        self.classes_ = np.array(self.model.classes_)
        self.is_fitted = True
        return self

    def predict(self, features_df: pl.DataFrame) -> ClassificationResult:
        """Predict disturbance class and class probabilities for input features.

        Returns:
            ClassificationResult containing Polars DataFrame with:
            - PREDICTED_DISTURBANCE_CLASS (String)
            - DISTURBANCE_CONFIDENCE (Float)
            - PROB_<CLASS> for each known class
        """
        if not self.is_fitted or self.classes_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        if features_df.is_empty():
            return ClassificationResult(predictions_df=features_df, feature_names=self.FEATURE_COLS)

        X = self._prepare_matrix(features_df)
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

    def evaluate(
        self,
        test_df: pl.DataFrame,
        target_column: str = "BASELINE_VOLTAGE_LABEL",
    ) -> dict[str, Any]:
        """Compute performance metrics, classification report, and confusion matrix on test data."""
        if not self.is_fitted or self.classes_ is None:
            raise RuntimeError("Model must be fitted before calling evaluate().")

        valid_df = test_df.filter(pl.col(target_column).is_not_null())
        if valid_df.is_empty():
            return {"error": "No valid labeled test samples."}

        X = self._prepare_matrix(valid_df)
        y_true = valid_df.get_column(target_column).to_numpy()
        y_pred = self.model.predict(X)

        acc = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)
        report_dict = classification_report(y_true, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "classes": self.classes_.tolist(),
            "confusion_matrix": cm.tolist(),
            "report": report_dict,
        }

    def save(self, filepath: Path | str) -> None:
        """Persist fitted model and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Path | str) -> MLVoltageDisturbanceClassifier:
        """Load a persisted classifier instance from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is {type(instance)}, expected {cls}")
        return instance
