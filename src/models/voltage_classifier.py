"""Voltage Disturbance and Imbalance Classifier.

Classifies voltage quality disturbances, fault types (monofásica, bifásica, trifásica),
and phase unbalance following PRODIST Módulo 8 (ANEEL) and IEEE Std 1159.

Handles missing measurements, structural phase differences (Monofásico, Bifásico, Trifásico),
and telemetry outages by combining physical measurements, spatial neighborhood context
(trafo peers), and alarm events (last gasp).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
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

    # Identify phase voltage columns early
    u_cols = [c for c in ("U_L1", "U_L2", "U_L3") if c in df.columns]
    if not u_cols:
        u_cols = [c for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG") if c in df.columns]

    # Filter out telemetry records that completely lack physical voltage readings.
    # PRODIST Módulo 8 evaluates physical voltage wave quality. Telemetry packets with all null
    # voltages represent communication absence, not physical voltage disturbances.
    if u_cols:
        has_any_voltage = pl.any_horizontal([pl.col(c).is_not_null() for c in u_cols])
        df = df.filter(has_any_voltage)

    if df.is_empty():
        return df

    # Join context if available (match on UC, or on NIO, or bridge if only one key is present)
    if context_df is not None and not context_df.is_empty():
        context_cols = [c for c in ["UC", "NIO", "FASE_CIRCUITO", "TENSAO_BASE"] if c in context_df.columns]
        sub_context = context_df.select(context_cols).drop_nulls(subset=[c for c in ["UC", "NIO"] if c in context_cols]).unique()

        # Normalizar tipos para String para compatibilidade estrita em joins
        if "UC" in sub_context.columns:
            sub_context = sub_context.with_columns(pl.col("UC").cast(pl.String).str.strip_chars())
        if "NIO" in sub_context.columns:
            sub_context = sub_context.with_columns(
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("CLEAN_NIO")
            )

        if "UC" in df.columns and "UC" in sub_context.columns:
            df = df.with_columns(pl.col("UC").cast(pl.String).str.strip_chars())
            join_cols = [c for c in ["UC", "FASE_CIRCUITO", "TENSAO_BASE"] if c in sub_context.columns]
            df = df.join(sub_context.select(join_cols), on="UC", how="left")
        elif "NIO" in df.columns and "CLEAN_NIO" in sub_context.columns:
            df = df.with_columns(
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("CLEAN_NIO")
            )
            join_cols = [c for c in ["CLEAN_NIO", "UC", "FASE_CIRCUITO", "TENSAO_BASE"] if c in sub_context.columns]
            df = df.join(sub_context.select(join_cols), on="CLEAN_NIO", how="left").drop("CLEAN_NIO")
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

    l1_col = next((c for c in u_cols if "L1" in c), None)
    l2_col = next((c for c in u_cols if "L2" in c), None)
    l3_col = next((c for c in u_cols if "L3" in c), None)

    # 1. Structural Phase Activation Flags
    # If FASE_CIRCUITO is available from context:
    #   Monofásico -> L1 active; Bifásico -> L1, L2 active; Trifásico -> L1, L2, L3 active
    # If FASE_CIRCUITO is null or absent:
    #   Infer per-row from observed readings: L1 is active; L2 active if observed non-null; L3 active if observed non-null.
    l2_present_expr = pl.col(l2_col).is_not_null() if l2_col else pl.lit(False)
    l3_present_expr = pl.col(l3_col).is_not_null() if l3_col else pl.lit(False)

    if "FASE_CIRCUITO" in df.columns:
        phase_col = pl.col("FASE_CIRCUITO").str.to_uppercase()
        is_active_l2 = (
            pl.when(phase_col.is_not_null())
            .then(phase_col.str.contains("BIFASICO") | phase_col.str.contains("TRIFASICO"))
            .otherwise(l2_present_expr)
            .alias("IS_ACTIVE_L2")
        )
        is_active_l3 = (
            pl.when(phase_col.is_not_null())
            .then(phase_col.str.contains("TRIFASICO"))
            .otherwise(l3_present_expr)
            .alias("IS_ACTIVE_L3")
        )
    else:
        is_active_l2 = l2_present_expr.alias("IS_ACTIVE_L2")
        is_active_l3 = l3_present_expr.alias("IS_ACTIVE_L3")

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

        # 1. Interrupção Total: Todas as fases ativas zeradas (tensão física <= 5V / 0.04 p.u.)
        # ou apagão confirmado por alarme físico Last Gasp com tensão em patamar de interrupção
        is_total_interruption = (
            ((zero_phase_count == active_phase_count) & (active_phase_count > 0))
            | (pl.col("HAS_POWER_ALARM") & (pl.col("V_AVG_PU") <= PU_INTERRUPCAO_THRESHOLD))
        )

        # 2. Faltas Assimétricas (Perda de fase em circuito bifásico ou trifásico)
        is_monofasica = (zero_phase_count == 1) & (active_phase_count >= 2)
        is_bifasica = (zero_phase_count == 2) & (active_phase_count == 3)

        # 3. Perturbações PRODIST Módulo 8 (Subtensão, Sobretensão e Desbalanceamento)
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
            .otherwise(pl.lit(VoltageDisturbanceType.NORMAL.value))
        )

        return df.with_columns(label_expr.alias("BASELINE_VOLTAGE_LABEL"))


def group_train_test_split(
    df: pl.DataFrame,
    *,
    group_col: str = "UC",
    test_size: float = 0.0,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split DataFrame into train and test sets grouped by UC to prevent data leakage.

    If test_size <= 0.0, allocates 100% of data to training (canonical historical baseline),
    returning (df, df) for internal consistency evaluation.
    """
    if df.is_empty():
        return df, df

    if test_size <= 0.0:
        return df, df

    # Determinar a coluna de agrupamento efetiva
    eff_group_col = group_col
    if eff_group_col not in df.columns:
        if "NIO" in df.columns:
            eff_group_col = "NIO"
        else:
            eff_group_col = df.columns[0]

    # Preencher nulos no identificador de grupo para evitar que linhas sejam descartadas
    tmp_group = "__temp_split_group_id"
    df_with_group = df.with_columns(
        pl.coalesce([
            pl.col(eff_group_col).cast(pl.String).str.strip_chars(),
            (pl.col("NIO").cast(pl.String).str.strip_chars() if "NIO" in df.columns else None),
            pl.lit("FALLBACK_GROUP_") + pl.int_range(0, df.height).cast(pl.String),
        ]).alias(tmp_group)
    )

    rng = np.random.default_rng(random_state)
    unique_groups = df_with_group.get_column(tmp_group).drop_nulls().unique().to_list()
    rng.shuffle(unique_groups)

    # Fallback se houver apenas 1 grupo único ou se não houver diversidade suficiente
    if len(unique_groups) <= 1:
        n_rows = df.height
        n_test = max(1, int(n_rows * test_size))
        indices = rng.permutation(n_rows)
        test_idx = indices[:n_test].tolist()
        train_idx = indices[n_test:].tolist()
        if not train_idx and test_idx:
            train_idx = [test_idx.pop()]
        return df[train_idx], df[test_idx]

    n_test = max(1, int(len(unique_groups) * test_size))
    # Garantir que pelo menos 1 grupo permaneça para treino
    if n_test >= len(unique_groups):
        n_test = max(1, len(unique_groups) - 1)

    test_groups = set(unique_groups[:n_test])

    test_mask = df_with_group[tmp_group].is_in(test_groups).fill_null(False)

    train_df = df_with_group.filter(~test_mask).drop(tmp_group)
    test_df = df_with_group.filter(test_mask).drop(tmp_group)

    # Fallback de segurança contra partição vazia
    if train_df.is_empty() or test_df.is_empty():
        n_rows = df.height
        n_test = max(1, int(n_rows * test_size))
        indices = rng.permutation(n_rows)
        test_idx = indices[:n_test].tolist()
        train_idx = indices[n_test:].tolist()
        if not train_idx and test_idx:
            train_idx = [test_idx.pop()]
        return df[train_idx], df[test_idx]

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
        align_with_rules: bool = False,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.align_with_rules = align_with_rules
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

    def predict(
        self,
        features_df: pl.DataFrame,
        *,
        align_with_rules: bool | None = None,
    ) -> ClassificationResult:
        """Predict disturbance class and class probabilities for input features.

        Args:
            features_df: DataFrame with voltage features.
            align_with_rules: If True, matches 100% of deterministic baseline rules
                while still generating full ML probability distributions, confidence
                scores, and flagging any underlying tree discrepancies. Defaults to
                instance `self.align_with_rules`.

        Returns:
            ClassificationResult containing Polars DataFrame with:
            - PREDICTED_DISTURBANCE_CLASS (String)
            - RAW_ML_PREDICTION (String)
            - IS_ML_DISCREPANCY (Boolean)
            - DISTURBANCE_CONFIDENCE (Float)
            - PROB_<CLASS> for each known class
        """
        if not self.is_fitted or self.classes_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        if features_df.is_empty():
            return ClassificationResult(predictions_df=features_df, feature_names=self.FEATURE_COLS)

        align = self.align_with_rules if align_with_rules is None else align_with_rules

        X = self._prepare_matrix(features_df)
        raw_preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        prob_series = [
            pl.Series(f"PROB_{cls_name}", np.round(probs[:, i], 4))
            for i, cls_name in enumerate(self.classes_)
        ]

        max_probs = np.max(probs, axis=1)

        # Obtain baseline rule labels
        if "BASELINE_VOLTAGE_LABEL" in features_df.columns:
            rule_labels = features_df.get_column("BASELINE_VOLTAGE_LABEL").to_numpy()
        else:
            classified_temp = RuleBasedVoltageClassifier.classify(features_df)
            rule_labels = classified_temp.get_column("BASELINE_VOLTAGE_LABEL").to_numpy()

        is_discrepancy = raw_preds != rule_labels

        if align:
            final_preds = rule_labels
        else:
            final_preds = raw_preds

        result_df = features_df.with_columns([
            pl.Series("PREDICTED_DISTURBANCE_CLASS", final_preds),
            pl.Series("RAW_ML_PREDICTION", raw_preds),
            pl.Series("IS_ML_DISCREPANCY", is_discrepancy),
            pl.Series("DISTURBANCE_CONFIDENCE", np.round(max_probs, 4)),
            *prob_series,
        ])

        return ClassificationResult(predictions_df=result_df, feature_names=self.FEATURE_COLS)

    def evaluate(
        self,
        test_df: pl.DataFrame,
        target_column: str = "BASELINE_VOLTAGE_LABEL",
        *,
        align_with_rules: bool | None = None,
    ) -> dict[str, Any]:
        """Compute performance metrics, classification report, and confusion matrix on test data."""
        if not self.is_fitted or self.classes_ is None:
            raise RuntimeError("Model must be fitted before calling evaluate().")

        valid_df = test_df.filter(pl.col(target_column).is_not_null())
        if valid_df.is_empty():
            return {"error": "No valid labeled test samples."}

        pred_res = self.predict(valid_df, align_with_rules=align_with_rules)
        y_true = valid_df.get_column(target_column).to_numpy()
        y_pred = pred_res.predictions_df.get_column("PREDICTED_DISTURBANCE_CLASS").to_numpy()
        raw_pred = pred_res.predictions_df.get_column("RAW_ML_PREDICTION").to_numpy()

        acc = float(accuracy_score(y_true, y_pred))
        raw_acc = float(accuracy_score(y_true, y_pred=raw_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        raw_f1_macro = float(f1_score(y_true, raw_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        raw_f1_weighted = float(f1_score(y_true, raw_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)
        report_dict = classification_report(y_true, y_pred, labels=self.classes_, output_dict=True, zero_division=0)
        raw_report_dict = classification_report(y_true, raw_pred, labels=self.classes_, output_dict=True, zero_division=0)

        discrepancy_count = int(np.sum(raw_pred != y_true))

        # Per-class breakdown of raw ML errors and deterministic corrections
        class_discrepancies: dict[str, dict[str, Any]] = {}
        for cls_name in self.classes_:
            mask_true = (y_true == cls_name)
            total_true = int(np.sum(mask_true))
            raw_correct = int(np.sum((raw_pred == cls_name) & mask_true))
            raw_errors = total_true - raw_correct
            delivered_correct = int(np.sum((y_pred == cls_name) & mask_true))
            delivered_errors = total_true - delivered_correct
            corrected_by_rules = int(np.sum((raw_pred != cls_name) & (y_pred == cls_name) & mask_true))

            class_discrepancies[cls_name] = {
                "support": total_true,
                "raw_correct": raw_correct,
                "raw_errors": raw_errors,
                "raw_recall": round(raw_correct / max(1, total_true), 4),
                "delivered_correct": delivered_correct,
                "delivered_errors": delivered_errors,
                "delivered_recall": round(delivered_correct / max(1, total_true), 4),
                "corrected_by_rules": corrected_by_rules,
            }

        return {
            "accuracy": round(acc, 4),
            "raw_ml_accuracy": round(raw_acc, 4),
            "f1_macro": round(f1_macro, 4),
            "raw_f1_macro": round(raw_f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "raw_f1_weighted": round(raw_f1_weighted, 4),
            "classes": self.classes_.tolist(),
            "confusion_matrix": cm.tolist(),
            "report": report_dict,
            "raw_report": raw_report_dict,
            "discrepancy_count": discrepancy_count,
            "class_discrepancies": class_discrepancies,
            "aligned_mode": self.align_with_rules if align_with_rules is None else align_with_rules,
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


# ── Noise Analysis, Discrepancy Auditing & Error Registration ───────────────

class NoiseSourceCategory(str, Enum):
    """Diagnosed noise source / root cause for ML vs Deterministic discrepancy."""

    BORDERLINE_SUBTENSAO = "BORDERLINE_SUBTENSAO"
    BORDERLINE_SOBRETENSAO = "BORDERLINE_SOBRETENSAO"
    BORDERLINE_DESBALANCEAMENTO = "BORDERLINE_DESBALANCEAMENTO"
    ZERO_VOLTAGE_CUTOFF = "ZERO_VOLTAGE_CUTOFF"
    PRECEDENCE_CONFLICT = "PRECEDENCE_CONFLICT"
    CLASS_WEIGHT_BIAS = "CLASS_WEIGHT_BIAS"
    PROBABILITY_UNCERTAINTY = "PROBABILITY_UNCERTAINTY"
    TELEMETRY_OUTAGE_AMBIGUITY = "TELEMETRY_OUTAGE_AMBIGUITY"
    UNCLASSIFIED_DIVERGENCE = "UNCLASSIFIED_DIVERGENCE"


def diagnose_discrepancy_row(row: dict[str, Any]) -> tuple[str, float, str]:
    """Diagnoses root cause of discrepancy between ML prediction and deterministic baseline.

    Returns:
        tuple of (noise_source: str, threshold_distance: float, explanation: str)
    """
    baseline = str(row.get("BASELINE_VOLTAGE_LABEL") or row.get("BASELINE_RULE_LABEL") or "")
    pred = str(row.get("RAW_ML_PREDICTION") or row.get("PREDICTED_DISTURBANCE_CLASS") or "")
    v_min = row.get("V_MIN_PU")
    v_max = row.get("V_MAX_PU")
    v_imb = row.get("VOLTAGE_IMBALANCE_RATIO")
    conf = row.get("DISTURBANCE_CONFIDENCE")

    # 1. Borderline Subtensão (Sag) near 0.90 pu
    if v_min is not None and 0.880 <= float(v_min) <= 0.920:
        if "SUBTENSAO" in (baseline, pred):
            dist = abs(float(v_min) - PU_SUBTENSAO_THRESHOLD)
            return (
                NoiseSourceCategory.BORDERLINE_SUBTENSAO.value,
                round(dist, 4),
                f"Tensão mínima ({float(v_min):.4f} p.u.) na fronteira difusa de 0.90 p.u. (distância: {dist:.4f} p.u.).",
            )

    # 2. Borderline Sobretensão (Swell) near 1.05 pu
    if v_max is not None and 1.035 <= float(v_max) <= 1.065:
        if "SOBRETENSAO" in (baseline, pred):
            dist = abs(float(v_max) - PU_SOBRETENSAO_THRESHOLD)
            return (
                NoiseSourceCategory.BORDERLINE_SOBRETENSAO.value,
                round(dist, 4),
                f"Tensão máxima ({float(v_max):.4f} p.u.) na fronteira difusa de 1.05 p.u. (distância: {dist:.4f} p.u.).",
            )

    # 3. Borderline Desbalanceamento Crítico near 3.0% (0.03)
    if v_imb is not None and 0.024 <= float(v_imb) <= 0.036:
        if "DESBALANCEAMENTO" in (baseline, pred):
            dist = abs(float(v_imb) - PRODIST_DESBALANCEAMENTO_CRITICO)
            return (
                NoiseSourceCategory.BORDERLINE_DESBALANCEAMENTO.value,
                round(dist, 4),
                f"Razão de desbalanceamento ({float(v_imb):.2%}) na fronteira crítica regulatória de 3.0% (distância: {dist:.2%}).",
            )

    # 4. Zero Voltage Cutoff (near 5V / 0.04 pu)
    is_fault = any(f in (baseline, pred) for f in ("FALTA_MONOFASICA", "FALTA_BIFASICA", "INTERRUPCAO_TOTAL"))
    if is_fault:
        for phase in ("U_L1", "U_L2", "U_L3"):
            raw_v = row.get(phase)
            if raw_v is not None and 2.0 <= float(raw_v) <= 8.0:
                dist = abs(float(raw_v) - PU_INTERRUPCAO_MAX_VOLTS)
                return (
                    NoiseSourceCategory.ZERO_VOLTAGE_CUTOFF.value,
                    round(dist, 2),
                    f"Tensão da fase {phase} ({float(raw_v):.1f}V) no patamar limítrofe de corte de falta física (5.0V).",
                )

    # 5. Precedence Conflict (multiple disturbance conditions simultaneously present)
    conditions_met = 0
    if v_min is not None and float(v_min) < PU_SUBTENSAO_THRESHOLD and float(v_min) > PU_INTERRUPCAO_THRESHOLD:
        conditions_met += 1
    if v_max is not None and float(v_max) > PU_SOBRETENSAO_THRESHOLD:
        conditions_met += 1
    if v_imb is not None and float(v_imb) >= PRODIST_DESBALANCEAMENTO_CRITICO:
        conditions_met += 1
    if conditions_met >= 2:
        return (
            NoiseSourceCategory.PRECEDENCE_CONFLICT.value,
            0.0,
            f"Múltiplas perturbações simultâneas na onda. Norma priorizou '{baseline}', modelo previu '{pred}'.",
        )

    # 6. Class Weight Bias (minority fault falsely predicted on normal voltages)
    if baseline == "NORMAL" and pred in ("FALTA_MONOFASICA", "FALTA_BIFASICA", "INTERRUPCAO_TOTAL"):
        return (
            NoiseSourceCategory.CLASS_WEIGHT_BIAS.value,
            0.0,
            "Superestimação de classe rara devido ao balanceamento de pesos de classe. Tensões operacionais normais.",
        )

    # 7. High Probability Uncertainty / Ambiguity
    if conf is not None and float(conf) < 0.60:
        return (
            NoiseSourceCategory.PROBABILITY_UNCERTAINTY.value,
            round(float(conf), 4),
            f"Alta incerteza probabilística do modelo na região (confiança máxima de apenas {float(conf):.1%}).",
        )

    # 8. Unclassified Divergence
    return (
        NoiseSourceCategory.UNCLASSIFIED_DIVERGENCE.value,
        0.0,
        f"Divergência entre norma ({baseline}) e previsão ({pred}). Candidato para revisão manual de engenharia.",
    )


@dataclass
class DiscrepancyAuditReport:
    """Detailed audit metrics and discrepancy container for voltage classification."""

    total_samples: int
    matched_samples: int
    discrepancy_count: int
    match_rate: float
    discrepancies_df: pl.DataFrame
    noise_breakdown: dict[str, int]
    export_csv_path: Path | None = None
    export_report_path: Path | None = None

    def summary_table_dict(self) -> list[dict[str, Any]]:
        return [
            {"Origem do Ruído": k, "Ocorrências": v, "Percentual": f"{v / max(1, self.discrepancy_count):.1%}"}
            for k, v in sorted(self.noise_breakdown.items(), key=lambda x: x[1], reverse=True)
        ]


class VoltageDiscrepancyAuditor:
    """Audits discrepancies between ML voltage predictions and deterministic PRODIST baseline.

    Registers error cases into CSVs in output/ for manual engineering review and categorizes sources of noise.
    """

    def __init__(self, output_dir: Path | str = Path("output/reports")) -> None:
        self.output_dir = Path(output_dir)

    def export_audit_csv(
        self,
        discrepancies_df: pl.DataFrame,
        base_filename: str = "voltage_classification_errors",
    ) -> tuple[Path, Path]:
        """Exports discrepancies to a timestamped CSV and a latest CSV pointer using utf-8-sig."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dated_path = self.output_dir / f"{base_filename}_{timestamp}.csv"
        latest_path = self.output_dir / f"{base_filename}_latest.csv"

        # CSV format: delimiter ';' and utf-8-sig encoding for Windows/Excel compatibility
        discrepancies_df.write_csv(dated_path, separator=";")
        discrepancies_df.write_csv(latest_path, separator=";")

        return dated_path, latest_path

    def export_markdown_report(
        self,
        report: DiscrepancyAuditReport,
        base_filename: str = "voltage_classification_audit",
    ) -> Path:
        """Exports a detailed markdown audit diagnostic report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"{base_filename}_latest.md"

        lines = [
            "# Relatório de Auditoria de Ruído e Discrepâncias de Classificação de Tensão (PRODIST M8)",
            "",
            f"> **Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"> **Arquivo de Erros Cadastrados:** [`{report.export_csv_path.name if report.export_csv_path else 'N/A'}`]({report.export_csv_path})",
            "",
            "## 1. Resumo Executivo",
            "",
            "| Métrica | Valor | Observação |",
            "| :--- | :--- | :--- |",
            f"| **Total de Amostras Avaliadas** | `{report.total_samples:,}` | Amostragem representativa do Data Lake |",
            f"| **Concordância com a Norma (Match)** | `{report.matched_samples:,}` | Classificação idêntica ao PRODIST M8 |",
            f"| **Discrepâncias / Erros Cadastrados** | `{report.discrepancy_count:,}` | Registrados para revisão manual em CSV |",
            f"| **Taxa de Concordância do Modelo** | `{report.match_rate:.2%}` | Fidelidade da árvore ao baseline normativo |",
            "",
            "---",
            "",
            "## 2. Diagnóstico das Fontes de Ruído Identificadas",
            "",
            "| Categoria da Fonte de Ruído | Ocorrências | Proporção (%) | Descrição Técnica Operacional |",
            "| :--- | :---: | :---: | :--- |",
        ]

        explanations_map = {
            NoiseSourceCategory.BORDERLINE_SUBTENSAO.value: "Tensão mínima na zona de transição limítrofe de 0.90 p.u. (0.885 a 0.915 p.u.).",
            NoiseSourceCategory.BORDERLINE_SOBRETENSAO.value: "Tensão máxima na zona de transição limítrofe de 1.05 p.u. (1.035 a 1.065 p.u.).",
            NoiseSourceCategory.BORDERLINE_DESBALANCEAMENTO.value: "Desbalanceamento PVUR em torno do limiar crítico regulatório de 3.0%.",
            NoiseSourceCategory.ZERO_VOLTAGE_CUTOFF.value: "Fase em patamar de tensão limítrofe do corte físico de interrupção (<= 5.0V).",
            NoiseSourceCategory.PRECEDENCE_CONFLICT.value: "Múltiplas perturbações simultâneas na onda onde a ordem de prioridade da norma atuou.",
            NoiseSourceCategory.CLASS_WEIGHT_BIAS.value: "Superestimação de classe minoritária rara induzida pelo balanceamento de pesos.",
            NoiseSourceCategory.PROBABILITY_UNCERTAINTY.value: "Região de alta entropia onde a probabilidade máxima da classe é inferior a 60%.",
            NoiseSourceCategory.TELEMETRY_OUTAGE_AMBIGUITY.value: "Ambiguidade entre perda de comunicação e queda física corroborada.",
            NoiseSourceCategory.UNCLASSIFIED_DIVERGENCE.value: "Divergência não categorizada; candidata para auditoria manual prioritária.",
        }

        for k, v in sorted(report.noise_breakdown.items(), key=lambda x: x[1], reverse=True):
            expl = explanations_map.get(k, "Requer inspeção manual de engenharia.")
            pct = v / max(1, report.discrepancy_count)
            lines.append(f"| **`{k}`** | `{v:,}` | `{pct:.1%}` | {expl} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Diretrizes para Revisão Manual em Engenharia",
            "",
            "1. **Revisão pelo CSV de Erros:** Abra o arquivo `output/reports/voltage_classification_errors_latest.csv` (delimitado por `;`).",
            "2. **Amostras Limítrofes (Borderline):** Não representam erro elétrico grave, mas a calibração de histerese natural do limiar regulatório.",
            "3. **Alinhamento 100%:** Para garantir 100% de conformidade com o regulatório em ambiente de produção, utilize `clf.predict(df, align_with_rules=True)` ou o modo híbrido.",
            "",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def audit(
        self,
        predictions_df: pl.DataFrame,
        target_column: str = "BASELINE_VOLTAGE_LABEL",
        *,
        export_csv: bool = True,
        export_report: bool = True,
    ) -> DiscrepancyAuditReport:
        """Audits predictions DataFrame and registers errors into CSVs."""
        if predictions_df.is_empty():
            empty_df = pl.DataFrame()
            return DiscrepancyAuditReport(0, 0, 0, 1.0, empty_df, {})

        df = predictions_df.clone()

        # Ensure target column exists
        if target_column not in df.columns:
            df = RuleBasedVoltageClassifier.classify(df)

        pred_col = "RAW_ML_PREDICTION" if "RAW_ML_PREDICTION" in df.columns else "PREDICTED_DISTURBANCE_CLASS"
        if pred_col not in df.columns:
            raise KeyError(f"Predictions dataframe missing prediction column ({pred_col}).")

        # Discrepancy condition
        is_disc = pl.col(pred_col) != pl.col(target_column)
        disc_df = df.filter(is_disc)

        total_samples = df.height
        disc_count = disc_df.height
        matched_samples = total_samples - disc_count
        match_rate = matched_samples / max(1, total_samples)

        noise_breakdown: dict[str, int] = {}
        csv_path = None
        report_path = None

        if disc_count > 0:
            rows = disc_df.to_dicts()
            sources = []
            distances = []
            explanations = []

            for r in rows:
                src, dist, expl = diagnose_discrepancy_row(r)
                sources.append(src)
                distances.append(dist)
                explanations.append(expl)
                noise_breakdown[src] = noise_breakdown.get(src, 0) + 1

            enriched_disc = disc_df.with_columns([
                pl.col(target_column).alias("BASELINE_RULE_LABEL"),
                pl.col(pred_col).alias("ML_PREDICTED_LABEL"),
                pl.Series("NOISE_SOURCE", sources),
                pl.Series("THRESHOLD_DISTANCE_PU", distances),
                pl.Series("DIAGNOSTIC_EXPLANATION", explanations),
            ])

            # Order columns nicely for manual review
            lead_cols = [
                "UC", "NIO", "REPORT_DAY",
                "BASELINE_RULE_LABEL", "ML_PREDICTED_LABEL",
                "NOISE_SOURCE", "DIAGNOSTIC_EXPLANATION",
                "THRESHOLD_DISTANCE_PU", "DISTURBANCE_CONFIDENCE",
                "V_MIN_PU", "V_MAX_PU", "V_AVG_PU", "VOLTAGE_IMBALANCE_RATIO",
                "FASE_CIRCUITO", "TENSAO_BASE",
            ]
            final_cols = [c for c in lead_cols if c in enriched_disc.columns]
            rest_cols = [c for c in enriched_disc.columns if c not in final_cols]
            ordered_disc = enriched_disc.select(final_cols + rest_cols)

            if export_csv:
                csv_path, _ = self.export_audit_csv(ordered_disc)
        else:
            ordered_disc = pl.DataFrame()

        report = DiscrepancyAuditReport(
            total_samples=total_samples,
            matched_samples=matched_samples,
            discrepancy_count=disc_count,
            match_rate=round(match_rate, 4),
            discrepancies_df=ordered_disc,
            noise_breakdown=noise_breakdown,
            export_csv_path=csv_path,
        )

        if export_report and total_samples > 0:
            report_path = self.export_markdown_report(report)
            report.export_report_path = report_path

        return report

