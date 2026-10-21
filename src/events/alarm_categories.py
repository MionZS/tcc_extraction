"""Classification and categorization of AMI Smart Meter Alarm Events.

Partitions alarm codes/descriptions into three mutually exclusive operational families:
1. COMMUNICATION_ALARMS: Telemetry drops, offline/online states, GPRS/mesh link quality, handshake failures.
2. ELECTRICAL_ALARMS: Grid physics anomalies, voltage sag/swell, current unbalance, phase failure, reverse power.
3. TAMPER_ALARMS: Physical meter intrusion, terminal/meter cover opening, magnetic interference, bypass.
"""

from __future__ import annotations

from typing import Any, Sequence
import polars as pl

# ── 1. Canonical Alarm Family Sets ──────────────────────────────────────────

COMMUNICATION_ALARM_KEYWORDS = {
    "OFFLINE",
    "ONLINE",
    "COMMUNICATION",
    "COMMUNICATION ABNORMAL",
    "COMMUNICATION RESTORED",
    "IOMMSOFFLINE",
    "IOMMSONLINE",
    "POORLINKQUALITY",
    "POOR LINK QUALITY",
    "AKEKERROR",
    "AKEK ERROR",
    "TIMEOUT",
    "RETRY EXCEEDED",
    "MODEM FAILURE",
    "SIM ERROR",
    "CARRIER DROP",
}

ELECTRICAL_ALARM_KEYWORDS = {
    "CURRENT UNBALANCE",
    "UNDER VOLTAGE",
    "OVER VOLTAGE",
    "MISSING CURRENT",
    "MISSING VOLTAGE",
    "PHASE FAILURE",
    "REVERSE CURRENT",
    "REVERSE POWER",
    "FREQUENCY ABNORMAL",
    "OVER CURRENT",
    "POWER FACTOR LOW",
    "SAG",
    "SWELL",
    "HARMONIC DISTORTION",
}

TAMPER_ALARM_KEYWORDS = {
    "STRONG MAGNETIC FIELD",
    "MAGNETIC FIELD",
    "MAGNETIC",
    "TERMINAL COVER OPEN",
    "TERMINAL COVER OPENED",
    "METER COVER OPEN",
    "METER COVER OPENED",
    "METER BOX OPENED",
    "BYPASS",
    "BYPASS DETECTED",
    "VOLTAGE AT LOAD SIDE WHILE DISCONNECTED",
    "REVERSE RUNNING",
    "EARTH FAULT / TAMPER",
    "CASE OPEN",
    "HARDWARE TAMPER",
}


def classify_alarm_family(event_desc: str | None) -> str:
    """Classify an alarm description into 'COMMUNICATION', 'ELECTRICAL', 'TAMPER', or 'OTHER'.

    Args:
        event_desc: Raw alarm description or code from AMI.T_EVENT table.

    Returns:
        One of 'COMMUNICATION', 'ELECTRICAL', 'TAMPER', 'OTHER'.
    """
    if not event_desc:
        return "OTHER"

    desc_norm = str(event_desc).strip().upper()

    # 1. Tamper alarms take priority if keywords match (tamper is safety/fraud critical)
    for kw in TAMPER_ALARM_KEYWORDS:
        if kw in desc_norm:
            return "TAMPER"

    # 2. Electrical alarms
    for kw in ELECTRICAL_ALARM_KEYWORDS:
        if kw in desc_norm:
            return "ELECTRICAL"

    # 3. Communication alarms
    for kw in COMMUNICATION_ALARM_KEYWORDS:
        if kw in desc_norm:
            return "COMMUNICATION"

    return "OTHER"


def categorize_alarm_events_df(alarm_events_df: pl.DataFrame) -> pl.DataFrame:
    """Enrich an alarm_events DataFrame with the ALARM_FAMILY column.

    Args:
        alarm_events_df: Normalized alarm events table (must have EVENT_DESCRIPTION or EVENT_CODE).

    Returns:
        DataFrame with an added 'ALARM_FAMILY' column.
    """
    if alarm_events_df.is_empty():
        return alarm_events_df.with_columns(pl.lit(None, dtype=pl.String).alias("ALARM_FAMILY"))

    col_target = "EVENT_DESCRIPTION" if "EVENT_DESCRIPTION" in alarm_events_df.columns else "EVENT_CODE"
    if col_target not in alarm_events_df.columns:
        return alarm_events_df.with_columns(pl.lit("OTHER").alias("ALARM_FAMILY"))

    # Vectorized classification using regex patterns for high performance
    comm_pattern = "(?i)(" + "|".join(COMMUNICATION_ALARM_KEYWORDS) + ")"
    elec_pattern = "(?i)(" + "|".join(ELECTRICAL_ALARM_KEYWORDS) + ")"
    tamper_pattern = "(?i)(" + "|".join(TAMPER_ALARM_KEYWORDS) + ")"

    col_expr = pl.col(col_target).cast(pl.String).fill_null("")

    return alarm_events_df.with_columns(
        pl.when(col_expr.str.contains(tamper_pattern))
        .then(pl.lit("TAMPER"))
        .when(col_expr.str.contains(elec_pattern))
        .then(pl.lit("ELECTRICAL"))
        .when(col_expr.str.contains(comm_pattern))
        .then(pl.lit("COMMUNICATION"))
        .otherwise(pl.lit("OTHER"))
        .alias("ALARM_FAMILY")
    )


def compute_alarm_family_metrics(
    alarm_events_df: pl.DataFrame,
) -> dict[str, Any]:
    """Compute aggregated summary metrics partitioned by alarm family.

    Returns dict with keys:
    - comm_alarm_count: int
    - elec_alarm_count: int
    - tamper_alarm_count: int
    - comm_latency_mean: float
    - comm_latency_p95: float
    - elec_latency_mean: float
    """
    if alarm_events_df.is_empty():
        return {
            "comm_alarm_count": 0,
            "elec_alarm_count": 0,
            "tamper_alarm_count": 0,
            "comm_latency_mean": 0.0,
            "comm_latency_p95": 0.0,
            "elec_latency_mean": 0.0,
        }

    df_cat = (
        alarm_events_df
        if "ALARM_FAMILY" in alarm_events_df.columns
        else categorize_alarm_events_df(alarm_events_df)
    )

    comm_df = df_cat.filter(pl.col("ALARM_FAMILY") == "COMMUNICATION")
    elec_df = df_cat.filter(pl.col("ALARM_FAMILY") == "ELECTRICAL")
    tamper_df = df_cat.filter(pl.col("ALARM_FAMILY") == "TAMPER")

    def _calc_lat(sub_df: pl.DataFrame) -> tuple[float, float]:
        if sub_df.is_empty() or "LATENCY_SECONDS" not in sub_df.columns:
            return 0.0, 0.0
        vals = sub_df["LATENCY_SECONDS"].drop_nulls()
        if vals.len() == 0:
            return 0.0, 0.0
        mean_v = float(vals.mean())
        p95_v = float(vals.quantile(0.95))
        return mean_v, p95_v

    comm_mean, comm_p95 = _calc_lat(comm_df)
    elec_mean, _ = _calc_lat(elec_df)

    return {
        "comm_alarm_count": comm_df.height,
        "elec_alarm_count": elec_df.height,
        "tamper_alarm_count": tamper_df.height,
        "comm_latency_mean": round(comm_mean, 2),
        "comm_latency_p95": round(comm_p95, 2),
        "elec_latency_mean": round(elec_mean, 2),
    }
