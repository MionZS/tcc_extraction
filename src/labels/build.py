"""Label construction from data sources.

Builds label DataFrames from various sources (MDM, CIS, field inspection)
following the contracts defined in docs/dataset_architecture_decision.md §5.6.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import polars as pl

from src.labels.contracts import LABEL_TYPES, LABEL_SOURCES


def build_labels(
    entity_type: str,
    entity_ids: list[str],
    *,
    reference_start: date,
    reference_end: date,
    prediction_horizon: int,
    label_type: str,
    label_value: bool,
    label_source: str,
    label_version: str = "v1",
    confidence: float = 1.0,
    reviewer_id: str = "",
    notes: str = "",
) -> pl.DataFrame:
    """Build a labels DataFrame for a batch of entities.

    Args:
        entity_type: Type of entity (NIO, UC, ALIMENTADOR, etc.).
        entity_ids: List of entity identifiers.
        reference_start: Start of the reference window.
        reference_end: End of the reference window.
        prediction_horizon: Days ahead for prediction.
        label_type: Type of label (e.g. "falha_comunicacao").
        label_value: True if the event occurred, False otherwise.
        label_source: Source of the label (e.g. "regra", "campo").
        label_version: Version tag for reproducibility.
        confidence: Confidence level (0.0 to 1.0).
        reviewer_id: Identifier of the reviewer (if manual).
        notes: Free-text notes.

    Returns:
        DataFrame with one row per entity.
    """
    now = datetime.now(timezone.utc)

    rows: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        rows.append({
            "ENTITY_TYPE": entity_type,
            "ENTITY_ID": entity_id,
            "REFERENCE_START": reference_start,
            "REFERENCE_END": reference_end,
            "PREDICTION_HORIZON": prediction_horizon,
            "LABEL_TYPE": label_type,
            "LABEL_VALUE": label_value,
            "LABEL_SOURCE": label_source,
            "CONFIDENCE": confidence,
            "REVIEWER_ID": reviewer_id,
            "CREATED_AT": now,
            "LABEL_VERSION": label_version,
            "NOTES": notes,
        })

    return pl.DataFrame(rows, strict=False)


def build_communication_failure_labels(
    nios: list[str],
    *,
    reference_day: date,
    lookahead_days: int = 2,
    label_version: str = "v1",
    label_source: str = "regra",
    confidence: float = 0.8,
) -> pl.DataFrame:
    """Build communication failure labels using a rule-based approach.

    A NIO is labeled as ``True`` (communication failure) if it has
    no MDM data for `lookahead_days` after the reference day.

    This function builds the label structure — the actual data check
    should be done by the caller passing the appropriate MDM coverage info.

    Args:
        nios: List of NIOs to label.
        reference_day: The last day of the feature window.
        lookahead_days: Number of days to look ahead for failure detection.
        label_version: Version tag.
        label_source: Source identifier.
        confidence: Confidence level.

    Returns:
        DataFrame with communication failure labels.
    """
    reference_start = reference_day - timedelta(days=6)  # 7-day feature window
    reference_end = reference_day

    return build_labels(
        entity_type="NIO",
        entity_ids=nios,
        reference_start=reference_start,
        reference_end=reference_end,
        prediction_horizon=lookahead_days,
        label_type="falha_comunicacao",
        label_value=False,  # placeholder — caller should set True/False based on data
        label_source=label_source,
        label_version=label_version,
        confidence=confidence,
        notes=f"Rule-based: check MDM presence in {lookahead_days} days after {reference_day}",
    )


def apply_communication_failure_rule(
    labels_df: pl.DataFrame,
    nios_with_data: set[str],
    nios_without_data: set[str],
) -> pl.DataFrame:
    """Apply the communication failure rule to a labels DataFrame.

    Sets LABEL_VALUE to True for NIOs that have no MDM data
    in the lookahead window.

    Args:
        labels_df: Labels DataFrame from ``build_communication_failure_labels``.
        nios_with_data: Set of NIOs that have MDM data in the lookahead window.
        nios_without_data: Set of NIOs without MDM data in the lookahead window.

    Returns:
        Updated labels DataFrame.
    """
    def _check_failure(nio: str) -> bool:
        return nio in nios_without_data

    return labels_df.with_columns(
        pl.col("ENTITY_ID")
        .map_elements(_check_failure, return_dtype=pl.Boolean)
        .alias("LABEL_VALUE")
    )
