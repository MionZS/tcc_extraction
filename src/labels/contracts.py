"""Label contracts and validation rules.

Defines the allowed values, types, and validation rules for labels,
following the architecture decision §5.6.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl


# ── Allowed values ───────────────────────────────────────────────────────────

LABEL_TYPES: dict[str, dict[str, Any]] = {
    "falha_comunicacao": {
        "description": "Communication failure — meter stopped sending data",
        "entity_types": ["NIO", "UC", "ALIMENTADOR"],
        "typical_horizon_days": [1, 2, 3, 7],
    },
    "anomalia_eletrica": {
        "description": "Electrical anomaly — unusual voltage, current, or energy pattern",
        "entity_types": ["NIO", "UC"],
        "typical_horizon_days": [1, 7],
    },
    "classe_perfil": {
        "description": "Consumer profile classification — residential, commercial, etc.",
        "entity_types": ["NIO", "UC"],
        "typical_horizon_days": [0],  # contemporaneous
    },
    "troca_medidor": {
        "description": "Meter replacement event",
        "entity_types": ["NIO"],
        "typical_horizon_days": [7, 30],
    },
    "reset_registro": {
        "description": "Register reset or rollover detected",
        "entity_types": ["NIO"],
        "typical_horizon_days": [1],
    },
}

LABEL_SOURCES: dict[str, str] = {
    "campo": "Field inspection or manual measurement",
    "alarme": "Automated alarm from SCADA/MDM system",
    "os": "Work order (ordem de serviço)",
    "regra": "Automated rule-based derivation",
    "especialista": "Expert human review",
    "mdm": "MDM system event log",
    "cis": "CIS system event or status change",
}

ENTITY_TYPES: list[str] = ["NIO", "UC", "ALIMENTADOR", "SUBESTACAO"]


# ── Validation ───────────────────────────────────────────────────────────────

class LabelValidationError(ValueError):
    """Raised when a label fails validation."""
    pass


def validate_label(row: dict[str, Any]) -> list[str]:
    """Validate a label row against contracts.

    Returns a list of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # ENTITY_TYPE
    entity_type = row.get("ENTITY_TYPE")
    if entity_type not in ENTITY_TYPES:
        errors.append(f"Invalid ENTITY_TYPE: {entity_type!r}. Must be one of {ENTITY_TYPES}")

    # ENTITY_ID
    entity_id = row.get("ENTITY_ID")
    if not entity_id or not str(entity_id).strip():
        errors.append("ENTITY_ID is required and must be non-empty")

    # REFERENCE_START / REFERENCE_END
    ref_start = row.get("REFERENCE_START")
    ref_end = row.get("REFERENCE_END")
    if ref_start is None:
        errors.append("REFERENCE_START is required")
    if ref_end is None:
        errors.append("REFERENCE_END is required")
    if ref_start is not None and ref_end is not None:
        if ref_start > ref_end:
            errors.append(
                f"REFERENCE_START ({ref_start}) must be <= REFERENCE_END ({ref_end})"
            )

    # PREDICTION_HORIZON
    horizon = row.get("PREDICTION_HORIZON")
    if horizon is not None and horizon < 0:
        errors.append(f"PREDICTION_HORIZON must be >= 0, got {horizon}")

    # LABEL_TYPE
    label_type = row.get("LABEL_TYPE")
    if label_type not in LABEL_TYPES:
        errors.append(
            f"Invalid LABEL_TYPE: {label_type!r}. Must be one of {list(LABEL_TYPES)}"
        )

    # LABEL_VALUE
    label_value = row.get("LABEL_VALUE")
    if label_value is None:
        errors.append("LABEL_VALUE is required")

    # LABEL_SOURCE
    label_source = row.get("LABEL_SOURCE")
    if label_source not in LABEL_SOURCES:
        errors.append(
            f"Invalid LABEL_SOURCE: {label_source!r}. Must be one of {list(LABEL_SOURCES)}"
        )

    # CONFIDENCE
    confidence = row.get("CONFIDENCE")
    if confidence is not None:
        if not (0.0 <= confidence <= 1.0):
            errors.append(f"CONFIDENCE must be in [0.0, 1.0], got {confidence}")

    # LABEL_VERSION
    label_version = row.get("LABEL_VERSION")
    if not label_version or not str(label_version).strip():
        errors.append("LABEL_VERSION is required and must be non-empty")

    # CREATED_AT
    created_at = row.get("CREATED_AT")
    if created_at is None:
        errors.append("CREATED_AT is required")

    return errors


def validate_labels_batch(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Validate all rows in a labels DataFrame.

    Returns a list of dicts with ``row_index`` and ``errors`` for
    each invalid row.
    """
    issues: list[dict[str, Any]] = []

    for i, row_dict in enumerate(df.iter_rows(named=True)):
        errors = validate_label(row_dict)
        if errors:
            issues.append({"row_index": i, "errors": errors})

    return issues
