"""Labels module — independent label construction for TCC research.

Labels are stored separately from features and support multiple
research questions (§5.6 of the architecture decision).
"""

from src.labels.contracts import (
    LABEL_TYPES,
    LABEL_SOURCES,
    ENTITY_TYPES,
    validate_label,
)
from src.labels.build import build_labels, build_communication_failure_labels
