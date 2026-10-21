"""Event extraction and processing layer (alarms, operational events)."""

from __future__ import annotations

from src.events.alarm_extractor import AlarmExtractor
from src.events.alarm_categories import (
    COMMUNICATION_ALARM_KEYWORDS,
    ELECTRICAL_ALARM_KEYWORDS,
    TAMPER_ALARM_KEYWORDS,
    categorize_alarm_events_df,
    classify_alarm_family,
    compute_alarm_family_metrics,
)

__all__ = [
    "AlarmExtractor",
    "classify_alarm_family",
    "categorize_alarm_events_df",
    "compute_alarm_family_metrics",
    "COMMUNICATION_ALARM_KEYWORDS",
    "ELECTRICAL_ALARM_KEYWORDS",
    "TAMPER_ALARM_KEYWORDS",
]
