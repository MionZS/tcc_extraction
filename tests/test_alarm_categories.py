"""Tests for Alarm Event Categorization (Communication vs. Electrical vs. Tamper)."""

from __future__ import annotations

import polars as pl
import pytest

from src.events.alarm_categories import (
    categorize_alarm_events_df,
    classify_alarm_family,
    compute_alarm_family_metrics,
)


def test_classify_alarm_family_keywords():
    # Communication
    assert classify_alarm_family("iommsOffline") == "COMMUNICATION"
    assert classify_alarm_family("Communication Abnormal") == "COMMUNICATION"
    assert classify_alarm_family("poorLinkQuality") == "COMMUNICATION"
    assert classify_alarm_family("Communication Restored") == "COMMUNICATION"

    # Electrical
    assert classify_alarm_family("Current Unbalance") == "ELECTRICAL"
    assert classify_alarm_family("Under Voltage") == "ELECTRICAL"
    assert classify_alarm_family("Over Voltage") == "ELECTRICAL"
    assert classify_alarm_family("Phase Failure") == "ELECTRICAL"
    assert classify_alarm_family("Reverse Current") == "ELECTRICAL"

    # Tamper
    assert classify_alarm_family("Strong Magnetic Field") == "TAMPER"
    assert classify_alarm_family("Terminal Cover Open") == "TAMPER"
    assert classify_alarm_family("Meter Cover Open") == "TAMPER"
    assert classify_alarm_family("Bypass Detected") == "TAMPER"
    assert classify_alarm_family("Voltage at load side while disconnected") == "TAMPER"

    # Other / None
    assert classify_alarm_family(None) == "OTHER"
    assert classify_alarm_family("Firmware updated successfully") == "OTHER"


def test_categorize_alarm_events_df():
    df = pl.DataFrame({
        "NIO": ["41001", "41001", "41002", "41003"],
        "EVENT_DESCRIPTION": [
            "iommsOffline",
            "Current Unbalance",
            "Strong Magnetic Field",
            "Routine Clock Sync",
        ],
        "LATENCY_SECONDS": [12.0, 5.0, 2.0, 1.0],
    })

    categorized = categorize_alarm_events_df(df)
    assert "ALARM_FAMILY" in categorized.columns
    families = categorized["ALARM_FAMILY"].to_list()
    assert families == ["COMMUNICATION", "ELECTRICAL", "TAMPER", "OTHER"]


def test_compute_alarm_family_metrics():
    df = pl.DataFrame({
        "EVENT_DESCRIPTION": [
            "iommsOffline",
            "Communication Abnormal",
            "Current Unbalance",
            "Terminal Cover Open",
            "Terminal Cover Open",
        ],
        "LATENCY_SECONDS": [10.0, 20.0, 4.0, 2.0, 2.0],
    })

    metrics = compute_alarm_family_metrics(df)
    assert metrics["comm_alarm_count"] == 2
    assert metrics["elec_alarm_count"] == 1
    assert metrics["tamper_alarm_count"] == 2
    assert metrics["comm_latency_mean"] == 15.0
    assert metrics["elec_latency_mean"] == 4.0
