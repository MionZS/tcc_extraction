"""Unit tests for multi-table AlarmExtractor."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from src.datasets.schemas import ALARM_EVENTS_SCHEMA
from src.events.alarm_extractor import AlarmExtractor


@pytest.fixture
def sqlite_alarm_engine():
    """Create in-memory SQLite engine with simulated daily T_EVENT tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        # Table for 2026-06-20
        conn.execute(text("""
            CREATE TABLE "AMI.T_EVENT_20260620" (
                EVENT_ID TEXT,
                OBJ_ID TEXT,
                ORIGIN_TIME TIMESTAMP,
                RECEIVE_TIME TIMESTAMP,
                CONTENT TEXT,
                SYSTEM_CODE TEXT,
                OBJECT_TYPE TEXT,
                OBJECT_ORG TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO "AMI.T_EVENT_20260620" VALUES
            ('EVT_001', '41000001', '2026-06-20 08:00:00', '2026-06-20 08:00:15', 'Falta de Energia', 'SYS1', 'METER', 'COPEL'),
            ('EVT_002', '41000002', '2026-06-20 09:30:00', '2026-06-20 09:30:10', 'Queda de Tensao', 'SYS1', 'METER', 'COPEL')
        """))

        # Table for 2026-06-21
        conn.execute(text("""
            CREATE TABLE "AMI.T_EVENT_20260621" (
                EVENT_ID TEXT,
                OBJ_ID TEXT,
                ORIGIN_TIME TIMESTAMP,
                RECEIVE_TIME TIMESTAMP,
                CONTENT TEXT,
                SYSTEM_CODE TEXT,
                OBJECT_TYPE TEXT,
                OBJECT_ORG TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO "AMI.T_EVENT_20260621" VALUES
            ('EVT_003', '41000001', '2026-06-21 14:00:00', '2026-06-21 14:00:45', 'Restabelecimento', 'SYS1', 'METER', 'COPEL')
        """))

        # Note: 2026-06-22 is intentionally missing (simulating missing day / ORA-00942)
        conn.commit()

    return engine


def test_alarm_extractor_single_day(sqlite_alarm_engine):
    extractor = AlarmExtractor(engine=sqlite_alarm_engine)
    df = extractor.extract_day(report_day=date(2026, 6, 20), nios=["41000001"])

    assert df.height == 1
    assert df["ALARM_ID"][0] == "EVT_001"
    assert df["NIO"][0] == "41000001"
    assert df["CONTENT"][0] == "Falta de Energia"
    assert df["LATENCY_SECONDS"][0] == 15.0


def test_alarm_extractor_missing_day_resilience(sqlite_alarm_engine):
    extractor = AlarmExtractor(engine=sqlite_alarm_engine)
    # Day 2026-06-22 table does not exist
    df = extractor.extract_day(report_day=date(2026, 6, 22), nios=["41000001"])

    assert df.is_empty()
    assert set(df.columns) == set(ALARM_EVENTS_SCHEMA.keys())


def test_alarm_extractor_window(sqlite_alarm_engine, tmp_path):
    extractor = AlarmExtractor(engine=sqlite_alarm_engine)
    # Window of 3 days ending on 2026-06-22 (covers 06-19, 06-20, 06-21, 06-22)
    df_window = extractor.extract_window(
        cutoff_date=date(2026, 6, 22),
        window_days=3,
        nios=["41000001", "41000002"],
        save_partitions=True,
        base_dir=tmp_path,
    )

    # 2 events from 06-20 + 1 event from 06-21 = 3 events
    assert df_window.height == 3
    assert set(df_window["NIO"].to_list()) == {"41000001", "41000002"}

    # Verify saved partitions in events/alarm_events/
    part_20 = tmp_path / "events" / "alarm_events" / "report_year=2026" / "report_month=06" / "report_day=20" / "data.parquet"
    part_21 = tmp_path / "events" / "alarm_events" / "report_year=2026" / "report_month=06" / "report_day=21" / "data.parquet"
    assert part_20.exists()
    assert part_21.exists()

    df_read_20 = pl.read_parquet(part_20)
    assert df_read_20.height == 2
