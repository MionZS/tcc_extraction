"""Multi-table Alarm Event Extractor for AMI Smart Meters.

Extracts complete alarm events across daily partitioned tables:
    AMI.T_EVENT_YYYYMMDD
for all considered days in an analytical window (e.g., 30 days) or a daily pipeline run.
Filters by population NIOs using Oracle SYS.ODCIVARCHAR2LIST with batching,
handles missing daily tables (ORA-00942) gracefully, and normalizes into the
canonical alarm_events semantic layer.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.datasets.normalize import normalize_alarm_events
from src.datasets.schemas import ALARM_EVENTS_SCHEMA, build_empty_frame
from src.datasets.writers import write_events_table
from src.db import create_engine


def _date_token(d: date) -> str:
    """Format date as YYYYMMDD string for table names."""
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


def _chunked(items: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    """Yield successive chunks from items."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _build_oracle_string_list(connection: Connection, values: Sequence[str]) -> Any:
    """Build SYS.ODCIVARCHAR2LIST object for Oracle binding."""
    driver_conn = getattr(connection.connection, "driver_connection", None)
    if driver_conn is not None and hasattr(driver_conn, "gettype"):
        obj_type = driver_conn.gettype("SYS.ODCIVARCHAR2LIST")
        return obj_type.newobject(list(values))
    # Fallback for mock/test connections
    return list(values)


class AlarmExtractor:
    """Extracts alarm events across multiple daily T_EVENT_YYYYMMDD tables."""

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        db_target: str = "orca",
        batch_size: int = 500,
        fetch_size: int = 1000,
    ):
        self._engine = engine
        self._db_target = db_target
        self.batch_size = batch_size
        self.fetch_size = fetch_size

    def _get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine
        return create_engine(self._db_target)

    def extract_day(
        self,
        report_day: date,
        *,
        nios: Sequence[str] | None = None,
        connection: Connection | None = None,
        run_id: str = "",
    ) -> pl.DataFrame:
        """Extract alarm events from AMI.T_EVENT_YYYYMMDD for a single day.

        Args:
            report_day: The target day to query.
            nios: Optional list of NIOs to filter. If None or empty, all alarms are fetched.
            connection: Optional open connection to reuse across calls.
            run_id: Execution run ID for metadata tracking.

        Returns:
            Normalized Polars DataFrame matching ALARM_EVENTS_SCHEMA.
        """
        table_name = f"AMI.T_EVENT_{_date_token(report_day)}"

        def _query_with_conn(conn: Connection) -> pl.DataFrame:
            # Check if dialect is Oracle to decide binding strategy and identifier quoting
            is_oracle = "oracle" in conn.dialect.name.lower()
            table_ref = table_name if is_oracle else f'"{table_name}"'

            chunks: list[pl.DataFrame] = []

            if not nios:
                # Query entire table for the day
                sql = f"SELECT * FROM {table_ref}"
                try:
                    res = conn.execute(text(sql))
                    rows = res.fetchall()
                    if rows:
                        cols = list(res.keys())
                        raw_df = pl.DataFrame(
                            [dict(zip(cols, r)) for r in rows],
                            infer_schema_length=None,
                            strict=False,
                        )
                        chunks.append(raw_df)
                except Exception as exc:
                    if "ORA-00942" in str(exc) or "no such table" in str(exc).lower():
                        return build_empty_frame("alarm_events")
                    raise
            else:
                clean_nios = [str(n).strip().lstrip("0") for n in nios if str(n).strip()]
                if not clean_nios:
                    return build_empty_frame("alarm_events")

                for nio_chunk in _chunked(clean_nios, self.batch_size):
                    if is_oracle:
                        oracle_list = _build_oracle_string_list(conn, nio_chunk)
                        sql = f"""
                            SELECT *
                            FROM {table_ref}
                            WHERE NULLIF(LTRIM(TRIM(OBJ_ID), '0'), '') IN (
                                SELECT COLUMN_VALUE FROM TABLE(CAST(:NIOS AS SYS.ODCIVARCHAR2LIST))
                            )
                        """
                        params: dict[str, Any] = {"NIOS": oracle_list}
                    else:
                        # Fallback for SQLite / mock DBs in tests
                        placeholders = ", ".join(f"'{n}'" for n in nio_chunk)
                        sql = f"""
                            SELECT *
                            FROM {table_ref}
                            WHERE OBJ_ID IN ({placeholders})
                        """
                        params = {}

                    try:
                        res = conn.execute(text(sql), params)
                        rows = res.fetchall()
                        if rows:
                            cols = list(res.keys())
                            raw_df = pl.DataFrame(
                                [dict(zip(cols, r)) for r in rows],
                                infer_schema_length=None,
                                strict=False,
                            )
                            chunks.append(raw_df)
                    except Exception as exc:
                        if "ORA-00942" in str(exc) or "no such table" in str(exc).lower():
                            return build_empty_frame("alarm_events")
                        raise

            if not chunks:
                return build_empty_frame("alarm_events")

            combined_raw = pl.concat(chunks, how="vertical_relaxed")
            return normalize_alarm_events(combined_raw, run_id=run_id)

        if connection is not None:
            return _query_with_conn(connection)

        engine = self._get_engine()
        owned_engine = self._engine is None
        try:
            with engine.connect() as conn:
                return _query_with_conn(conn)
        finally:
            if owned_engine:
                engine.dispose()

    def extract_window(
        self,
        cutoff_date: date,
        *,
        window_days: int = 30,
        nios: Sequence[str] | None = None,
        run_id: str = "",
        save_partitions: bool = True,
        base_dir: Path | str = "output",
    ) -> pl.DataFrame:
        """Extract and aggregate alarm events across all days in [cutoff_date - window_days, cutoff_date].

        Args:
            cutoff_date: Reference end date for the window.
            window_days: Number of days to look back (inclusive).
            nios: Target NIOs to filter.
            run_id: Run ID for metadata.
            save_partitions: If True, saves each available day to output/events/alarm_events/.
            base_dir: Base directory for partition storage.

        Returns:
            Concatenated Polars DataFrame of all alarms in the window.
        """
        all_dates = [cutoff_date - timedelta(days=i) for i in range(window_days, -1, -1)]
        engine = self._get_engine()
        owned_engine = self._engine is None

        daily_dfs: list[pl.DataFrame] = []

        try:
            with engine.connect() as conn:
                for target_day in all_dates:
                    day_df = self.extract_day(
                        report_day=target_day,
                        nios=nios,
                        connection=conn,
                        run_id=run_id,
                    )
                    if not day_df.is_empty():
                        daily_dfs.append(day_df)
                        if save_partitions:
                            write_events_table(
                                day_df,
                                table_name="alarm_events",
                                report_day=target_day,
                                base_dir=base_dir,
                            )
        finally:
            if owned_engine:
                engine.dispose()

        if not daily_dfs:
            return build_empty_frame("alarm_events")

        return pl.concat(daily_dfs, how="vertical_relaxed")
