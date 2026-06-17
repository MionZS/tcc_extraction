"""Daily ARAUCARIA extraction pipeline.

Flow
----
1. Run the CIS query to get the ARAUCARIA population.
2. Extract unique NIOs from that result.
3. Feed those NIOs to the ORCA/MDM query through SQLAlchemy using
   ``SYS.ODCIVARCHAR2LIST``.
4. Save outputs in a datalake-style layout:
   - raw/CIS
   - raw/ORCA
   - refined/reports
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Sequence

import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.db import create_engine

QUERIES_DIR = Path(__file__).resolve().parent / "queries"
CIS_QUERY_PATH = QUERIES_DIR / "cis_araucaria_ml_extract_lightweight_alt.sql"
MDM_QUERY_PATH = QUERIES_DIR / "mdm_coluna.sql"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_RAW_CIS_DIR = DEFAULT_OUTPUT_DIR / "raw" / "CIS"
DEFAULT_RAW_ORCA_DIR = DEFAULT_OUTPUT_DIR / "raw" / "ORCA"
DEFAULT_REFINED_REPORTS_DIR = DEFAULT_OUTPUT_DIR / "refined" / "reports"
DEFAULT_MDM_BATCH_SIZE = 500
DEFAULT_FETCH_SIZE = 1000


@dataclass(frozen=True)
class MdmExtractResult:
    parquet_path: Path | None
    csv_path: Path | None
    row_count: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class PipelineResult:
    report_day: date
    cis_parquet: Path
    cis_csv: Path
    mdm_parquet: Path | None
    mdm_csv: Path | None
    joined_parquet: Path | None
    joined_csv: Path | None
    total_cis_rows: int
    total_nios: int
    total_mdm_rows: int



def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}")



def _load_sql(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    sql_text = path.read_text(encoding="utf-8").strip()
    while sql_text.endswith(";"):
        sql_text = sql_text[:-1].rstrip()
    if sql_text.endswith("/"):
        sql_text = sql_text[:-1].rstrip()
    return sql_text



def _report_day_from_days_back(days_back: int) -> date:
    if days_back < 0:
        raise ValueError("days_back must be >= 0")
    return date.today() - timedelta(days=days_back)



def _date_token(report_day: date) -> str:
    return report_day.strftime("%Y%m%d")



def _materialize_scalar(value: Any) -> Any:
    if value is None:
        return None

    reader = getattr(value, "read", None)
    if callable(reader):
        try:
            value = reader()
        except Exception:
            return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin1", errors="replace")

    return value



def _materialize_rows(rows: Sequence[Any]) -> list[tuple[Any, ...]]:
    return [tuple(_materialize_scalar(value) for value in row) for row in rows]



def _rows_to_frame(rows: Sequence[Any], columns: Sequence[str]) -> pl.DataFrame:
    if not rows:
        return _build_empty_frame(columns)

    materialized_rows = _materialize_rows(rows)
    return pl.DataFrame(
        materialized_rows,
        schema=list(columns),
        orient="row",
        infer_schema_length=None,
        strict=False,
    )



def _chunked(values: Sequence[str], size: int) -> Iterator[list[str]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])



def _normalize_nio(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        text_value = str(value)
    elif isinstance(value, Decimal):
        if value == value.to_integral_value():
            text_value = str(int(value))
        else:
            text_value = format(value, "f")
    elif isinstance(value, float):
        if value.is_integer():
            text_value = str(int(value))
        else:
            text_value = str(value)
    else:
        text_value = str(value)

    digits_only = re.sub(r"\D", "", text_value)
    if not digits_only:
        return None

    normalized = digits_only.lstrip("0")
    if not normalized:
        return None

    return normalized



def _normalize_nio_column(df: pl.DataFrame, column_name: str = "NIO") -> pl.DataFrame:
    if column_name not in df.columns:
        return df

    return df.with_columns(
        pl.col(column_name)
        .map_elements(_normalize_nio, return_dtype=pl.String)
        .alias(column_name)
    )



def _build_empty_frame(columns: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.String for column in columns})



def _write_dataframe_outputs(df: pl.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    df.write_csv(csv_path, separator=";")



def _prepare_temp_dir(output_dir: Path, report_day: date) -> Path:
    tmp_dir = output_dir / "tmp" / f"araucaria_{_date_token(report_day)}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir



def _execute_query(connection: Connection, sql_text: str, params: dict[str, Any] | None = None) -> tuple[pl.DataFrame, list[str]]:
    result = connection.execute(text(sql_text), params or {})
    columns = list(result.keys())
    rows = result.fetchall()
    return _rows_to_frame(rows, columns), columns



def _extract_cis_dataframe(cis_sql_path: Path) -> pl.DataFrame:
    cis_sql = _load_sql(cis_sql_path)
    engine = create_engine("cis")
    started_at = time.perf_counter()

    try:
        with engine.connect() as connection:
            dataframe, _ = _execute_query(connection, cis_sql)
    finally:
        engine.dispose()

    nio_column = next(
        (column for column in dataframe.columns if str(column).strip().upper() == "NIO"),
        None,
    )
    if nio_column is None:
        raise ValueError(
            f"CIS query output must contain an NIO column. Returned columns: {dataframe.columns}"
        )
    if nio_column != "NIO":
        dataframe = dataframe.rename({nio_column: "NIO"})

    dataframe = (
        _normalize_nio_column(dataframe, "NIO")
        .filter(pl.col("NIO").is_not_null())
        .unique(subset=["NIO"], keep="first", maintain_order=True)
        .sort("NIO")
    )

    elapsed = time.perf_counter() - started_at
    _log(f"CIS extract complete: {dataframe.height:,} rows in {elapsed:.1f}s")
    return dataframe



def _extract_nios(cis_df: pl.DataFrame) -> list[str]:
    if cis_df.is_empty():
        return []
    return sorted(cis_df.get_column("NIO").drop_nulls().unique().to_list())



def _build_oracle_string_list(connection: Connection, values: Sequence[str]):
    driver_connection = connection.connection.driver_connection
    object_type = driver_connection.gettype("SYS.ODCIVARCHAR2LIST")
    return object_type.newobject(list(values))



def _merge_temp_parquets(parquet_files: Sequence[Path], final_parquet: Path, final_csv: Path) -> None:
    lazy_frame = pl.scan_parquet([str(path) for path in parquet_files]).sort(["DIA", "NIO"])
    lazy_frame.sink_parquet(final_parquet)
    lazy_frame.sink_csv(final_csv, separator=";")



def _extract_mdm_dataframe(
    mdm_sql_path: Path,
    nios: Sequence[str],
    output_dir: Path,
    report_day: date,
    days_back: int,
    batch_size: int,
    fetch_size: int,
    *,
    keep_temp: bool = False,
) -> MdmExtractResult:
    if not nios:
        _log("No NIOs returned by CIS; skipping MDM extract.")
        return MdmExtractResult(None, None, 0, tuple())

    mdm_sql = _load_sql(mdm_sql_path)
    tmp_dir = _prepare_temp_dir(output_dir, report_day)
    final_parquet = output_dir / f"araucaria_mdm_{_date_token(report_day)}.parquet"
    final_csv = output_dir / f"araucaria_mdm_{_date_token(report_day)}.csv"

    batch_files: list[Path] = []
    mdm_columns: list[str] = []
    total_rows = 0

    engine = create_engine("orca")
    started_at = time.perf_counter()

    try:
        with engine.connect() as connection:
            for batch_index, nio_batch in enumerate(_chunked(list(nios), batch_size), start=1):
                batch_started_at = time.perf_counter()
                _log(
                    f"MDM batch {batch_index}: {len(nio_batch):,} NIOs "
                    f"({nio_batch[0]}..{nio_batch[-1]})"
                )

                oracle_list = _build_oracle_string_list(connection, nio_batch)
                result = connection.execute(
                    text(mdm_sql),
                    {
                        "DAYS_BACK": days_back,
                        "UCS": oracle_list,
                    },
                )

                current_columns = list(result.keys())
                if not mdm_columns:
                    mdm_columns = current_columns

                dataframe_chunks: list[pl.DataFrame] = []
                while True:
                    rows = result.fetchmany(fetch_size)
                    if not rows:
                        break
                    dataframe_chunks.append(
                        _normalize_nio_column(_rows_to_frame(rows, current_columns), "NIO")
                    )

                batch_elapsed = time.perf_counter() - batch_started_at
                if not dataframe_chunks:
                    _log(f"MDM batch {batch_index}: 0 rows in {batch_elapsed:.1f}s")
                    continue

                batch_df = pl.concat(dataframe_chunks, how="vertical_relaxed")
                total_rows += batch_df.height

                batch_path = tmp_dir / f"mdm_batch_{batch_index:05d}.parquet"
                batch_df.write_parquet(batch_path)
                batch_files.append(batch_path)

                _log(
                    f"MDM batch {batch_index}: {batch_df.height:,} rows in {batch_elapsed:.1f}s "
                    f"-> {batch_path.name}"
                )
    finally:
        engine.dispose()

    if batch_files:
        _merge_temp_parquets(batch_files, final_parquet, final_csv)
    elif mdm_columns:
        empty_df = _build_empty_frame(mdm_columns)
        _write_dataframe_outputs(empty_df, final_parquet, final_csv)
        _log("MDM query returned no rows for the requested day.")
    else:
        if not keep_temp and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return MdmExtractResult(None, None, 0, tuple())

    if not keep_temp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    elapsed = time.perf_counter() - started_at
    _log(f"MDM extract complete: {total_rows:,} rows in {elapsed:.1f}s")
    return MdmExtractResult(final_parquet, final_csv, total_rows, tuple(mdm_columns))



def _build_joined_report(
    cis_df: pl.DataFrame,
    mdm_result: MdmExtractResult,
    output_dir: Path,
    report_day: date,
) -> tuple[Path | None, Path | None]:
    joined_parquet = output_dir / f"araucaria_daily_report_{_date_token(report_day)}.parquet"
    joined_csv = output_dir / f"araucaria_daily_report_{_date_token(report_day)}.csv"
    report_day_text = report_day.isoformat()

    if mdm_result.parquet_path is None:
        joined_df = cis_df.with_columns(
            pl.lit(report_day_text).alias("REPORT_DAY"),
            pl.lit(False).alias("HAS_MDM_DATA"),
        )
        _write_dataframe_outputs(joined_df, joined_parquet, joined_csv)
        return joined_parquet, joined_csv

    joined_lazy = (
        cis_df.lazy()
        .join(pl.scan_parquet(str(mdm_result.parquet_path)), on="NIO", how="left")
        .with_columns(
            pl.lit(report_day_text).alias("REPORT_DAY"),
            pl.col("DIA").is_not_null().fill_null(False).alias("HAS_MDM_DATA"),
        )
        .sort(["NIO"])
    )

    joined_lazy.sink_parquet(joined_parquet)
    joined_lazy.sink_csv(joined_csv, separator=";")
    return joined_parquet, joined_csv



def run_daily_araucaria_pipeline(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    days_back: int = 1,
    mdm_batch_size: int = DEFAULT_MDM_BATCH_SIZE,
    fetch_size: int = DEFAULT_FETCH_SIZE,
    cis_sql_path: Path | str = CIS_QUERY_PATH,
    mdm_sql_path: Path | str = MDM_QUERY_PATH,
    keep_temp: bool = False,
) -> PipelineResult:
    """Run the daily CIS -> MDM chained extraction for ARAUCARIA.

    ``days_back=1`` means yesterday with the current SQL semantics.
    """
    output_dir = Path(output_dir)
    cis_sql_path = Path(cis_sql_path)
    mdm_sql_path = Path(mdm_sql_path)
    report_day = _report_day_from_days_back(days_back)

    raw_cis_dir = output_dir / "raw" / "CIS"
    raw_orca_dir = output_dir / "raw" / "ORCA"
    refined_reports_dir = output_dir / "refined" / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cis_dir.mkdir(parents=True, exist_ok=True)
    raw_orca_dir.mkdir(parents=True, exist_ok=True)
    refined_reports_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Starting ARAUCARIA daily pipeline")
    _log(f"Target report day: {report_day.isoformat()} (days_back={days_back})")
    _log(f"CIS query: {cis_sql_path}")
    _log(f"MDM query: {mdm_sql_path}")

    cis_df = _extract_cis_dataframe(cis_sql_path)
    cis_parquet = raw_cis_dir / f"araucaria_cis_{_date_token(report_day)}.parquet"
    cis_csv = raw_cis_dir / f"araucaria_cis_{_date_token(report_day)}.csv"
    _write_dataframe_outputs(cis_df, cis_parquet, cis_csv)

    nios = _extract_nios(cis_df)
    _log(f"Unique NIOs selected for MDM: {len(nios):,}")

    mdm_result = _extract_mdm_dataframe(
        mdm_sql_path=mdm_sql_path,
        nios=nios,
        output_dir=raw_orca_dir,
        report_day=report_day,
        days_back=days_back,
        batch_size=mdm_batch_size,
        fetch_size=fetch_size,
        keep_temp=keep_temp,
    )

    joined_parquet, joined_csv = _build_joined_report(
        cis_df=cis_df,
        mdm_result=mdm_result,
        output_dir=refined_reports_dir,
        report_day=report_day,
    )

    _log("Pipeline complete")
    _log(f"CIS CSV: {cis_csv}")
    if mdm_result.csv_path is not None:
        _log(f"MDM CSV: {mdm_result.csv_path}")
    if joined_csv is not None:
        _log(f"Joined CSV: {joined_csv}")

    return PipelineResult(
        report_day=report_day,
        cis_parquet=cis_parquet,
        cis_csv=cis_csv,
        mdm_parquet=mdm_result.parquet_path,
        mdm_csv=mdm_result.csv_path,
        joined_parquet=joined_parquet,
        joined_csv=joined_csv,
        total_cis_rows=cis_df.height,
        total_nios=len(nios),
        total_mdm_rows=mdm_result.row_count,
    )
