"""Daily ARAUCARIA extraction pipeline.

Flow
----
1. Run the CIS query to get the ARAUCARIA population.
2. Extract unique UCs from that result.
3. Feed those UCs to the GEO query through SQLAlchemy using
   ``SYS.ODCIVARCHAR2LIST``.
4. Left-join GEO data on CIS by UC.
5. Extract unique NIOs from the joined result.
6. Feed those NIOs to the ORCA/MDM query through SQLAlchemy using
   ``SYS.ODCIVARCHAR2LIST`` (with batching).
7. Save outputs in a datalake-style layout:
   - raw/CIS
   - raw/GEO
   - raw/ORCA
   - refined/reports
8. Publish final CSV to OneDrive target.
9. Save run manifest.
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
from src.run_manifest import create_manifest
from src.export_manager import publish_to_target

QUERIES_DIR = Path(__file__).resolve().parent / "queries"
CIS_QUERY_PATH = QUERIES_DIR / "cis_araucaria_ml_extract_lightweight_alt.sql"
GEO_QUERY_PATH = QUERIES_DIR / "geo_ucs.sql"
MDM_QUERY_PATH = QUERIES_DIR / "mdm_coluna.sql"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_RAW_CIS_DIR = DEFAULT_OUTPUT_DIR / "raw" / "CIS"
DEFAULT_RAW_GEO_DIR = DEFAULT_OUTPUT_DIR / "raw" / "GEO"
DEFAULT_RAW_ORCA_DIR = DEFAULT_OUTPUT_DIR / "raw" / "ORCA"
DEFAULT_REFINED_REPORTS_DIR = DEFAULT_OUTPUT_DIR / "refined" / "reports"
DEFAULT_MDM_BATCH_SIZE = 500
DEFAULT_GEO_BATCH_SIZE = 500
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
    geo_parquet: Path | None
    geo_csv: Path | None
    mdm_parquet: Path | None
    mdm_csv: Path | None
    joined_parquet: Path | None
    joined_csv: Path | None
    total_cis_rows: int
    total_geo_rows: int
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

    # Also normalize UC column to uppercase (case-insensitive lookup)
    uc_column = next(
        (column for column in dataframe.columns if str(column).strip().upper() == "UC"),
        None,
    )
    if uc_column is not None and uc_column != "UC":
        dataframe = dataframe.rename({uc_column: "UC"})

    # Normalizar valores de UC (remover zeros à esquerda) igual ao GEO faz
    if uc_column is not None:
        dataframe = _normalize_nio_column(dataframe, "UC")

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


def _extract_ucs(df: pl.DataFrame, column_name: str = "UC") -> list[str]:
    """Extrai valores únicos da coluna UC, normalizados.

    Returns:
        Lista ordenada de UCs (dígitos, sem zeros à esquerda).
    """
    if column_name not in df.columns:
        _log(f"Column {column_name!r} not found in DataFrame; no UCs to extract.")
        return []

    ucs: set[str] = set()
    for value in df.get_column(column_name).drop_nulls().unique():
        text = str(value).strip()
        digits_only = re.sub(r"\D", "", text)
        if not digits_only:
            continue
        normalized = digits_only.lstrip("0")
        if normalized:
            ucs.add(normalized)

    sorted_ucs = sorted(ucs)
    _log(f"Unique UCs extracted: {len(sorted_ucs):,}")
    return sorted_ucs


def _extract_geo_dataframe(
    geo_sql_path: Path,
    ucs: Sequence[str],
    fetch_size: int,
    batch_size: int = DEFAULT_GEO_BATCH_SIZE,
) -> pl.DataFrame:
    """Extrai dados GEO para a lista de UCs fornecida.

    Usa batching (``ODCIVARCHAR2LIST``) similar ao MDM.
    Retorna DataFrame vazio se ``ucs`` estiver vazio.
    """
    if not ucs:
        _log("No UCs provided; skipping GEO extract.")
        return _build_empty_frame([])

    geo_sql = _load_sql(geo_sql_path)
    engine = create_engine("geo")
    started_at = time.perf_counter()
    total_rows = 0
    geo_columns: list[str] = []
    geo_chunks: list[pl.DataFrame] = []

    try:
        with engine.connect() as connection:
            for batch_index, ucs_batch in enumerate(
                _chunked(list(ucs), batch_size), start=1
            ):
                batch_started_at = time.perf_counter()
                _log(
                    f"GEO batch {batch_index}: {len(ucs_batch):,} UCs "
                    f"({ucs_batch[0]}..{ucs_batch[-1]})"
                )

                oracle_list = _build_oracle_string_list(connection, ucs_batch)
                result = connection.execute(
                    text(geo_sql),
                    {"UCS": oracle_list},
                )

                current_columns = list(result.keys())
                if not geo_columns:
                    geo_columns = current_columns

                while True:
                    rows = result.fetchmany(fetch_size)
                    if not rows:
                        break
                    geo_chunks.append(_rows_to_frame(rows, current_columns))

                batch_elapsed = time.perf_counter() - batch_started_at
                _log(f"GEO batch {batch_index}: done in {batch_elapsed:.1f}s")
    finally:
        engine.dispose()

    if not geo_chunks:
        _log("GEO query returned no rows.")
        return _build_empty_frame(geo_columns or [])

    geo_df = pl.concat(geo_chunks, how="vertical_relaxed")
    # Normalizar nomes de colunas para maiúsculo (Oracle pode retornar case variado)
    geo_df = geo_df.rename({col: col.upper() for col in geo_df.columns})
    total_rows = geo_df.height

    elapsed = time.perf_counter() - started_at
    _log(f"GEO extract complete: {total_rows:,} rows in {elapsed:.1f}s")
    return geo_df


def _left_join_geo(
    cis_df: pl.DataFrame,
    geo_df: pl.DataFrame,
    *,
    join_column: str = "UC",
    prefix: str = "GEO_",
) -> pl.DataFrame:
    """Faz left join do GEO no CIS por UC, prefixando colunas GEO.

    Retorna o DataFrame CIS com colunas GEO_* adicionadas.
    """
    if geo_df.is_empty() or geo_df.width == 0:
        _log("GEO DataFrame is empty; returning CIS unchanged.")
        return cis_df

    if join_column not in cis_df.columns:
        _log(f"Join column {join_column!r} not in CIS DataFrame; returning CIS unchanged.")
        return cis_df

    if join_column not in geo_df.columns:
        _log(f"Join column {join_column!r} not in GEO DataFrame; returning CIS unchanged.")
        return cis_df

    # Rename GEO columns with prefix (except the join column)
    geo_rename = {
        col: f"{prefix}{col}"
        for col in geo_df.columns
        if col != join_column
    }

    geo_renamed = geo_df.rename(geo_rename)

    # Garantir que a coluna de join seja string em ambos os lados
    cis_df = cis_df.with_columns(pl.col(join_column).cast(pl.Utf8))
    geo_renamed = geo_renamed.with_columns(pl.col(join_column).cast(pl.Utf8))

    result = cis_df.join(
        geo_renamed,
        on=join_column,
        how="inner",
    )

    _log(f"Left join GEO complete: {result.height:,} rows, {result.width:,} columns")
    return result


def _build_oracle_string_list(connection: Connection, values: Sequence[str]):
    driver_connection = connection.connection.driver_connection
    object_type = driver_connection.gettype("SYS.ODCIVARCHAR2LIST")
    return object_type.newobject(list(values))



def _merge_temp_parquets(parquet_files: Sequence[Path], final_parquet: Path, final_csv: Path) -> None:
    lazy_frame = pl.scan_parquet([str(path) for path in parquet_files]).sort(["dia", "nio"])
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
        .join(
            pl.scan_parquet(str(mdm_result.parquet_path)).rename({"nio": "NIO"}),
            on="NIO",
            how="left",
        )
        .with_columns(
            pl.lit(report_day_text).alias("REPORT_DAY"),
            pl.col("dia").is_not_null().fill_null(False).alias("HAS_MDM_DATA"),
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
    geo_batch_size: int = DEFAULT_GEO_BATCH_SIZE,
    fetch_size: int = DEFAULT_FETCH_SIZE,
    cis_sql_path: Path | str = CIS_QUERY_PATH,
    geo_sql_path: Path | str = GEO_QUERY_PATH,
    mdm_sql_path: Path | str = MDM_QUERY_PATH,
    publish_target: Path | None = None,
    keep_temp: bool = False,
) -> PipelineResult:
    """Run the daily ARAUCARIA extraction pipeline with CIS, GEO and MDM.

    Flow: CIS → GEO (left join por UC) → MDM → relatório final.

    ``days_back=1`` means yesterday with the current SQL semantics.
    """
    output_dir = Path(output_dir)
    cis_sql_path = Path(cis_sql_path)
    geo_sql_path = Path(geo_sql_path)
    mdm_sql_path = Path(mdm_sql_path)
    report_day = _report_day_from_days_back(days_back)

    raw_cis_dir = output_dir / "raw" / "CIS"
    raw_geo_dir = output_dir / "raw" / "GEO"
    raw_orca_dir = output_dir / "raw" / "ORCA"
    refined_reports_dir = output_dir / "refined" / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cis_dir.mkdir(parents=True, exist_ok=True)
    raw_geo_dir.mkdir(parents=True, exist_ok=True)
    raw_orca_dir.mkdir(parents=True, exist_ok=True)
    refined_reports_dir.mkdir(parents=True, exist_ok=True)

    # Manifesto de rastreabilidade
    manifest = create_manifest(report_day, sample_size=0)

    _log(f"Starting ARAUCARIA daily pipeline")
    _log(f"Target report day: {report_day.isoformat()} (days_back={days_back})")
    _log(f"CIS query: {cis_sql_path}")
    _log(f"GEO query: {geo_sql_path}")
    _log(f"MDM query: {mdm_sql_path}")

    # --- Step 1: CIS extract ---
    cis_df = _extract_cis_dataframe(cis_sql_path)
    cis_parquet = raw_cis_dir / f"araucaria_cis_{_date_token(report_day)}.parquet"
    cis_csv = raw_cis_dir / f"araucaria_cis_{_date_token(report_day)}.csv"
    _write_dataframe_outputs(cis_df, cis_parquet, cis_csv)
    manifest.record_step("cis_extract", rows=cis_df.height, output=str(cis_csv))

    # --- Step 2: GEO extract ---
    ucs = _extract_ucs(cis_df)
    geo_df = _extract_geo_dataframe(
        geo_sql_path=geo_sql_path,
        ucs=ucs,
        fetch_size=fetch_size,
        batch_size=geo_batch_size,
    )
    geo_parquet = raw_geo_dir / f"araucaria_geo_ucs_{_date_token(report_day)}.parquet"
    geo_csv = raw_geo_dir / f"araucaria_geo_ucs_{_date_token(report_day)}.csv"
    _write_dataframe_outputs(geo_df, geo_parquet, geo_csv)
    manifest.record_step("geo_extract", rows=geo_df.height, output=str(geo_csv))

    # --- Step 3: Left join GEO on CIS ---
    cis_geo_df = _left_join_geo(cis_df, geo_df, join_column="UC", prefix="GEO_")
    manifest.record_step("geo_join", rows=cis_geo_df.height)

    # --- Step 4: MDM extract (usando NIOs do CIS+GEO) ---
    nios = _extract_nios(cis_geo_df)
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
    manifest.record_step(
        "mdm_extract",
        rows=mdm_result.row_count,
        output=str(mdm_result.csv_path) if mdm_result.csv_path else None,
    )

    # --- Step 5: Final join (CIS+GEO + MDM) ---
    joined_parquet, joined_csv = _build_joined_report(
        cis_df=cis_geo_df,
        mdm_result=mdm_result,
        output_dir=refined_reports_dir,
        report_day=report_day,
    )
    manifest.record_step("final_join", output=str(joined_csv) if joined_csv else None)

    # --- Step 6: Publish para OneDrive ---
    if publish_target is not None and joined_csv is not None:
        try:
            pub_result = publish_to_target(joined_csv, publish_target)
            manifest.record_step(
                "publish",
                target=str(pub_result.target),
                bytes=pub_result.bytes_copied,
            )
            _log(f"Published to: {pub_result.target}")
        except Exception as exc:
            _log(f"Publish failed: {exc}")
            manifest.add_error(f"Publish failed: {exc}")

    # --- Step 7: Finalizar e salvar manifest ---
    manifest.finalize()
    try:
        manifest_path = manifest.save()
        _log(f"Run manifest saved: {manifest_path}")
    except Exception as exc:
        _log(f"Manifest save failed: {exc}")

    _log("Pipeline complete")
    _log(f"CIS CSV  : {cis_csv}")
    _log(f"GEO CSV  : {geo_csv}")
    if mdm_result.csv_path is not None:
        _log(f"MDM CSV  : {mdm_result.csv_path}")
    if joined_csv is not None:
        _log(f"Joined   : {joined_csv}")

    return PipelineResult(
        report_day=report_day,
        cis_parquet=cis_parquet,
        cis_csv=cis_csv,
        geo_parquet=geo_parquet,
        geo_csv=geo_csv,
        mdm_parquet=mdm_result.parquet_path,
        mdm_csv=mdm_result.csv_path,
        joined_parquet=joined_parquet,
        joined_csv=joined_csv,
        total_cis_rows=cis_df.height,
        total_geo_rows=geo_df.height,
        total_nios=len(nios),
        total_mdm_rows=mdm_result.row_count,
    )


@dataclass(frozen=True)
class PeriodResult:
    """Resultado de uma execução em período (múltiplos dias)."""
    results: list[PipelineResult]
    total_days: int
    success_days: int
    failed_days: int
    start_date: date
    end_date: date
    duration_seconds: float


def run_period_pipeline(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    mdm_batch_size: int = DEFAULT_MDM_BATCH_SIZE,
    geo_batch_size: int = DEFAULT_GEO_BATCH_SIZE,
    fetch_size: int = DEFAULT_FETCH_SIZE,
    cis_sql_path: Path | str = CIS_QUERY_PATH,
    geo_sql_path: Path | str = GEO_QUERY_PATH,
    mdm_sql_path: Path | str = MDM_QUERY_PATH,
    publish_target: Path | None = None,
    keep_temp: bool = False,
    continue_on_error: bool = True,
) -> PeriodResult:
    """Executa a pipeline para um período de dias.

    Itera de ``start_date`` a ``end_date`` (inclusive), calculando
    o ``days_back`` apropriado para cada dia e chamando
    ``run_daily_araucaria_pipeline()``.

    Args:
        start_date: Data inicial (inclusive).
        end_date: Data final (inclusive).
        continue_on_error: Se True, continua no próximo dia mesmo se
            um dia falhar.
            Demais parâmetros: mesmos de ``run_daily_araucaria_pipeline()``.

    Returns:
        PeriodResult com a lista de resultados individuais.
    """
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) não pode ser posterior a end_date ({end_date})"
        )

    output_dir = Path(output_dir)
    cis_sql_path = Path(cis_sql_path)
    geo_sql_path = Path(geo_sql_path)
    mdm_sql_path = Path(mdm_sql_path)

    today = date.today()
    results: list[PipelineResult] = []
    total_days = (end_date - start_date).days + 1

    _log(f"Starting period pipeline: {start_date.isoformat()} to {end_date.isoformat()}")
    _log(f"Total days: {total_days}")

    started_at = time.perf_counter()

    for i in range(total_days):
        day = start_date + timedelta(days=i)
        days_back = (today - day).days

        if days_back < 0:
            _log(f"Skipping future date: {day.isoformat()}")
            continue

        _log(f"\n--- Day {i + 1}/{total_days}: {day.isoformat()} (days_back={days_back}) ---")

        try:
            result = run_daily_araucaria_pipeline(
                output_dir=output_dir,
                days_back=days_back,
                mdm_batch_size=mdm_batch_size,
                geo_batch_size=geo_batch_size,
                fetch_size=fetch_size,
                cis_sql_path=cis_sql_path,
                geo_sql_path=geo_sql_path,
                mdm_sql_path=mdm_sql_path,
                publish_target=publish_target,
                keep_temp=keep_temp,
            )
            results.append(result)
            _log(f"  ✅ Day {day.isoformat()} OK — CIS: {result.total_cis_rows:,}, GEO: {result.total_geo_rows:,}, MDM: {result.total_mdm_rows:,}")
        except Exception as exc:
            _log(f"  ❌ Day {day.isoformat()} FAILED: {exc}")
            if not continue_on_error:
                raise
            # Append a placeholder for failed days
            results.append(PipelineResult(
                report_day=day,
                cis_parquet=Path(),
                cis_csv=Path(),
                geo_parquet=None,
                geo_csv=None,
                mdm_parquet=None,
                mdm_csv=None,
                joined_parquet=None,
                joined_csv=None,
                total_cis_rows=0,
                total_geo_rows=0,
                total_nios=0,
                total_mdm_rows=0,
            ))

    elapsed = time.perf_counter() - started_at
    success = sum(1 for r in results if r.total_cis_rows > 0)
    failed = sum(1 for r in results if r.total_cis_rows == 0)

    _log(f"\nPeriod pipeline complete")
    _log(f"Total days : {total_days}")
    _log(f"Success    : {success}")
    _log(f"Failed     : {failed}")
    _log(f"Duration   : {elapsed:.1f}s")

    return PeriodResult(
        results=results,
        total_days=total_days,
        success_days=success,
        failed_days=failed,
        start_date=start_date,
        end_date=end_date,
        duration_seconds=elapsed,
    )
