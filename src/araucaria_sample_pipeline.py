from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import oracledb
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.db import create_engine
from export_manager import DEFAULT_PUBLISH_TARGET, publish_to_target, print_publish_report
from src.run_manifest import create_manifest
from src.checks.verify_output import verify_model_input, print_verification_report

oracledb.defaults.fetch_lobs = False

ROOT_DIR = Path(__file__).resolve().parent
QUERIES_DIR = ROOT_DIR / "queries"
DEFAULT_CIS_SQL = QUERIES_DIR / "cis_araucaria_ml_extract_lightweight_alt.sql"
DEFAULT_GEO_SQL = QUERIES_DIR / "geo_ucs.sql"
DEFAULT_MDM_SQL = QUERIES_DIR / "mdm_coluna.sql"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "output"
DEFAULT_RAW_CIS_DIR = DEFAULT_OUTPUT_ROOT / "raw" / "CIS"
DEFAULT_RAW_ORCA_DIR = DEFAULT_OUTPUT_ROOT / "raw" / "ORCA"
DEFAULT_RAW_GEO_DIR = DEFAULT_OUTPUT_ROOT / "raw" / "GEO"
DEFAULT_REFINED_REPORTS_DIR = DEFAULT_OUTPUT_ROOT / "refined" / "reports"
DEFAULT_CIS_FETCH_SIZE = 1000
DEFAULT_MDM_FETCH_SIZE = 100
DEFAULT_SAMPLE_SIZE = 200


@dataclass(frozen=True)
class SampleCandidate:
    source_index: int
    nio: str
    smart: bool
    row: dict[str, str]



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



def _expand_csv_field_size_limit() -> None:
    try:
        csv.field_size_limit(1_000_000_000)
    except OverflowError:
        csv.field_size_limit(2_147_483_647)



def _normalize_nio(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    digits_only = re.sub(r"\D", "", text)
    if not digits_only:
        return ""

    normalized = digits_only.lstrip("0")
    return normalized or "0"



def _is_truthy(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() in {"1", "S", "SIM", "Y", "YES", "TRUE", "T"}



def _materialize_value(value: Any) -> Any:
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



def _materialize_row(row: Sequence[Any]) -> list[Any]:
    return [_materialize_value(value) for value in row]



def _build_oracle_string_list(connection: Connection, values: Sequence[str]):
    raw_conn = connection.connection
    assert raw_conn is not None, "Connection is not attached to a driver connection"
    driver_connection = raw_conn.driver_connection
    assert driver_connection is not None, "Driver connection is not available"
    object_type = driver_connection.gettype("SYS.ODCIVARCHAR2LIST")
    bind_object = object_type.newobject()
    bind_object.extend(list(values))
    return bind_object



def _export_query_to_csv(
    connection: Connection,
    sql_text: str,
    output_path: Path,
    *,
    params: dict[str, Any] | None = None,
    fetch_size: int,
) -> tuple[list[str], int]:
    result = connection.execute(text(sql_text), params or {})
    columns = list(result.keys())
    row_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter=";",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(columns)

        while True:
            rows = result.fetchmany(fetch_size)
            if not rows:
                break

            for row in rows:
                writer.writerow(_materialize_row(row))
                row_count += 1

    return columns, row_count



def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    _expand_csv_field_size_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows



def _find_column(columns: Iterable[str], target: str) -> str | None:
    target_upper = target.strip().upper()
    return next((column for column in columns if str(column).strip().upper() == target_upper), None)



def _ensure_column(columns: Iterable[str], target: str, path: Path) -> str:
    column = _find_column(columns, target)
    if column is None:
        raise ValueError(f"Coluna {target} não encontrada em {path}")
    return column



def _write_rows_to_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def _take_spread_sample(items: Sequence[SampleCandidate], limit: int) -> list[SampleCandidate]:
    if limit <= 0 or not items:
        return []
    if limit >= len(items):
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]

    selected: list[SampleCandidate] = []
    used_indexes: set[int] = set()
    total = len(items)

    for i in range(limit):
        index = round(i * (total - 1) / (limit - 1))

        if index in used_indexes:
            candidate = index + 1
            while candidate < total and candidate in used_indexes:
                candidate += 1
            if candidate < total:
                index = candidate
            else:
                candidate = index - 1
                while candidate >= 0 and candidate in used_indexes:
                    candidate -= 1
                index = candidate

        if index < 0:
            continue

        used_indexes.add(index)
        selected.append(items[index])

    return sorted(selected, key=lambda item: item.source_index)



def _collect_unique_candidates(
    rows: Sequence[dict[str, str]],
    nio_column: str,
    smart_column: str | None,
) -> list[SampleCandidate]:
    seen_nios: set[str] = set()
    candidates: list[SampleCandidate] = []

    for source_index, row in enumerate(rows):
        nio = _normalize_nio(row.get(nio_column, ""))
        if not nio or nio in seen_nios:
            continue

        seen_nios.add(nio)
        smart = _is_truthy(row.get(smart_column, "")) if smart_column else False
        candidates.append(
            SampleCandidate(
                source_index=source_index,
                nio=nio,
                smart=smart,
                row=row,
            )
        )

    return candidates



def _write_sample_inputs(
    cis_csv_path: Path,
    sample_csv_path: Path,
    nio_list_path: Path,
    sample_size: int,
) -> list[str]:
    columns, rows = _read_csv_rows(cis_csv_path)
    nio_column = _ensure_column(columns, "NIO", cis_csv_path)
    smart_column = _find_column(columns, "SMART")

    candidates = _collect_unique_candidates(rows, nio_column, smart_column)
    smart_candidates = [candidate for candidate in candidates if candidate.smart]
    other_candidates = [candidate for candidate in candidates if not candidate.smart]

    if len(smart_candidates) >= sample_size:
        selected = _take_spread_sample(smart_candidates, sample_size)
    else:
        selected = list(smart_candidates)
        selected.extend(_take_spread_sample(other_candidates, sample_size - len(selected)))
        selected = sorted(selected, key=lambda item: item.source_index)

    selected_rows = [candidate.row for candidate in selected]
    selected_nios = [candidate.nio for candidate in selected]

    _write_rows_to_csv(sample_csv_path, columns, selected_rows)
    nio_list_path.parent.mkdir(parents=True, exist_ok=True)
    nio_list_path.write_text("\n".join(selected_nios), encoding="utf-8")

    _log(
        f"Selected {len(selected_nios):,} representative NIOs "
        f"from {len(candidates):,} unique CIS NIOs"
    )
    if smart_column is not None:
        _log(f"SMART NIOs available in CIS: {len(smart_candidates):,}")

    return selected_nios



def _build_lookup(
    rows: Sequence[dict[str, str]],
    nio_column: str,
    source_name: str,
) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    duplicates: dict[str, int] = {}

    for row in rows:
        key = _normalize_nio(row.get(nio_column, ""))
        if not key:
            continue
        if key in lookup:
            duplicates[key] = duplicates.get(key, 1) + 1
            continue
        lookup[key] = row

    if duplicates:
        sample = ", ".join(f"{key} ({count} linhas)" for key, count in list(duplicates.items())[:10])
        raise ValueError(
            f"O arquivo {source_name} não está com uma linha única por NIO normalizado. "
            f"Duplicados: {sample}"
        )

    return lookup



def _build_right_column_map(
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    right_nio_column: str,
    *,
    prefix: str = "MDM_",
) -> tuple[list[str], dict[str, str]]:
    output_columns: list[str] = list(left_columns)
    rename_map: dict[str, str] = {}

    for column in right_columns:
        if column == right_nio_column:
            continue

        output_name = column
        if output_name in output_columns:
            output_name = f"{prefix}{output_name}"
            suffix = 2
            while output_name in output_columns:
                output_name = f"{prefix}{column}_{suffix}"
                suffix += 1

        rename_map[column] = output_name
        output_columns.append(output_name)

    return output_columns, rename_map



def _join_daily_files(left_path: Path, right_path: Path, output_path: Path) -> Path:
    left_columns, left_rows = _read_csv_rows(left_path)
    right_columns, right_rows = _read_csv_rows(right_path)

    left_nio_column = _ensure_column(left_columns, "NIO", left_path)
    right_nio_column = _ensure_column(right_columns, "NIO", right_path)

    left_lookup = _build_lookup(left_rows, left_nio_column, left_path.name)
    output_columns, right_column_map = _build_right_column_map(left_columns, right_columns, right_nio_column)

    matched = 0
    right_only = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for right_row in right_rows:
            key = _normalize_nio(right_row.get(right_nio_column, ""))
            left_row = left_lookup.get(key)

            if left_row is None:
                right_only += 1
                output_row = {column: "" for column in left_columns}
                output_row[left_nio_column] = right_row.get(right_nio_column, "") or key
            else:
                matched += 1
                output_row = dict(left_row)

            for right_column, output_name in right_column_map.items():
                output_row[output_name] = right_row.get(right_column, "")

            writer.writerow(output_row)

    _log(
        f"Right join complete: {matched:,} matched / {right_only:,} ORCA-only / "
        f"{len(right_rows):,} output rows"
    )
    return output_path


def _left_join_daily_files_by_column(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    *,
    join_column: str,
    right_join_column: str | None = None,
    right_prefix: str = "GEO_",
) -> Path:
    left_columns, left_rows = _read_csv_rows(left_path)
    right_columns, right_rows = _read_csv_rows(right_path)

    left_join_column = _ensure_column(left_columns, join_column, left_path)
    right_join_column = _ensure_column(right_columns, right_join_column or join_column, right_path)

    right_lookup = _build_lookup(right_rows, right_join_column, right_path.name)
    output_columns, right_column_map = _build_right_column_map(
        left_columns,
        right_columns,
        right_join_column,
        prefix=right_prefix,
    )

    matched = 0
    left_only = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for left_row in left_rows:
            key = _normalize_nio(left_row.get(left_join_column, ""))
            right_row = right_lookup.get(key)

            output_row = dict(left_row)
            if right_row is None:
                left_only += 1
                for right_column, output_name in right_column_map.items():
                    output_row[output_name] = ""
            else:
                matched += 1
                for right_column, output_name in right_column_map.items():
                    output_row[output_name] = right_row.get(right_column, "")

            writer.writerow(output_row)

    _log(
        f"Left join complete: {matched:,} matched / {left_only:,} left-only / "
        f"{len(left_rows):,} output rows"
    )
    return output_path



def _run_cis_extract(
    cis_sql_path: Path,
    output_path: Path,
    *,
    fetch_size: int,
) -> int:
    cis_sql = _load_sql(cis_sql_path)
    engine = create_engine("cis")
    started_at = time.perf_counter()

    try:
        with engine.connect() as connection:
            _, row_count = _export_query_to_csv(
                connection,
                cis_sql,
                output_path,
                fetch_size=fetch_size,
            )
    finally:
        engine.dispose()

    elapsed = time.perf_counter() - started_at
    _log(f"CIS extract complete: {row_count:,} rows in {elapsed:.1f}s -> {output_path}")
    return row_count



def _run_mdm_extract(
    mdm_sql_path: Path,
    nios: Sequence[str],
    output_path: Path,
    *,
    days_back: int,
    fetch_size: int,
) -> int:
    if not nios:
        raise ValueError("Nenhum NIO foi selecionado para a extração MDM.")

    mdm_sql = _load_sql(mdm_sql_path)
    engine = create_engine("orca")
    started_at = time.perf_counter()

    try:
        with engine.connect() as connection:
            ucs_bind = _build_oracle_string_list(connection, nios)
            _, row_count = _export_query_to_csv(
                connection,
                mdm_sql,
                output_path,
                params={
                    "DAYS_BACK": days_back,
                    "UCS": ucs_bind,
                },
                fetch_size=fetch_size,
            )
    finally:
        engine.dispose()

    elapsed = time.perf_counter() - started_at
    _log(f"MDM extract complete: {row_count:,} rows in {elapsed:.1f}s -> {output_path}")
    return row_count


def _collect_unique_values(path: Path, column_name: str) -> list[str]:
    columns, rows = _read_csv_rows(path)
    column = _ensure_column(columns, column_name, path)

    seen: set[str] = set()
    values: list[str] = []

    for row in rows:
        value = _normalize_nio(row.get(column, ""))
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)

    return values


def _run_geo_extract(
    geo_sql_path: Path,
    ucs: Sequence[str],
    output_path: Path,
    *,
    fetch_size: int,
) -> int:
    if not ucs:
        raise ValueError("Nenhuma UC foi selecionada para a extração GEO.")

    geo_sql = _load_sql(geo_sql_path)
    engine = create_engine("geo")
    started_at = time.perf_counter()

    try:
        with engine.connect() as connection:
            ucs_bind = _build_oracle_string_list(connection, ucs)
            _, row_count = _export_query_to_csv(
                connection,
                geo_sql,
                output_path,
                params={"UCS": ucs_bind},
                fetch_size=fetch_size,
            )
    finally:
        engine.dispose()

    elapsed = time.perf_counter() - started_at
    _log(f"GEO extract complete: {row_count:,} rows in {elapsed:.1f}s -> {output_path}")
    return row_count



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa um teste standalone do pipeline ARAUCARIA: exporta CIS completo, "
            "seleciona 200 NIOs representativos, exporta MDM/ORCA e gera o join final."
        )
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Dias para trás no MDM. 1 = ontem com a semântica atual.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Quantidade de NIOs representativos para testar no MDM.",
    )
    parser.add_argument(
        "--cis-fetch-size",
        type=int,
        default=DEFAULT_CIS_FETCH_SIZE,
        help="Quantidade de linhas por fetch na exportação CIS.",
    )
    parser.add_argument(
        "--mdm-fetch-size",
        type=int,
        default=DEFAULT_MDM_FETCH_SIZE,
        help="Quantidade de linhas por fetch na exportação MDM.",
    )
    parser.add_argument(
        "--cis-sql",
        type=Path,
        default=DEFAULT_CIS_SQL,
        help="Arquivo SQL da extração CIS.",
    )
    parser.add_argument(
        "--geo-sql",
        type=Path,
        default=DEFAULT_GEO_SQL,
        help="Arquivo SQL da extração GEO por UC.",
    )
    parser.add_argument(
        "--mdm-sql",
        type=Path,
        default=DEFAULT_MDM_SQL,
        help="Arquivo SQL da extração MDM/ORCA.",
    )
    parser.add_argument(
        "--cis-output-dir",
        type=Path,
        default=DEFAULT_RAW_CIS_DIR,
        help="Diretório base de saída do CIS (a amostra vai para uma subpasta sampleN).",
    )
    parser.add_argument(
        "--orca-output-dir",
        type=Path,
        default=DEFAULT_RAW_ORCA_DIR,
        help="Diretório base de saída do ORCA/MDM (a amostra vai para uma subpasta sampleN).",
    )
    parser.add_argument(
        "--geo-output-dir",
        type=Path,
        default=DEFAULT_RAW_GEO_DIR,
        help="Diretório base de saída do GEO (a amostra vai para uma subpasta sampleN).",
    )
    parser.add_argument(
        "--joined-output-dir",
        type=Path,
        default=DEFAULT_REFINED_REPORTS_DIR,
        help="Diretório base de saída do arquivo final joinado (a amostra vai para uma subpasta sampleN).",
    )
    parser.add_argument(
        "--publish-target",
        type=Path,
        default=DEFAULT_PUBLISH_TARGET,
        help="Diretório target para publicação (OneDrive).",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Pular etapa de publicação no OneDrive.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Pular etapa de verificação de integridade.",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()

    report_day = _report_day_from_days_back(args.days_back)
    token = _date_token(report_day)

    # Criar manifest de rastreabilidade
    manifest = create_manifest(report_day, args.sample_size)

    cis_output_dir = args.cis_output_dir.resolve()
    orca_output_dir = args.orca_output_dir.resolve()
    geo_output_dir = args.geo_output_dir.resolve()
    joined_output_dir = args.joined_output_dir.resolve()
    sample_tag = f"sample{args.sample_size}"

    cis_sample_output_dir = cis_output_dir / sample_tag
    orca_sample_output_dir = orca_output_dir / sample_tag
    geo_sample_output_dir = geo_output_dir / sample_tag
    joined_sample_output_dir = joined_output_dir / sample_tag

    cis_csv_path = cis_output_dir / f"araucaria_cis_{token}.csv"
    cis_sample_csv_path = cis_sample_output_dir / f"araucaria_cis_sample_{args.sample_size}_{token}.csv"
    cis_sample_geo_csv_path = cis_sample_output_dir / f"araucaria_cis_sample_{args.sample_size}_{token}_geo.csv"
    nio_list_path = cis_sample_output_dir / f"araucaria_sample_nios_{args.sample_size}_{token}.txt"
    geo_csv_path = geo_sample_output_dir / f"araucaria_geo_ucs_sample_{args.sample_size}_{token}.csv"
    mdm_csv_path = orca_sample_output_dir / f"araucaria_mdm_sample_{args.sample_size}_{token}.csv"
    joined_csv_path = joined_sample_output_dir / f"araucaria_model_input_sample_{args.sample_size}_{token}.csv"

    _log("Starting standalone ARAUCARIA sample pipeline")
    _log(f"Target report day: {report_day.isoformat()} (days_back={args.days_back})")
    _log(f"CIS SQL: {Path(args.cis_sql).resolve()}")
    _log(f"GEO SQL: {Path(args.geo_sql).resolve()}")
    _log(f"MDM SQL: {Path(args.mdm_sql).resolve()}")

    # Step 1: CIS extract
    cis_rows = _run_cis_extract(
        Path(args.cis_sql),
        cis_csv_path,
        fetch_size=args.cis_fetch_size,
    )
    manifest.record_step("cis_extract", rows=cis_rows, output=str(cis_csv_path))

    # Step 2: Sample selection
    selected_nios = _write_sample_inputs(
        cis_csv_path,
        cis_sample_csv_path,
        nio_list_path,
        args.sample_size,
    )
    selected_ucs = _collect_unique_values(cis_sample_csv_path, "UC")
    manifest.record_step("sample_selection", nio_count=len(selected_nios), uc_count=len(selected_ucs))

    # Step 3: GEO extract
    geo_sample_output_dir.mkdir(parents=True, exist_ok=True)
    geo_rows = _run_geo_extract(
        Path(args.geo_sql),
        selected_ucs,
        geo_csv_path,
        fetch_size=args.cis_fetch_size,
    )
    manifest.record_step("geo_extract", rows=geo_rows, output=str(geo_csv_path))

    # Step 4: GEO join
    _left_join_daily_files_by_column(
        cis_sample_csv_path,
        geo_csv_path,
        cis_sample_geo_csv_path,
        join_column="UC",
        right_join_column="UC",
        right_prefix="GEO_",
    )
    manifest.record_step("geo_join", output=str(cis_sample_geo_csv_path))

    # Step 5: MDM extract
    mdm_rows = _run_mdm_extract(
        Path(args.mdm_sql),
        selected_nios,
        mdm_csv_path,
        days_back=args.days_back,
        fetch_size=args.mdm_fetch_size,
    )
    manifest.record_step("mdm_extract", rows=mdm_rows, output=str(mdm_csv_path))

    # Step 6: Final join
    _join_daily_files(cis_sample_geo_csv_path, mdm_csv_path, joined_csv_path)
    manifest.record_step("final_join", output=str(joined_csv_path))

    # Step 7: Verify output
    if not args.no_verify:
        verification = verify_model_input(joined_csv_path, expected_rows=args.sample_size)
        print_verification_report(verification)
        manifest.record_step("verify", **verification.to_dict())
        if not verification.passed:
            manifest.add_error("Verificação de integridade falhou")
            manifest.finalize()
            manifest.save()
            return 1

    # Step 8: Publish to OneDrive target
    if not args.no_publish:
        publish_result = publish_to_target(joined_csv_path, args.publish_target.resolve())
        print_publish_report(publish_result)
        manifest.record_step("publish", target=str(publish_result.target), bytes=publish_result.bytes_copied)

    # Step 9: Save run manifest
    manifest.finalize()
    manifest_path = manifest.save()
    _log(f"Run manifest saved: {manifest_path}")

    print()
    print("Summary")
    print("-------")
    print(f"Report day         : {report_day.isoformat()}")
    print(f"CIS rows           : {cis_rows:,}")
    print(f"Sampled NIOs       : {len(selected_nios):,}")
    print(f"Sampled UCs        : {len(selected_ucs):,}")
    print(f"GEO rows           : {geo_rows:,}")
    print(f"MDM rows           : {mdm_rows:,}")
    print(f"CIS CSV            : {cis_csv_path}")
    print(f"CIS sample CSV     : {cis_sample_csv_path}")
    print(f"CIS+GEO sample CSV : {cis_sample_geo_csv_path}")
    print(f"GEO CSV            : {geo_csv_path}")
    print(f"Sample NIO list    : {nio_list_path}")
    print(f"MDM CSV            : {mdm_csv_path}")
    print(f"Joined CSV         : {joined_csv_path}")
    print(f"Run manifest       : {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
