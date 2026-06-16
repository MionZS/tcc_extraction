from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_LEFT = Path(r"output/manual extractions/araucaria_20260615.csv")
DEFAULT_RIGHT = Path(r"output/manual extractions/mdm_araucaria_flat_20260615.csv")


def normalize_nio(value: object) -> str:
    """Normalize NIO by removing only leading zeros from the textual value."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    normalized = re.sub(r"^0+", "", text)
    return normalized or "0"


def infer_output_path(left_path: Path, right_path: Path) -> Path:
    token = None
    for candidate in (left_path.stem, right_path.stem):
        match = re.search(r"(20\d{6})", candidate)
        if match:
            token = match.group(1)
            break

    if token:
        return left_path.parent / f"model_input_{token}.csv"
    return left_path.parent / f"{left_path.stem}_joined.csv"


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def ensure_nio_column(columns: Iterable[str], path: Path) -> str:
    nio_column = next((col for col in columns if str(col).strip().upper() == "NIO"), None)
    if nio_column is None:
        raise ValueError(f"Coluna NIO não encontrada em {path}")
    return nio_column


def build_lookup(rows: list[dict[str, str]], nio_column: str, source_name: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    lookup: dict[str, dict[str, str]] = {}
    duplicates: Counter[str] = Counter()

    for row in rows:
        key = normalize_nio(row.get(nio_column, ""))
        if not key:
            continue
        if key in lookup:
            duplicates[key] += 1
            continue
        lookup[key] = row

    if duplicates:
        sample = ", ".join(f"{key} ({count + 1} linhas)" for key, count in duplicates.most_common(10))
        raise ValueError(
            f"O arquivo {source_name} não está com uma linha única por NIO normalizado. "
            f"Duplicados: {sample}"
        )

    return lookup, sorted(lookup.keys())


def build_right_column_map(left_columns: list[str], right_columns: list[str], right_nio_column: str) -> tuple[list[str], dict[str, str]]:
    output_columns: list[str] = list(left_columns)
    rename_map: dict[str, str] = {}

    for column in right_columns:
        if column == right_nio_column:
            continue

        output_name = column
        if output_name in output_columns:
            output_name = f"MDM_{output_name}"
            suffix = 2
            while output_name in output_columns:
                output_name = f"MDM_{column}_{suffix}"
                suffix += 1

        rename_map[column] = output_name
        output_columns.append(output_name)

    return output_columns, rename_map


def join_daily_files(left_path: Path, right_path: Path, output_path: Path) -> Path:
    left_columns, left_rows = read_csv_rows(left_path)
    right_columns, right_rows = read_csv_rows(right_path)

    left_nio_column = ensure_nio_column(left_columns, left_path)
    right_nio_column = ensure_nio_column(right_columns, right_path)

    right_lookup, _ = build_lookup(right_rows, right_nio_column, right_path.name)
    output_columns, right_column_map = build_right_column_map(left_columns, right_columns, right_nio_column)

    matched = 0
    unmatched = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for left_row in left_rows:
            output_row = dict(left_row)
            key = normalize_nio(left_row.get(left_nio_column, ""))
            right_row = right_lookup.get(key)

            if right_row is None:
                unmatched += 1
                for right_column, output_name in right_column_map.items():
                    output_row[output_name] = ""
            else:
                matched += 1
                for right_column, output_name in right_column_map.items():
                    output_row[output_name] = right_row.get(right_column, "")

            writer.writerow(output_row)

    print(f"Arquivo esquerdo : {left_path}")
    print(f"Arquivo direito  : {right_path}")
    print(f"Arquivo de saída : {output_path}")
    print(f"Linhas esquerda  : {len(left_rows):,}")
    print(f"Linhas direita   : {len(right_rows):,}")
    print(f"Casados          : {matched:,}")
    print(f"Sem match        : {unmatched:,}")

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faz o join diário entre o arquivo ARAUCARIA e o arquivo MDM flat usando NIO normalizado.",
    )
    parser.add_argument(
        "--left",
        type=Path,
        default=DEFAULT_LEFT,
        help="CSV diário base (ex.: araucaria_YYYYMMDD.csv)",
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=DEFAULT_RIGHT,
        help="CSV diário MDM flat (ex.: mdm_araucaria_flat_YYYYMMDD.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho do CSV final. Se omitido, gera model_input_YYYYMMDD.csv na mesma pasta.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    left_path = args.left.resolve()
    right_path = args.right.resolve()
    output_path = args.output.resolve() if args.output else infer_output_path(left_path, right_path)

    join_daily_files(left_path=left_path, right_path=right_path, output_path=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
