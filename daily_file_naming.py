"""daily_file_naming.py — Validação de nomenclatura de arquivos diários.

Verifica que cada arquivo de saída segue o padrão ``araucaria_*_YYYYMMDD.csv``
e que o nome corresponde ao ``days_back`` esperado.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path


# Matches: araucaria_cis_20260615.csv, araucaria_mdm_20260615.csv,
#          araucaria_cis_sample_200_20260615.csv, etc.
_DATE_PATTERN = re.compile(r"_(\d{8})\.(?:csv|parquet)$")
_PREFIX_PATTERNS: dict[str, re.Pattern[str]] = {
    "cis": re.compile(r"^araucaria_cis(?:_sample_\d+)?_\d{8}\.(?:csv|parquet)$"),
    "mdm": re.compile(r"^araucaria_mdm(?:_sample_\d+)?_\d{8}\.(?:csv|parquet)$"),
    "model": re.compile(r"^araucaria_model_input(?:_sample_\d+)?_\d{8}\.(?:csv|parquet)$"),
    "geo": re.compile(r"^araucaria_geo_ucs(?:_sample_\d+)?_\d{8}\.(?:csv|parquet)$"),
}


def extract_date_from_filename(filename: str) -> date | None:
    """Extrai a data (YYYYMMDD) do nome de um arquivo.

    Returns:
        ``date`` se encontrada e válida, ``None`` caso contrário.
    """
    match = _DATE_PATTERN.search(filename)
    if not match:
        return None
    try:
        return date.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def validate_file_naming(
    directory: Path,
    *,
    prefix: str | None = None,
) -> list[str]:
    """Valida se os arquivos seguem o padrão de nomenclatura.

    Args:
        directory: Diretório a inspecionar.
        prefix: Filtro opcional — ``"cis"``, ``"mdm"``, ``"model"``, ``"geo"``.

    Returns:
        Lista de nomes de arquivo inválidos (vazia se tudo OK).
    """
    if prefix and prefix not in _PREFIX_PATTERNS:
        raise ValueError(f"Prefixo desconhecido: {prefix!r}. Use: {list(_PREFIX_PATTERNS)}")

    invalid: list[str] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("_"):
            continue
        if path.suffix not in {".csv", ".parquet"}:
            continue

        # Se prefixo foi informado, verificar só aquele padrão
        if prefix:
            pattern = _PREFIX_PATTERNS[prefix]
            if not pattern.match(path.name):
                invalid.append(path.name)
        else:
            # Sem prefixo: verificar se TEM uma data válida
            if extract_date_from_filename(path.name) is None:
                invalid.append(path.name)

    return invalid


def check_days_back_consistency(
    csv_path: Path,
    days_back: int,
    *,
    report_day: date | None = None,
) -> dict:
    """Verifica se o nome do arquivo corresponde ao ``days_back``.

    Returns:
        Dict com ``valid``, ``file_date``, ``expected_date``, ``message``.
    """
    file_date = extract_date_from_filename(csv_path.name)
    if file_date is None:
        return {
            "valid": False,
            "file_date": None,
            "expected_date": None,
            "message": f"Não foi possível extrair data do nome: {csv_path.name}",
        }

    if report_day is None:
        report_day = date.today() - timedelta(days=days_back)

    valid = file_date == report_day
    message = (
        f"OK — {file_date.isoformat()}"
        if valid
        else f"MISMATCH — arquivo: {file_date.isoformat()}, esperado: {report_day.isoformat()}"
    )

    return {
        "valid": valid,
        "file_date": file_date,
        "expected_date": report_day,
        "message": message,
    }
