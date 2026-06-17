"""extract_by_days_back.py — Extração sob demanda de arquivos por days_back.

Permite extrair um arquivo específico do diretório de saída
para um diretório de destino, com o nome correto.
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

from daily_file_naming import extract_date_from_filename


def _report_day_from_days_back(days_back: int) -> date:
    """Calcula a data de referência a partir de days_back."""
    if days_back < 0:
        raise ValueError("days_back must be >= 0")
    return date.today() - timedelta(days=days_back)


def find_file_for_days_back(
    source_dir: Path,
    days_back: int,
    *,
    prefix: str = "araucaria_cis",
    suffix: str = ".csv",
) -> Path | None:
    """Encontra o arquivo no ``source_dir`` correspondente ao ``days_back``.

    Args:
        source_dir: Diretório de origem.
        days_back: Dias para trás (1 = ontem).
        prefix: Prefixo do nome do arquivo.
        suffix: Extensão do arquivo.

    Returns:
        ``Path`` do arquivo encontrado ou ``None``.
    """
    target_date = _report_day_from_days_back(days_back)
    pattern = f"{prefix}*{suffix}"

    for path in source_dir.glob(pattern):
        file_date = extract_date_from_filename(path.name)
        if file_date == target_date:
            return path
    return None


def extract_file_by_days_back(
    source_dir: Path,
    days_back: int,
    output_dir: Path,
    *,
    prefix: str = "araucaria_cis",
    suffix: str = ".csv",
    filename_override: str | None = None,
) -> Path | None:
    """Copia arquivo de um dia específico para o diretório de destino.

    Args:
        source_dir: Diretório de origem.
        days_back: Dias para trás.
        output_dir: Diretório de destino.
        prefix: Prefixo do nome do arquivo.
        suffix: Extensão.
        filename_override: Nome personalizado no destino (None = manter original).

    Returns:
        ``Path`` do arquivo copiado ou ``None`` se não encontrado.
    """
    source = find_file_for_days_back(source_dir, days_back, prefix=prefix, suffix=suffix)
    if source is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    if filename_override:
        dest = output_dir / filename_override
    else:
        dest = output_dir / source.name

    shutil.copy2(source, dest)
    return dest


def extract_all_days(
    source_dir: Path,
    output_dir: Path,
    *,
    start_days_back: int = 1,
    end_days_back: int = 30,
    prefix: str = "araucaria_cis",
    suffix: str = ".csv",
) -> dict[int, Path | None]:
    """Extrai múltiplos dias em um intervalo.

    Returns:
        Dict mapeando ``days_back`` → ``Path`` ou ``None``.
    """
    results: dict[int, Path | None] = {}
    for db in range(start_days_back, end_days_back + 1):
        result = extract_file_by_days_back(
            source_dir, db, output_dir, prefix=prefix, suffix=suffix,
        )
        results[db] = result
    return results
