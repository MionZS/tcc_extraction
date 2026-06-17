"""period_coverage.py — Verificação de cobertura de período diário.

Verifica se o conjunto de arquivos cobre todo o período entre
a data mais antiga e a mais nova, sem lacunas.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.checks.daily_file_naming import extract_date_from_filename


def get_file_dates(
    directory: Path,
    *,
    suffix: str = ".csv",
    prefix: str | None = None,
) -> sorted[set[date]]:
    """Coleta todas as datas presentes em nomes de arquivos.

    Args:
        directory: Diretório a inspecionar.
        suffix: Extensão dos arquivos (``.csv`` ou ``.parquet``).
        prefix: Filtro opcional pelo início do nome.

    Returns:
        Conjunto ordenado de ``date`` extraídas dos nomes.
    """
    dates: set[date] = set()
    pattern = f"*{suffix}"
    for path in directory.glob(pattern):
        if prefix and not path.name.startswith(prefix):
            continue
        d = extract_date_from_filename(path.name)
        if d is not None:
            dates.add(d)
    return sorted(dates)


def generate_date_range(start_date: date, end_date: date) -> list[date]:
    """Gera todas as datas entre ``start_date`` e ``end_date`` (inclusive).

    Raises:
        ValueError: Se ``start_date > end_date``.
    """
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) não pode ser posterior a end_date ({end_date})"
        )
    days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(days)]


def check_period_coverage(
    dates: set[date] | sorted[set[date]],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    """Verifica se há lacunas no período coberto.

    Se ``start_date`` e ``end_date`` não forem informados, usa
    o menor e maior das datas fornecidas.

    Returns:
        Dict com ``missing_dates``, ``total_days``, ``covered_days``,
        ``coverage_percent``, ``complete``.
    """
    if not dates:
        return {
            "missing_dates": [],
            "total_days": 0,
            "covered_days": 0,
            "coverage_percent": 0.0,
            "complete": True,
            "start_date": None,
            "end_date": None,
        }

    date_list = sorted(dates)
    if start_date is None:
        start_date = date_list[0]
    if end_date is None:
        end_date = date_list[-1]

    all_dates = set(generate_date_range(start_date, end_date))
    missing = sorted(all_dates - set(dates))
    total = len(all_dates)
    covered = total - len(missing)
    pct = (covered / total * 100) if total > 0 else 0.0

    return {
        "missing_dates": [d.isoformat() for d in missing],
        "total_days": total,
        "covered_days": covered,
        "coverage_percent": round(pct, 1),
        "complete": len(missing) == 0,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
