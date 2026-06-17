"""file_size_analysis.py — Análise de tamanhos de arquivos diários.

Calcula média, desvio padrão e detecta outliers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.checks.daily_file_naming import extract_date_from_filename


@dataclass(frozen=True)
class SizeStats:
    """Estatísticas de tamanho de arquivos."""
    count: int
    mean_bytes: float
    std_bytes: float
    min_bytes: int
    max_bytes: int
    min_date: date | None
    max_date: date | None
    median_bytes: float


@dataclass(frozen=True)
class FileSizeRecord:
    """Registro de tamanho de um arquivo."""
    path: Path
    file_date: date | None
    size_bytes: int


def collect_file_sizes(
    directory: Path,
    *,
    suffix: str = ".csv",
    prefix: str | None = None,
) -> list[FileSizeRecord]:
    """Coleta tamanhos de arquivos no diretório.

    Args:
        directory: Diretório a inspecionar.
        suffix: Extensão dos arquivos.
        prefix: Filtro opcional pelo início do nome.
    """
    records: list[FileSizeRecord] = []
    pattern = f"*{suffix}"
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        file_date = extract_date_from_filename(path.name)
        records.append(FileSizeRecord(
            path=path,
            file_date=file_date,
            size_bytes=path.stat().st_size,
        ))
    return records


def compute_size_stats(records: list[FileSizeRecord]) -> SizeStats | None:
    """Calcula estatísticas de tamanho a partir dos registros.

    Returns:
        ``SizeStats`` ou ``None`` se não houver registros.
    """
    if not records:
        return None

    sizes = [r.size_bytes for r in records]
    dated = [(r.file_date, r.size_bytes) for r in records if r.file_date is not None]

    mean = statistics.mean(sizes)
    std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0

    min_date = min(dated, key=lambda x: x[1])[0] if dated else None
    max_date = max(dated, key=lambda x: x[1])[0] if dated else None

    return SizeStats(
        count=len(sizes),
        mean_bytes=mean,
        std_bytes=std,
        min_bytes=min(sizes),
        max_bytes=max(sizes),
        min_date=min_date,
        max_date=max_date,
        median_bytes=statistics.median(sizes),
    )


def detect_size_outliers(
    records: list[FileSizeRecord],
    *,
    threshold: float = 1.5,
) -> list[FileSizeRecord]:
    """Detecta arquivos com tamanho fora do esperado.

    Um outlier é definido como arquivo cujo tamanho está a mais de
    ``threshold`` desvios-padrão da média.

    Args:
        threshold: Número de desvios-padrão para considerar outlier.
    """
    stats = compute_size_stats(records)
    if stats is None or stats.std_bytes == 0:
        return []

    upper = stats.mean_bytes + threshold * stats.std_bytes
    lower = max(0, stats.mean_bytes - threshold * stats.std_bytes)

    return [r for r in records if r.size_bytes > upper or r.size_bytes < lower]


def format_size(size_bytes: int) -> str:
    """Formata tamanho em bytes para formato legível."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
