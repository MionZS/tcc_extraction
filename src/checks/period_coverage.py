"""period_coverage.py — Verificação de cobertura de período diário.

Verifica se o conjunto de arquivos cobre todo o período entre
a data mais antiga e a mais nova, sem lacunas.

Agora com suporte a cadência configurável por medidor (NIO) ou
por tipo de métrica, integrando com ``src.datasets.normalize``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from src.checks.daily_file_naming import extract_date_from_filename
from src.datasets.normalize import (
    compute_expected_points,
    compute_coverage,
)


# --- Configuração de cadência por NIO / métrica ---

# Cadência padrão (minutos) para cada tipo de dado
DEFAULT_CADENCES: dict[str, int] = {
    "interval": 5,        # meter_interval: 5 min → 288 pontos/dia
    "instantaneous": 5,   # meter_instantaneous: 5 min → 288 pontos/dia
    "register": 360,      # meter_register_snapshot: 6h → 4 pontos/dia
}

# Cadências específicas por medidor (NIO → tipo → minutos)
# Preenchido via arquivo de configuração ou banco.
PER_METER_CADENCES: dict[str, dict[str, int]] = {}


def load_cadence_config(path: Path | None = None) -> None:
    """Carrega cadências por medidor a partir de um CSV com colunas
    ``NIO``, ``metric_type``, ``cadence_minutes``.

    Args:
        path: Caminho para o CSV de configuração. Se None, mantém
              o dicionário vazio (usa cadências padrão).
    """
    global PER_METER_CADENCES
    if path is None:
        return

    df = pl.read_csv(path)
    for row in df.iter_rows(named=True):
        nio = str(row.get("NIO", "")).strip()
        mtype = str(row.get("metric_type", "")).strip()
        cad = int(row.get("cadence_minutes", 0))
        if nio and mtype and cad > 0:
            PER_METER_CADENCES.setdefault(nio, {})[mtype] = cad


def get_cadence(nio: str, metric_type: str = "interval") -> int:
    """Retorna a cadência configurada para um NIO e tipo de métrica.

    Falls back para a cadência padrão se não houver configuração
    específica para o medidor.

    Args:
        nio: Número de identificação do medidor.
        metric_type: ``interval``, ``instantaneous`` ou ``register``.

    Returns:
        Cadência em minutos.
    """
    meter_config = PER_METER_CADENCES.get(nio, {})
    return meter_config.get(metric_type, DEFAULT_CADENCES.get(metric_type, 5))


def get_expected_points(nio: str, metric_type: str = "interval") -> int:
    """Pontos esperados por dia para um NIO + tipo de métrica.

    Args:
        nio: Número de identificação do medidor.
        metric_type: ``interval``, ``instantaneous`` ou ``register``.

    Returns:
        Número de pontos esperados em um dia completo.
    """
    cadence = get_cadence(nio, metric_type)
    return compute_expected_points(cadence)


# --- Funções originais (estendidas) ---


def get_file_dates(
    directory: Path,
    *,
    suffix: str = ".csv",
    prefix: str | None = None,
) -> list[date]:
    """Coleta todas as datas presentes em nomes de arquivos.

    Suporta tanto ``.csv`` quanto ``.parquet``.

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
    dates: set[date] | list[date],
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


# --- Novas funções: cobertura temporal com cadência ---


def check_nio_coverage(
    df: pl.DataFrame,
    nio: str,
    *,
    timestamp_col: str = "TIMESTAMP_UTC",
    nio_col: str = "NIO",
    metric_type: str = "interval",
    report_day: date | None = None,
) -> dict:
    """Verifica a cobertura temporal de um único medidor.

    Examina quantos pontos existem no DataFrame para o NIO e compara
    com o esperado para a cadência configurada.

    Args:
        df: DataFrame com os dados do medidor.
        nio: NIO do medidor a verificar.
        timestamp_col: Nome da coluna de timestamp.
        nio_col: Nome da coluna de identificação do medidor.
        metric_type: Tipo de métrica (``interval``, ``instantaneous``,
                     ``register``).
        report_day: Data de referência. Se omitido, inferida dos dados.

    Returns:
        Dict com ``nio``, ``cadence_minutes``, ``expected_points``,
        ``actual_points``, ``coverage_percent``, ``complete``.
    """
    expected = get_expected_points(nio, metric_type)

    nio_df = df.filter(pl.col(nio_col) == nio)
    actual = nio_df.select(pl.len()).item() if nio_df.height > 0 else 0

    coverage = compute_coverage(actual, expected)

    return {
        "nio": nio,
        "cadence_minutes": get_cadence(nio, metric_type),
        "expected_points": expected,
        "actual_points": actual,
        "coverage_percent": round(coverage * 100, 1),
        "complete": actual >= expected,
    }


def check_batch_coverage(
    df: pl.DataFrame,
    *,
    nio_col: str = "NIO",
    timestamp_col: str = "TIMESTAMP_UTC",
    metric_type: str = "interval",
    min_coverage: float = 0.9,
) -> pl.DataFrame:
    """Verifica a cobertura temporal de todos os medidores em lote.

    Args:
        df: DataFrame com dados de múltiplos medidores.
        nio_col: Nome da coluna de NIO.
        timestamp_col: Nome da coluna de timestamp.
        metric_type: Tipo de métrica.
        min_coverage: Limiar mínimo de cobertura (0.0 a 1.0).

    Returns:
        DataFrame com colunas: NIO, cadence_minutes, expected_points,
        actual_points, coverage_percent, below_threshold.
    """
    expected_default = get_expected_points("", metric_type)

    coverage_rows: list[dict] = []
    for nio_val in df.get_column(nio_col).unique():
        nio_str = str(nio_val).strip()
        expected = get_expected_points(nio_str, metric_type)

        nio_df = df.filter(pl.col(nio_col) == nio_val)
        actual = nio_df.height
        coverage_pct = (actual / expected * 100) if expected > 0 else 0.0

        coverage_rows.append({
            "NIO": nio_str,
            "cadence_minutes": get_cadence(nio_str, metric_type),
            "expected_points": expected,
            "actual_points": actual,
            "coverage_percent": round(coverage_pct, 1),
            "below_threshold": coverage_pct < (min_coverage * 100),
        })

    return pl.DataFrame(coverage_rows, strict=False)
