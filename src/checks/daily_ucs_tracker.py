"""daily_ucs_tracker.py — Rastreamento diário de UCs cadastradas.

Mantém ``output/raw/cis/daily_cadastrados/*.parquet`` com UCs únicas
por dia para referência rápida.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

DEFAULT_DAILY_UCS_DIR = Path(__file__).resolve().parent / "output" / "raw" / "CIS" / "daily_cadastrados"


def _validate_polars() -> None:
    if pl is None:  # pragma: no cover
        raise ImportError("polars é necessário. Instale com: uv add polars")


def save_daily_ucs(
    ucs: list[str],
    output_dir: Path,
    report_date: date,
) -> Path:
    """Salva UCs únicas em parquet com timestamp.

    Args:
        ucs: Lista de UCs únicas do dia.
        output_dir: Diretório de saída.
        report_date: Data de referência.

    Returns:
        ``Path`` do arquivo parquet criado.
    """
    _validate_polars()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ucs_{report_date.strftime('%Y%m%d')}.parquet"
    output_path = output_dir / filename

    df = pl.DataFrame({
        "ucs": ucs,
        "count": [len(ucs)] * len(ucs),
        "date": [report_date.isoformat()] * len(ucs),
        "created_at": [datetime.now(timezone.utc).isoformat()] * len(ucs),
    })

    df.write_parquet(output_path)
    return output_path


def load_daily_ucs(
    input_dir: Path,
    report_date: date,
) -> Any:
    """Carrega UCs para um dia específico.

    Returns:
        ``polars.DataFrame`` com colunas ``ucs``, ``count``, ``date``, ``created_at``.
    """
    _validate_polars()
    filename = f"ucs_{report_date.strftime('%Y%m%d')}.parquet"
    path = input_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de UCs não encontrado: {path}")

    return pl.read_parquet(path)


def get_ucs_list(
    input_dir: Path,
    report_date: date,
) -> list[str]:
    """Retorna lista de UCs para um dia específico.

    Returns:
        Lista de strings UC.
    """
    df = load_daily_ucs(input_dir, report_date)
    return df["ucs"].to_list()


def get_ucs_for_date_range(
    input_dir: Path,
    start_date: date,
    end_date: date,
) -> Any:
    """Retorna UCs para um intervalo de datas.

    Returns:
        ``polars.DataFrame`` com todas as UCs do período.
    """
    _validate_polars()
    frames = []
    current = start_date
    while current <= end_date:
        try:
            df = load_daily_ucs(input_dir, current)
            frames.append(df)
        except FileNotFoundError:
            pass
        current = date.fromordinal(current.toordinal() + 1)

    if not frames:
        return pl.DataFrame({"ucs": [], "date": [], "created_at": []})

    return pl.concat(frames)
