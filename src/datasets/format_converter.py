"""Bi-directional Parquet <-> CSV Synchronization & Format Converter.

Scans the Data Lake across all semantic layers (measurements, context, features,
and model inputs), identifies missing file formats (.parquet or .csv), and
converts between them to guarantee 100% format integrity and interoperability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()


def convert_parquet_to_csv(parquet_path: Path, csv_path: Path | None = None) -> Path:
    """Convert a Parquet file to CSV (semicolon separator)."""
    if csv_path is None:
        csv_path = parquet_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(parquet_path)
    df.write_csv(csv_path, separator=";")
    return csv_path


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path | None = None, compression: str = "zstd") -> Path:
    """Convert a CSV file to compressed Parquet."""
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(csv_path, separator=";", infer_schema_length=10000, ignore_errors=True)
    df.write_parquet(parquet_path, compression=compression)
    return parquet_path


def scan_format_discrepancies(
    base_dir: Path | str = "output",
    subdirs: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Scan the data lake to find partitions having Parquet but missing CSV, and vice versa.

    Returns:
        Dict with keys 'missing_csv' and 'missing_parquet', each containing file metadata.
    """
    base = Path(base_dir)
    if subdirs is None:
        subdirs = ["measurements", "context", "features", "model_input"]

    missing_csv: list[dict[str, Any]] = []
    missing_parquet: list[dict[str, Any]] = []

    for sub in subdirs:
        sub_path = base / sub
        if not sub_path.exists():
            continue

        # 1. Check all .parquet files
        for p in sub_path.rglob("*.parquet"):
            if p.is_file() and p.stat().st_size > 0:
                csv_peer = p.with_suffix(".csv")
                if not csv_peer.exists() or csv_peer.stat().st_size == 0:
                    missing_csv.append({
                        "source": p,
                        "target": csv_peer,
                        "layer": sub,
                        "rel_path": p.relative_to(base),
                        "size_bytes": p.stat().st_size,
                    })

        # 2. Check all .csv files
        for c in sub_path.rglob("*.csv"):
            if c.is_file() and c.stat().st_size > 0:
                p_peer = c.with_suffix(".parquet")
                if not p_peer.exists() or p_peer.stat().st_size == 0:
                    missing_parquet.append({
                        "source": c,
                        "target": p_peer,
                        "layer": sub,
                        "rel_path": c.relative_to(base),
                        "size_bytes": c.stat().st_size,
                    })

    return {
        "missing_csv": missing_csv,
        "missing_parquet": missing_parquet,
    }


def sync_datalake_formats(
    base_dir: Path | str = "output",
    direction: str = "both",  # 'both', 'to_csv', 'to_parquet'
    compression: str = "zstd",
) -> dict[str, int]:
    """Synchronize file formats across the data lake by converting missing files.

    Args:
        base_dir: Root directory of data lake.
        direction: 'both', 'to_csv' (Parquet -> CSV only), or 'to_parquet' (CSV -> Parquet only).
        compression: Parquet compression codec.

    Returns:
        Dict with counts of converted files.
    """
    discrepancies = scan_format_discrepancies(base_dir)
    to_csv_items = discrepancies["missing_csv"] if direction in ("both", "to_csv") else []
    to_parquet_items = discrepancies["missing_parquet"] if direction in ("both", "to_parquet") else []

    csv_converted = 0
    parquet_converted = 0
    errors: list[tuple[Path, str]] = []

    # Parquet -> CSV
    for item in to_csv_items:
        src: Path = item["source"]
        tgt: Path = item["target"]
        try:
            convert_parquet_to_csv(src, tgt)
            csv_converted += 1
        except Exception as exc:
            errors.append((src, str(exc)))

    # CSV -> Parquet
    for item in to_parquet_items:
        src: Path = item["source"]
        tgt: Path = item["target"]
        try:
            convert_csv_to_parquet(src, tgt, compression=compression)
            parquet_converted += 1
        except Exception as exc:
            errors.append((src, str(exc)))

    return {
        "parquet_to_csv": csv_converted,
        "csv_to_parquet": parquet_converted,
        "errors": len(errors),
    }


def render_format_audit_table(base_dir: Path | str = "output") -> None:
    """Render a Rich summary table comparing Parquet vs CSV availability across layers."""
    discrepancies = scan_format_discrepancies(base_dir)
    missing_csv = discrepancies["missing_csv"]
    missing_parquet = discrepancies["missing_parquet"]

    table = Table(
        title="Auditoria de Formatos do Data Lake (.parquet vs .csv)",
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Camada / Subdiretório", style="bold yellow")
    table.add_column("Faltam CSV (Parquet existe)", justify="right", style="cyan")
    table.add_column("Faltam Parquet (CSV existe)", justify="right", style="green")
    table.add_column("Status de Interoperabilidade", justify="center")

    layers = sorted(list({item["layer"] for item in missing_csv + missing_parquet}))
    if not layers:
        table.add_row("Todas as Camadas", "0", "0", "[bold green]✔ 100% Sincronizado (Ambos Formatos)[/bold green]")
        console.print(table)
        return

    for lyr in layers:
        n_csv = sum(1 for x in missing_csv if x["layer"] == lyr)
        n_p = sum(1 for x in missing_parquet if x["layer"] == lyr)
        status = "[bold green]Sincronizado[/bold green]" if n_csv == 0 and n_p == 0 else "[bold yellow]Pendente Conversão[/bold yellow]"
        table.add_row(lyr, f"{n_csv:,}", f"{n_p:,}", status)

    console.print(table)
