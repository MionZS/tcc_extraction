"""Data Lake Continuity, Timeline & Governance Calendar Visualizations.

Provides rich terminal rendering of calendar grids, continuity timelines,
and gap analyses to audit temporal coverage across daily AMI partitions,
with full support for filtering and comparing file formats (.parquet vs .csv).
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def scan_datalake_dates(
    base_dir: Path | str = "output",
    file_type: str = "both",  # 'both', 'parquet', 'csv'
) -> dict[date, dict[str, Any]]:
    """Scan partitioned measurements and raw files to discover dates, row counts, and formats.

    Args:
        base_dir: Path to output directory.
        file_type: 'both' (presence of either or both), 'parquet' (requires parquet),
                   or 'csv' (requires csv).

    Returns:
        Dict mapping date -> dict with telemetry flags and format coverage info.
    """
    base = Path(base_dir)
    records: dict[date, dict[str, Any]] = {}

    meas_dir = base / "measurements"
    if meas_dir.exists():
        for m_type in ["ami_instantaneous", "ami_interval", "ami_registers"]:
            type_dir = meas_dir / m_type
            if not type_dir.exists():
                continue

            patterns = []
            if file_type in ("both", "parquet"):
                patterns.append("report_year=*/report_month=*/report_day=*/data.parquet")
            if file_type in ("both", "csv"):
                patterns.append("report_year=*/report_month=*/report_day=*/data.csv")

            for pat in patterns:
                for p in type_dir.glob(pat):
                    if p.stat().st_size <= 0:
                        continue
                    try:
                        parts = p.parts
                        y = int(next(part.split("=")[1] for part in parts if part.startswith("report_year=")))
                        m = int(next(part.split("=")[1] for part in parts if part.startswith("report_month=")))
                        d = int(next(part.split("=")[1] for part in parts if part.startswith("report_day=")))
                        dt = date(y, m, d)
                        if dt not in records:
                            records[dt] = {
                                "instantaneous": False,
                                "interval": False,
                                "registers": False,
                                "has_parquet": False,
                                "has_csv": False,
                                "size_bytes": 0,
                            }
                        key = m_type.replace("ami_", "")
                        records[dt][key] = True
                        if p.suffix == ".parquet":
                            records[dt]["has_parquet"] = True
                        elif p.suffix == ".csv":
                            records[dt]["has_csv"] = True
                        records[dt]["size_bytes"] += p.stat().st_size
                    except Exception:
                        continue

    # Also check if raw CIS exists
    raw_cis = base / "raw" / "CIS"
    if raw_cis.exists() and file_type in ("both", "csv"):
        for p in raw_cis.glob("araucaria_cis_*.csv"):
            stem = p.stem.replace("araucaria_cis_", "")
            if len(stem) == 8 and stem.isdigit():
                try:
                    dt = date(int(stem[:4]), int(stem[4:6]), int(stem[6:]))
                    if dt not in records:
                        records[dt] = {
                            "instantaneous": False,
                            "interval": False,
                            "registers": False,
                            "has_parquet": False,
                            "has_csv": True,
                            "size_bytes": p.stat().st_size,
                        }
                    else:
                        records[dt]["has_csv"] = True
                except Exception:
                    pass

    return records


def compute_continuity_segments(
    start_date: date,
    end_date: date,
    available_dates: set[date],
) -> list[dict[str, Any]]:
    """Partition a date range into contiguous present and missing segments."""
    if start_date > end_date:
        return []

    segments: list[dict[str, Any]] = []
    current_date = start_date
    current_status = current_date in available_dates
    seg_start = current_date

    while current_date <= end_date:
        status = current_date in available_dates
        if status != current_status:
            # Finalize previous segment
            seg_end = current_date - timedelta(days=1)
            days = (seg_end - seg_start).days + 1
            segments.append({
                "status": "PRESENTE" if current_status else "LACUNA",
                "start_date": seg_start,
                "end_date": seg_end,
                "days": days,
            })
            seg_start = current_date
            current_status = status
        current_date += timedelta(days=1)

    # Finalize last segment
    days = (end_date - seg_start).days + 1
    segments.append({
        "status": "PRESENTE" if current_status else "LACUNA",
        "start_date": seg_start,
        "end_date": end_date,
        "days": days,
    })

    return segments


def render_continuity_calendar(
    start_date: date = date(2026, 8, 1),
    end_date: date | None = None,
    base_dir: Path | str = "output",
    file_type: str = "both",  # 'both', 'parquet', 'csv'
) -> None:
    """Render a comprehensive Rich governance calendar and timeline in the terminal.

    Args:
        start_date: Initial date of target window.
        end_date: Final date of target window (defaults to yesterday).
        base_dir: Base datalake directory.
        file_type: 'both' (Geral / Ambos), 'parquet' (Apenas Parquet), 'csv' (Apenas CSV).
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    records = scan_datalake_dates(base_dir, file_type=file_type)

    # Determine eligibility based on file_type filter
    available_dates: set[date] = set()
    for d, meta in records.items():
        if file_type == "parquet" and meta.get("has_parquet"):
            available_dates.add(d)
        elif file_type == "csv" and meta.get("has_csv"):
            available_dates.add(d)
        elif file_type == "both" and (meta.get("has_parquet") or meta.get("has_csv")):
            available_dates.add(d)

    total_days = max(1, (end_date - start_date).days + 1)
    covered_days = sum(1 for d in available_dates if start_date <= d <= end_date)
    missing_days = total_days - covered_days
    pct_covered = (covered_days / total_days) * 100.0

    # Format-specific breakdown counts
    parquet_count = sum(1 for d, m in records.items() if start_date <= d <= end_date and m.get("has_parquet"))
    csv_count = sum(1 for d, m in records.items() if start_date <= d <= end_date and m.get("has_csv"))
    synced_count = sum(1 for d, m in records.items() if start_date <= d <= end_date and m.get("has_parquet") and m.get("has_csv"))

    # 1. Header Banner
    banner_color = "green" if pct_covered >= 90 else ("yellow" if pct_covered >= 50 else "red")
    banner_text = Text()
    banner_text.append("📅 AUDITORIA DE GOVERNANÇA & CONTINUIDADE TEMPORAL\n", style="bold white")
    banner_text.append(f"Período Alvo: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')} ({total_days} dias)\n", style="dim")
    banner_text.append(f"Filtro de Formato: [bold white on blue] {file_type.upper()} [/bold white on blue]  |  ", style="bold")
    banner_text.append(f"Parquet: {parquet_count} | CSV: {csv_count} | Ambos Sincronizados: {synced_count}\n", style="cyan")
    banner_text.append(f"Cobertura: {covered_days}/{total_days} ({pct_covered:.1f}%) | Dias Faltantes: {missing_days} | Status: ", style="bold")
    if pct_covered >= 90:
        banner_text.append("INTEGRO (Alta Continuidade)", style="bold green")
    elif pct_covered >= 50:
        banner_text.append("PARCIAL (Lacunas Moderadas)", style="bold yellow")
    else:
        banner_text.append("CRÍTICO (Requer Extração do Período)", style="bold red")

    console.print(Panel(banner_text, border_style=banner_color, padding=(1, 2)))

    # 2. Monthly Visual Calendar Grid
    months: list[tuple[int, int]] = []
    curr_m = start_date.replace(day=1)
    while curr_m <= end_date.replace(day=1):
        months.append((curr_m.year, curr_m.month))
        # advance month
        if curr_m.month == 12:
            curr_m = date(curr_m.year + 1, 1, 1)
        else:
            curr_m = date(curr_m.year, curr_m.month + 1, 1)

    month_names = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
        7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }

    grid_title = f"Visualização Mensal no Data Lake [Filtro: {file_type.upper()}] (Verde = Sincronizado, Ciano = Parcial, Vermelho = Faltante)"
    cal_table = Table(
        title=grid_title,
        border_style="cyan",
        header_style="bold magenta",
        show_lines=True,
    )
    for y, m in months:
        cal_table.add_column(f"{month_names[m]} {y}", justify="center")

    # Build calendar matrix for each month
    month_grids: list[list[str]] = []
    cal_obj = calendar.Calendar(firstweekday=0)  # Monday first

    max_weeks = 6
    for y, m in months:
        lines: list[str] = ["Seg Ter Qua Qui Sex Sab Dom"]
        weeks = cal_obj.monthdayscalendar(y, m)
        for w in weeks:
            row_tokens = []
            for d in w:
                if d == 0:
                    row_tokens.append("   ")
                else:
                    dt = date(y, m, d)
                    if dt < start_date or dt > end_date:
                        row_tokens.append(f"[dim]{d:02d} [/dim]")
                    elif dt in records:
                        m_meta = records[dt]
                        has_p = m_meta.get("has_parquet", False)
                        has_c = m_meta.get("has_csv", False)
                        if file_type == "both":
                            if has_p and has_c:
                                row_tokens.append(f"[bold white on green]{d:02d} [/bold white on green]")
                            elif has_p or has_c:
                                row_tokens.append(f"[bold black on cyan]{d:02d} [/bold black on cyan]")
                            else:
                                row_tokens.append(f"[bold white on red]{d:02d} [/bold white on red]")
                        elif file_type == "parquet":
                            if has_p:
                                row_tokens.append(f"[bold white on green]{d:02d} [/bold white on green]")
                            else:
                                row_tokens.append(f"[bold white on red]{d:02d} [/bold white on red]")
                        elif file_type == "csv":
                            if has_c:
                                row_tokens.append(f"[bold white on green]{d:02d} [/bold white on green]")
                            else:
                                row_tokens.append(f"[bold white on red]{d:02d} [/bold white on red]")
                    else:
                        row_tokens.append(f"[bold white on red]{d:02d} [/bold white on red]")
            lines.append(" ".join(row_tokens))
        while len(lines) < max_weeks + 1:
            lines.append("                            ")
        month_grids.append(lines)

    for line_idx in range(len(month_grids[0])):
        cal_table.add_row(*[month_grids[m_idx][line_idx] for m_idx in range(len(months))])

    console.print(cal_table)

    # 3. Continuity Segments Table
    segments = compute_continuity_segments(start_date, end_date, available_dates)

    seg_table = Table(title=f"Linha do Tempo de Continuidade (Segmentos & Quebras) [{file_type.upper()}]", border_style="blue")
    seg_table.add_column("#", justify="right", style="dim", width=4)
    seg_table.add_column("Status de Governança", justify="left")
    seg_table.add_column("Período do Segmento", justify="center", style="bold")
    seg_table.add_column("Duração", justify="right")
    seg_table.add_column("Formatos Encontrados", justify="left", style="cyan")

    for idx, s in enumerate(segments, 1):
        is_pres = s["status"] == "PRESENTE"
        status_styled = "[bold green]✔ PRESENTE (OK)[/bold green]" if is_pres else "[bold red]✘ LACUNA (QUEBRA)[/bold red]"
        period_str = f"{s['start_date'].strftime('%d/%m/%Y')} → {s['end_date'].strftime('%d/%m/%Y')}"
        duration_str = f"{s['days']} dia{'s' if s['days'] > 1 else ''}"

        if is_pres:
            sample_date = s["start_date"]
            meta = records.get(sample_date, {})
            fmt_tokens = []
            if meta.get("has_parquet"):
                fmt_tokens.append("Parquet")
            if meta.get("has_csv"):
                fmt_tokens.append("CSV")
            format_str = " + ".join(fmt_tokens) if fmt_tokens else "Disponível"
        else:
            format_str = "[red]Sem dados no Data Lake[/red]"

        seg_table.add_row(str(idx), status_styled, period_str, duration_str, format_str)

    console.print(seg_table)

    # 4. Discrepancy Alert / Recommendation
    if (parquet_count != csv_count or synced_count < covered_days) and file_type == "both":
        diff = abs(parquet_count - csv_count)
        console.print(
            Panel(
                f"[bold yellow]⚠️ DISCREPÂNCIA DE FORMATOS DETECTADA:[/bold yellow]\n\n"
                f"• Dias com Parquet: [bold green]{parquet_count}[/bold green]\n"
                f"• Dias com CSV:     [bold green]{csv_count}[/bold green]\n"
                f"• Dias com ambos:   [bold green]{synced_count}[/bold green]\n\n"
                f"Você pode utilizar a funcionalidade [bold cyan]Sincronizar Formatos (Converter Faltantes)[/bold cyan] "
                f"para preencher automaticamente os arquivos de um formato a partir do outro sem nova extração.",
                title="Sincronização de Formatos Recomendada",
                border_style="yellow",
            )
        )
