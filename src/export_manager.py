"""export_manager.py — Cópia do CSV final para target OneDrive.

Inspiração: Fluxo_BI incremental_publish.py — _copy_file_to_baseline().
Adaptação: cópia direta + .meta.json (sem shadow, dado o porte da amostra).
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# Target OneDrive sincronizado
DEFAULT_PUBLISH_TARGET = Path(r"E:\mion\OneDrive - copel.com\transfer_area\machine_learning_pipeline\input")


@dataclass
class PublishResult:
    source: Path
    target: Path
    bytes_copied: int
    meta_path: Path


def _count_csv_rows(csv_path: Path) -> int:
    """Conta linhas de dados de um CSV (sem cabeçalho)."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader, None)  # pular cabeçalho
        return sum(1 for _ in reader)


def publish_to_target(
    csv_source: Path,
    target_dir: Path,
    *,
    filename_override: str | None = None,
) -> PublishResult:
    """Copia CSV para o diretório target e cria .meta.json ao lado.

    Args:
        csv_source: Caminho do CSV gerado pelo pipeline.
        target_dir: Diretório de destino (OneDrive).
        filename_override: Nome do arquivo no target (default: mesmo nome da origem).

    Returns:
        PublishResult com detalhes da cópia.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = filename_override or csv_source.name
    target_path = target_dir / filename

    # Backup do arquivo existente (se houver)
    if target_path.exists():
        backup_name = f"{target_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_path.suffix}"
        backup_path = target_path.parent / "backups" / backup_name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, backup_path)

    # Copiar
    shutil.copy2(csv_source, target_path)
    bytes_copied = target_path.stat().st_size

    # Criar .meta.json
    meta = {
        "source": str(csv_source.resolve()),
        "target": str(target_path.resolve()),
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "row_count": _count_csv_rows(target_path),
        "file_size_bytes": bytes_copied,
    }
    meta_path = target_path.with_suffix(target_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return PublishResult(
        source=csv_source,
        target=target_path,
        bytes_copied=bytes_copied,
        meta_path=meta_path,
    )


def print_publish_report(result: PublishResult) -> None:
    """Imprime relatório de publicação formatado."""
    print("\n  Publish:")
    print(f"    Source : {result.source}")
    print(f"    Target : {result.target}")
    print(f"    Size   : {result.bytes_copied:,} bytes")
    print(f"    Meta   : {result.meta_path}")
