"""run_manifest.py — JSON de rastreabilidade por execução do pipeline.

Inspiração: Fluxo_BI — run logs e metadata.
Adaptação: manifest completo conforme docs/dataset_architecture_decision.md §9.

Cada execução deve gerar manifesto imutável com:
- run_id, pipeline_git_commit, config_hash, query_hash, schema_version
- source_window_start, source_window_end, extracted_at
- row_count por tabela
- min/max timestamp, unique_nio
- null_count crítico, duplicate_count, late_arrival_count
- file paths, file sizes, checksums
- status
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "output" / "runs"


@dataclass
class TableStats:
    """Statistics for a single dataset table in the manifest."""
    table_name: str
    row_count: int
    column_count: int
    file_path: str | None = None
    file_size_bytes: int | None = None
    checksum: str | None = None
    unique_nio: int | None = None
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    null_counts: dict[str, int] | None = None
    duplicate_count: int | None = None


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    report_day: str
    sample_size: int

    # Pipeline metadata (§9)
    pipeline_git_commit: str | None = None
    config_hash: str | None = None
    query_hash: str | None = None
    schema_version: str = "v1"
    source_window_start: str | None = None
    source_window_end: str | None = None
    extracted_at: str | None = None

    # Per-table statistics
    tables: list[dict[str, Any]] = field(default_factory=list)

    # Execution log
    steps: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # Finalization
    finished_at: str | None = None
    duration_seconds: float | None = None
    status: str = "running"  # running, completed, failed

    def record_step(self, name: str, **kwargs) -> None:
        """Registra o resultado de um passo do pipeline."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        self.steps[name] = clean

    def record_table(
        self,
        table_name: str,
        df: Any,
        *,
        file_path: str | None = None,
        file_size_bytes: int | None = None,
        checksum: str | None = None,
    ) -> None:
        """Record per-table statistics from a Polars DataFrame.

        Args:
            table_name: Name of the dataset table.
            df: Polars DataFrame (or any object with .height, .columns, etc.).
            file_path: Absolute path to the Parquet/CSV file.
            file_size_bytes: File size in bytes.
            checksum: SHA-256 checksum of the file.
        """
        import polars as pl

        stats: dict[str, Any] = {"table_name": table_name}

        if hasattr(df, "height"):
            stats["row_count"] = df.height
        if hasattr(df, "columns"):
            stats["column_count"] = len(df.columns)

        if file_path:
            stats["file_path"] = file_path
        if file_size_bytes is not None:
            stats["file_size_bytes"] = file_size_bytes
        if checksum:
            stats["checksum"] = checksum

        # NIO uniqueness
        nio_col = self._find_column(df, "NIO")
        if nio_col is not None:
            try:
                stats["unique_nio"] = df[nio_col].n_unique()
            except Exception:
                pass

        # Timestamp range
        for ts_col in ("TIMESTAMP_UTC", "REPORT_DAY", "TIMESTAMP_LOCAL"):
            col = self._find_column(df, ts_col)
            if col is not None:
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if min_val is not None:
                        stats[f"min_{ts_col.lower()}"] = str(min_val)
                    if max_val is not None:
                        stats[f"max_{ts_col.lower()}"] = str(max_val)
                except Exception:
                    pass

        # Null counts (only for critical columns)
        critical_cols = ["NIO", "REPORT_DAY", "TIMESTAMP_UTC"]
        nulls: dict[str, int] = {}
        for c in critical_cols:
            col = self._find_column(df, c)
            if col is not None:
                try:
                    nc = df[col].null_count()
                    if nc > 0:
                        nulls[c] = nc
                except Exception:
                    pass
        if nulls:
            stats["null_counts"] = nulls

        # Duplicate check by grain
        keys = self._infer_keys(table_name)
        if keys and hasattr(df, "unique"):
            try:
                valid_keys = [k for k in keys if self._find_column(df, k) is not None]
                if valid_keys:
                    dup_count = df.height - df.unique(subset=valid_keys).height
                    if dup_count > 0:
                        stats["duplicate_count"] = dup_count
            except Exception:
                pass

        self.tables.append(stats)

    def record_table_stats(self, table_stats: TableStats) -> None:
        """Record a pre-built TableStats dataclass."""
        self.tables.append(asdict(table_stats))

    def _find_column(self, df: Any, name: str) -> str | None:
        """Find a column by name ignoring case."""
        if not hasattr(df, "columns"):
            return None
        for col in df.columns:
            if str(col).strip().upper() == name.upper():
                return str(col)
        return None

    def _infer_keys(self, table_name: str) -> tuple[str, ...]:
        """Infer candidate keys for a table name."""
        key_map: dict[str, tuple[str, ...]] = {
            "meter_day_metadata": ("REPORT_DAY", "NIO"),
            "meter_interval": ("NIO", "TIMESTAMP_UTC"),
            "meter_instantaneous": ("NIO", "TIMESTAMP_UTC"),
            "meter_register_snapshot": ("NIO", "TIMESTAMP_UTC"),
            "meter_day_features": ("NIO", "REPORT_DAY", "FEATURE_SET_VERSION"),
            "labels": ("ENTITY_TYPE", "ENTITY_ID", "REFERENCE_START",
                       "LABEL_TYPE", "LABEL_VERSION"),
        }
        return key_map.get(table_name, ())

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        if self.status == "running":
            self.status = "failed"

    def finalize(self) -> None:
        """Finaliza o manifest com timestamp, duração e status."""
        self.finished_at = datetime.now(timezone.utc).isoformat()
        start = datetime.fromisoformat(self.started_at)
        self.duration_seconds = round(
            (datetime.now(timezone.utc) - start).total_seconds(), 1
        )
        if self.status == "running":
            self.status = "completed" if not self.errors else "failed"

    def save(self, output_dir: Path | None = None) -> Path:
        """Salva o manifest como JSON no diretório de runs."""
        save_dir = output_dir or DEFAULT_RUNS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"run_{self.run_id}.json"
        data = asdict(self)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path


def create_manifest(
    report_day: date,
    sample_size: int = 0,
    *,
    config_hash: str | None = None,
    query_hash: str | None = None,
    schema_version: str = "v1",
) -> RunManifest:
    """Cria um novo manifest para uma execução do pipeline.

    Args:
        report_day: Data de referência da extração.
        sample_size: Número de NIOs na amostra (0 = completo).
        config_hash: Hash SHA-256 do arquivo de configuração.
        query_hash: Hash SHA-256 dos arquivos de query.
        schema_version: Versão do schema dos datasets.

    Returns:
        RunManifest configurado com metadados iniciais.
    """
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")
    return RunManifest(
        run_id=run_id,
        started_at=now.isoformat(),
        report_day=report_day.isoformat(),
        sample_size=sample_size,
        config_hash=config_hash,
        query_hash=query_hash,
        schema_version=schema_version,
        extracted_at=now.isoformat(),
    )


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_config_hash(config_path: Path | None = None) -> str | None:
    """Compute hash of the config file, if it exists."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.json"
    if config_path.exists():
        return compute_file_hash(config_path)
    return None


def compute_query_hash(query_dir: Path | None = None) -> str | None:
    """Compute combined hash of all SQL query files."""
    if query_dir is None:
        query_dir = Path(__file__).resolve().parents[1] / "queries"
    if not query_dir.exists():
        return None

    combined = hashlib.sha256()
    for sql_file in sorted(query_dir.rglob("*.sql")):
        combined.update(sql_file.read_bytes())
    return combined.hexdigest()
