"""run_manifest.py — JSON de rastreabilidade por execução do pipeline.

Inspiração: Fluxo_BI — run logs e metadata.
Adaptação: manifest simples para execução single-shot da amostra.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "output" / "runs"


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    report_day: str
    sample_size: int
    steps: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    finished_at: str | None = None
    duration_seconds: float | None = None

    def record_step(self, name: str, **kwargs) -> None:
        """Registra o resultado de um passo do pipeline."""
        self.steps[name] = {k: v for k, v in kwargs.items() if v is not None}

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def finalize(self) -> None:
        """Finaliza o manifest com timestamp e duração."""
        self.finished_at = datetime.now(timezone.utc).isoformat()
        start = datetime.fromisoformat(self.started_at)
        self.duration_seconds = round(
            (datetime.now(timezone.utc) - start).total_seconds(), 1
        )

    def save(self, output_dir: Path | None = None) -> Path:
        """Salva o manifest como JSON no diretório de runs."""
        save_dir = output_dir or DEFAULT_RUNS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"run_{self.run_id}.json"
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def create_manifest(report_day: date, sample_size: int) -> RunManifest:
    """Cria um novo manifest para uma execução."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RunManifest(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        report_day=report_day.isoformat(),
        sample_size=sample_size,
    )
