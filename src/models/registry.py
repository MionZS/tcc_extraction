"""Model Registry and Governance for Versioned Machine Learning Models.

Provides centralized tracking, persistence, metadata auditing, and comparison
of trained models across training iterations, date ranges, and feature configurations.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class ModelMetadata:
    """Metadata describing a trained model and its data provenance."""

    version_id: str
    model_name: str
    algorithm: str
    created_at: str
    data_slice: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    tag: str = "candidate"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            version_id=data["version_id"],
            model_name=data["model_name"],
            algorithm=data.get("algorithm", "Unknown"),
            created_at=data.get("created_at", ""),
            data_slice=data.get("data_slice", {}),
            hyperparameters=data.get("hyperparameters", {}),
            feature_names=data.get("feature_names", []),
            tag=data.get("tag", "candidate"),
            description=data.get("description", ""),
        )


class ModelRegistry:
    """Manages versioned persistence and comparison of ML models."""

    def __init__(self, registry_dir: Path | str = "output/models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "registry_index.json"
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_file.exists():
            return []
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_index(self, records: list[dict[str, Any]]) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def register_model(
        self,
        model_name: str,
        model_obj: Any,
        metrics: dict[str, Any],
        data_slice: dict[str, Any] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        tag: str = "candidate",
        description: str = "",
        custom_version: str | None = None,
    ) -> str:
        """Register and persist a trained model with full provenance and metrics.

        Args:
            model_name: Identifier of model class (e.g. 'voltage_disturbance_classifier').
            model_obj: Fitted model object (must be serializable via joblib).
            metrics: Evaluation metrics dictionary (accuracy, f1_macro, etc.).
            data_slice: Provenance info (start_date, end_date, feeder, total_samples, etc.).
            hyperparameters: Model configuration parameters.
            feature_names: List of input feature column names.
            tag: Stage alias ('candidate', 'production', 'baseline', 'archived').
            description: Free-text note about the run.
            custom_version: Optional explicit version string (defaults to timestamp-based).

        Returns:
            The generated version_id string.
        """
        now = datetime.now()
        version_id = custom_version or f"v_{now.strftime('%Y%m%d_%H%M%S')}"

        model_dir = self.registry_dir / model_name / version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save serialized model
        model_file = model_dir / "model.joblib"
        joblib.dump(model_obj, model_file)

        # 2. Metadata
        algorithm_name = getattr(model_obj, "__class__", type(model_obj)).__name__
        meta = ModelMetadata(
            version_id=version_id,
            model_name=model_name,
            algorithm=algorithm_name,
            created_at=now.isoformat(),
            data_slice=data_slice or {},
            hyperparameters=hyperparameters or {},
            feature_names=feature_names or [],
            tag=tag,
            description=description,
        )

        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

        # 3. Metrics
        with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 4. Update index
        index = self._read_index()
        # Remove any existing entry with same version_id
        index = [item for item in index if not (item.get("model_name") == model_name and item.get("version_id") == version_id)]

        entry = {
            "version_id": version_id,
            "model_name": model_name,
            "algorithm": algorithm_name,
            "created_at": now.isoformat(),
            "tag": tag,
            "description": description,
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "start_date": (data_slice or {}).get("start_date"),
            "end_date": (data_slice or {}).get("end_date"),
            "feeder": (data_slice or {}).get("feeder"),
            "train_samples": (data_slice or {}).get("train_samples"),
            "test_samples": (data_slice or {}).get("test_samples"),
            "model_path": str(model_file.relative_to(self.registry_dir.parent.parent)),
        }
        index.append(entry)
        index.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        self._write_index(index)

        return version_id

    def list_models(self, model_name: str | None = None) -> list[dict[str, Any]]:
        """List all registered models, optionally filtered by model_name."""
        index = self._read_index()
        if model_name:
            return [m for m in index if m.get("model_name") == model_name]
        return index

    def load_model(self, model_name: str, version_id: str = "latest") -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Load model object, metadata, and metrics by version_id.

        Args:
            model_name: Model identifier.
            version_id: Specific version string, or 'latest'.

        Returns:
            Tuple of (model_obj, metadata_dict, metrics_dict).
        """
        models = self.list_models(model_name)
        if not models:
            raise FileNotFoundError(f"No models registered under name '{model_name}'.")

        if version_id == "latest":
            target_entry = models[0]
            version_id = target_entry["version_id"]
        else:
            match = [m for m in models if m["version_id"] == version_id]
            if not match:
                raise FileNotFoundError(f"Version '{version_id}' not found for model '{model_name}'.")
            target_entry = match[0]

        model_dir = self.registry_dir / model_name / version_id
        model_file = model_dir / "model.joblib"
        meta_file = model_dir / "metadata.json"
        metrics_file = model_dir / "metrics.json"

        model_obj = joblib.load(model_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        return model_obj, meta, metrics

    def compare_models(self, model_name: str, version_a: str, version_b: str) -> dict[str, Any]:
        """Generate a comparison dictionary between two model versions."""
        _, meta_a, metrics_a = self.load_model(model_name, version_a)
        _, meta_b, metrics_b = self.load_model(model_name, version_b)

        acc_a = metrics_a.get("accuracy", 0.0)
        acc_b = metrics_b.get("accuracy", 0.0)
        f1_a = metrics_a.get("f1_macro", 0.0)
        f1_b = metrics_b.get("f1_macro", 0.0)

        return {
            "model_name": model_name,
            "version_a": {
                "version_id": version_a,
                "created_at": meta_a.get("created_at"),
                "data_slice": meta_a.get("data_slice", {}),
                "accuracy": acc_a,
                "f1_macro": f1_a,
                "f1_weighted": metrics_a.get("f1_weighted", 0.0),
                "report": metrics_a.get("report", {}),
            },
            "version_b": {
                "version_id": version_b,
                "created_at": meta_b.get("created_at"),
                "data_slice": meta_b.get("data_slice", {}),
                "accuracy": acc_b,
                "f1_macro": f1_b,
                "f1_weighted": metrics_b.get("f1_weighted", 0.0),
                "report": metrics_b.get("report", {}),
            },
            "deltas": {
                "accuracy_delta": acc_b - acc_a,
                "f1_macro_delta": f1_b - f1_a,
            },
        }

    def render_registry_table(self, model_name: str | None = None) -> None:
        """Render a rich table displaying all registered models."""
        models = self.list_models(model_name)
        if not models:
            console.print("[yellow]Nenhum modelo registrado no repositório.[/yellow]")
            return

        table = Table(title="🏛️ Repositório de Modelos Treinados (Model Registry)", border_style="cyan")
        table.add_column("Versão", style="bold yellow")
        table.add_column("Modelo", style="dim")
        table.add_column("Tag", style="bold")
        table.add_column("Período dos Dados", style="cyan")
        table.add_column("Alimentador", style="magenta")
        table.add_column("Amostras", justify="right", style="green")
        table.add_column("Acurácia", justify="right", style="bold white")
        table.add_column("F1 Macro", justify="right", style="bold green")
        table.add_column("Data Criação", style="dim")

        for m in models:
            v_id = m.get("version_id", "")
            m_name = m.get("model_name", "")
            tag = m.get("tag", "candidate")
            tag_styled = f"[green]{tag}[/]" if tag == "production" else f"[blue]{tag}[/]"

            s_date = m.get("start_date") or "-"
            e_date = m.get("end_date") or "-"
            period_str = f"{s_date} → {e_date}" if s_date != "-" else "Todos"

            feeder_str = m.get("feeder") or "Todos"
            train_cnt = m.get("train_samples") or 0
            test_cnt = m.get("test_samples") or 0
            samples_str = f"{train_cnt + test_cnt:,}" if (train_cnt or test_cnt) else "-"

            acc = m.get("accuracy")
            acc_str = f"{acc:.2%}" if acc is not None else "-"

            f1 = m.get("f1_macro")
            f1_str = f"{f1:.4f}" if f1 is not None else "-"

            dt_created = m.get("created_at", "")[:19].replace("T", " ")

            table.add_row(v_id, m_name, tag_styled, period_str, feeder_str, samples_str, acc_str, f1_str, dt_created)

        console.print(table)
