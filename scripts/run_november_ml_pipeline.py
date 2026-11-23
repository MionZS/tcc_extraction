"""Legacy entrypoint delegated to frozen OOT_MULTIVIEW runner.

Eliminates prior OOT re-fitting, empty-intersection label bugs (0 candidates),
and schema truncation on `classified_ucs_latest.parquet`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.contracts import atomic_write_text, load_experiment_config
from src.experiments.oot_pipeline import run_oot_multiview


def main() -> int:
    cfg = load_experiment_config(PROJECT_ROOT / "configs" / "experiments" / "november_2026.yaml", pipeline_override="oot_multiview")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "experiments" / "runs" / f"{ts}_oot_multiview"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "config.yaml", cfg.raw_yaml)
    metrics = run_oot_multiview(cfg, run_dir)
    print(f"[+] OOT_MULTIVIEW validado em {run_dir}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
