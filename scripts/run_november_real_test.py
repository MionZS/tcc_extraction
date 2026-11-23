"""Legacy entrypoint delegated to the reproducible DAILY_RULES & OOT runner.

Eliminates prior hardcoded strings (412, 1.226.702, 93 dias) and `or True` checks
by executing the fail-closed `run_daily_rules` pipeline from `src.experiments`.
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
from src.experiments.daily_rules import run_daily_rules


def main() -> int:
    cfg = load_experiment_config(PROJECT_ROOT / "configs" / "experiments" / "november_2026.yaml", pipeline_override="daily_rules")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "experiments" / "runs" / f"{ts}_daily_rules"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "config.yaml", cfg.raw_yaml)
    metrics = run_daily_rules(cfg, run_dir)
    print(f"[+] DAILY_RULES validado em {run_dir}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
