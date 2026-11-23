"""Unified Reproducible Experiment Runner for the Copel AMI TCC.

Usage examples:
    uv run python scripts/run_tcc_experiments.py audit --config configs/experiments/november_2026.yaml
    uv run python scripts/run_tcc_experiments.py daily-rules --config configs/experiments/november_2026.yaml
    uv run python scripts/run_tcc_experiments.py oot-multiview --config configs/experiments/november_2026.yaml
    uv run python scripts/run_tcc_experiments.py all --config configs/experiments/november_2026.yaml
    uv run python scripts/run_tcc_experiments.py compare-runs --baseline <RUN_ID_A> --candidate <RUN_ID_B>
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.experiments.case_auditor import run_case_traces, run_comparator
from src.experiments.contracts import atomic_write_json, atomic_write_text, load_experiment_config
from src.experiments.daily_rules import run_audit, run_daily_rules
from src.experiments.oot_pipeline import (
    run_ablation,
    run_gating_sensitivity,
    run_oot_multiview,
    run_score_stability,
)
from src.experiments.reporting import generate_all_deliverables

EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_ROOT / "runs"
COMPARISONS_DIR = EXPERIMENTS_ROOT / "comparisons"
SUITE_INDEX_PATH = RUNS_DIR / "latest_suite.json"


def _make_run_dir(pipeline_id: str, config_raw: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    run_dir = RUNS_DIR / f"{ts}_{pipeline_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    if config_raw:
        atomic_write_text(run_dir / "config.yaml", config_raw)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible TCC AMI Experimental Suite Runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in (
        "audit",
        "daily-rules",
        "oot-multiview",
        "ablation",
        "gating-sensitivity",
        "score-stability",
        "case-traces",
        "eda",
        "generate-report",
        "all",
    ):
        p = sub.add_parser(cmd)
        p.add_argument(
            "--config",
            type=Path,
            default=Path("configs/experiments/november_2026.yaml"),
            help="Path to experiment YAML configuration.",
        )

    cmp_p = sub.add_parser("compare-runs")
    cmp_p.add_argument("--baseline", type=str, required=True, help="Baseline RUN_ID under experiments/runs/")
    cmp_p.add_argument("--candidate", type=str, required=True, help="Candidate RUN_ID under experiments/runs/")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    if args.command == "compare-runs":
        res = run_comparator(RUNS_DIR / args.baseline, RUNS_DIR / args.candidate, COMPARISONS_DIR)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.command == "generate-report":
        if not SUITE_INDEX_PATH.exists():
            raise FileNotFoundError(f"Suite index not found at {SUITE_INDEX_PATH}. Run 'all' first.")
        suite_map = {k: Path(v) for k, v in json.loads(SUITE_INDEX_PATH.read_text(encoding="utf-8")).items()}
        written = generate_all_deliverables(suite_map)
        print(f"[+] Generated {len(written)} canonical deliverables.")
        return 0

    if args.command == "eda":
        proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "eda_city_profile_and_case_studies.py")])
        return int(proc.returncode)

    if args.command == "audit":
        cfg = load_experiment_config(args.config, pipeline_override="audit")
        rdir = _make_run_dir("audit", cfg.raw_yaml)
        res = run_audit(cfg, rdir)
        print(f"[+] audit validated in {rdir.name} (future_violations={res['future_violations']})")
        return 0

    if args.command == "daily-rules":
        cfg = load_experiment_config(args.config, pipeline_override="daily_rules")
        rdir = _make_run_dir("daily_rules", cfg.raw_yaml)
        res = run_daily_rules(cfg, rdir)
        print(f"[+] daily-rules validated in {rdir.name} (mean_candidates={res['mean_daily_candidates']})")
        return 0

    if args.command == "oot-multiview":
        cfg = load_experiment_config(args.config, pipeline_override="oot_multiview")
        rdir = _make_run_dir("oot_multiview", cfg.raw_yaml)
        res = run_oot_multiview(cfg, rdir)
        print(f"[+] oot-multiview validated in {rdir.name} (candidates={res['oot_candidates_vistoria']})")
        return 0

    if args.command == "ablation":
        cfg = load_experiment_config(args.config, pipeline_override="ablation_m0_m5")
        rdir = _make_run_dir("ablation_m0_m5", cfg.raw_yaml)
        res = run_ablation(cfg, rdir)
        print(f"[+] ablation validated in {rdir.name}: {res['set_reconciliation']}")
        return 0

    if args.command == "gating-sensitivity":
        cfg = load_experiment_config(args.config, pipeline_override="gating_sensitivity")
        rdir = _make_run_dir("gating_sensitivity", cfg.raw_yaml)
        run_gating_sensitivity(cfg, rdir)
        print(f"[+] gating-sensitivity validated in {rdir.name}")
        return 0

    if args.command == "score-stability":
        cfg = load_experiment_config(args.config, pipeline_override="score_stability")
        rdir = _make_run_dir("score_stability", cfg.raw_yaml)
        run_score_stability(cfg, rdir)
        print(f"[+] score-stability validated in {rdir.name}")
        return 0

    if args.command == "case-traces":
        cfg = load_experiment_config(args.config, pipeline_override="case_traces")
        rdir = _make_run_dir("case_traces", cfg.raw_yaml)
        run_case_traces(cfg, rdir)
        print(f"[+] case-traces validated in {rdir.name}")
        return 0

    if args.command == "all":
        print("=" * 80)
        print("EXECUTANDO SUITE EXPERIMENTAL REPRODUZIVEL COMPLETA (TCC AMI)")
        print("=" * 80)
        run_dirs: dict[str, Path] = {}

        # 1. Audit
        cfg_aud = load_experiment_config(args.config, pipeline_override="audit")
        run_dirs["audit"] = _make_run_dir("audit", cfg_aud.raw_yaml)
        run_audit(cfg_aud, run_dirs["audit"])
        print(f"[1/7] audit         -> {run_dirs['audit'].name}")

        # 2. DAILY_RULES
        cfg_day = load_experiment_config(args.config, pipeline_override="daily_rules")
        run_dirs["daily_rules"] = _make_run_dir("daily_rules", cfg_day.raw_yaml)
        daily_m = run_daily_rules(cfg_day, run_dirs["daily_rules"])
        print(f"[2/7] daily_rules   -> {run_dirs['daily_rules'].name} (mean_cand={daily_m['mean_daily_candidates']})")

        # 3. OOT_MULTIVIEW
        cfg_oot = load_experiment_config(args.config, pipeline_override="oot_multiview")
        run_dirs["oot_multiview"] = _make_run_dir("oot_multiview", cfg_oot.raw_yaml)
        oot_m = run_oot_multiview(cfg_oot, run_dirs["oot_multiview"])
        oot_dec = pl.read_parquet(run_dirs["oot_multiview"] / "decisions.parquet")
        print(f"[3/7] oot_multiview -> {run_dirs['oot_multiview'].name} (oot_cand={oot_m['oot_candidates_vistoria']:,})")

        # 4. ABLATION_M0_M5
        cfg_abl = load_experiment_config(args.config, pipeline_override="ablation_m0_m5")
        run_dirs["ablation"] = _make_run_dir("ablation_m0_m5", cfg_abl.raw_yaml)
        abl_m = run_ablation(cfg_abl, run_dirs["ablation"], oot_decisions=oot_dec)
        print(f"[4/7] ablation      -> {run_dirs['ablation'].name} (M4={abl_m['set_reconciliation']['m4_count']}, M5={abl_m['set_reconciliation']['m5_sequential_count']})")

        # 5. Gating Sensitivity
        cfg_gat = load_experiment_config(args.config, pipeline_override="gating_sensitivity")
        run_dirs["gating"] = _make_run_dir("gating_sensitivity", cfg_gat.raw_yaml)
        run_gating_sensitivity(cfg_gat, run_dirs["gating"], oot_decisions=oot_dec)
        print(f"[5/7] gating        -> {run_dirs['gating'].name}")

        # 6. Score Stability & Subdetector Audit
        cfg_stb = load_experiment_config(args.config, pipeline_override="score_stability")
        run_dirs["score_stability"] = _make_run_dir("score_stability", cfg_stb.raw_yaml)
        run_score_stability(cfg_stb, run_dirs["score_stability"])
        print(f"[6/7] score_stab    -> {run_dirs['score_stability'].name}")

        # 7. Case Traces
        cfg_cas = load_experiment_config(args.config, pipeline_override="case_traces")
        run_dirs["case_traces"] = _make_run_dir("case_traces", cfg_cas.raw_yaml)
        run_case_traces(cfg_cas, run_dirs["case_traces"], oot_run_dir=run_dirs["oot_multiview"])
        print(f"[7/7] case_traces   -> {run_dirs['case_traces'].name}")

        # 8. Compare DAILY_RULES vs OOT_MULTIVIEW
        cmp_res = run_comparator(run_dirs["daily_rules"], run_dirs["oot_multiview"], COMPARISONS_DIR)
        print(f"[+] Comparação salva: Jaccard={cmp_res['jaccard_candidates']} (comuns={cmp_res['candidates_common']})")

        # 9. Save latest suite index (relative portable POSIX paths) & generate all 15 deliverables
        suite_idx = {
            k: (v.relative_to(PROJECT_ROOT).as_posix() if v.is_relative_to(PROJECT_ROOT) else str(v).replace("\\", "/"))
            for k, v in run_dirs.items()
        }
        atomic_write_json(SUITE_INDEX_PATH, suite_idx)
        written = generate_all_deliverables(run_dirs)
        print(f"[+] {len(written)} documentos canônicos gerados a partir de runs validadas em {time.time() - t0:.1f}s!")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
