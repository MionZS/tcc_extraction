"""Central Reproducible Evidence Generator for the Copel AMI TCC.

Orchestrates 100% local, idempotent, checkpointed generation of all quantitative
audits, EDA figures, ML experiments, gating sensitivity, ablation, case studies,
operational/economic scenarios, and academic Markdown reports under `docs/tcc_evidence/`.

Usage:
    uv run python scripts/generate_tcc_evidence.py --mode sample --sample-size 3000 --seed 42
    uv run python scripts/generate_tcc_evidence.py --mode full --seed 42
    uv run python scripts/generate_tcc_evidence.py --mode full --resume
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import polars as pl
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tcc_evidence.audit_lake import (
    run_cockpit_reconciliation_audit,
    run_data_lake_audit,
    run_pipeline_engineering_benchmark,
)
from src.tcc_evidence.cases_ops_econ import run_case_studies, run_operations_and_economics
from src.tcc_evidence.eda_suite import run_eda_suite
from src.tcc_evidence.ml_experiments import (
    run_ablation_study,
    run_behavior_and_cpd_study,
    run_gating_sensitivity_study,
    run_ml_methods_and_fusion_audit,
)
from src.tcc_evidence.report_builder import build_all_tcc_reports


EVIDENCE_SUBDIRS = (
    "manifest",
    "pipeline",
    "data_quality",
    "eda",
    "ml_methods",
    "ablation",
    "gating",
    "behavior",
    "case_studies",
    "tui",
    "cockpit",
    "operations",
    "economics",
    "reports",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central TCC Evidence Generator (100% Local & Reproducible).")
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT / "output", help="Local Data Lake directory.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs" / "tcc_evidence", help="Evidence output directory.")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample", help="Sampled or full population for ML experiments.")
    parser.add_argument("--sample-size", type=int, default=3000, help="Sample size when --mode sample (0 = all).")
    parser.add_argument("--feeder", type=str, default="", help="Optional feeder filter.")
    parser.add_argument("--start-date", type=str, default="", help="Optional start cutoff date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="", help="Optional end cutoff date (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed stage checkpoints.")
    parser.add_argument("--no-censor", action="store_true", help="Disable LGPD UC masking (default: masked for academic export).")
    return parser.parse_args()


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tcc_evidence")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out or "unknown"
    except Exception:
        head_file = PROJECT_ROOT / ".git" / "HEAD"
        if head_file.exists():
            return head_file.read_text(encoding="utf-8").strip()[:40]
        return "unavailable"


def _load_classified_dataset(
    base_dir: Path,
    *,
    feeder: str = "",
    start_date: str = "",
    end_date: str = "",
) -> pl.DataFrame:
    pq_path = base_dir / "reports" / "classified_ucs_latest.parquet"
    csv_path = base_dir / "reports" / "classified_ucs_latest.csv"
    if pq_path.exists() and pq_path.stat().st_size > 0:
        df = pl.read_parquet(pq_path)
    elif csv_path.exists() and csv_path.stat().st_size > 0:
        df = pl.read_csv(csv_path, separator=";", infer_schema_length=10000, ignore_errors=True)
    else:
        raise FileNotFoundError(f"Dataset classificado não encontrado em {pq_path} ou {csv_path}.")

    if feeder and "meta__feeder" in df.columns:
        df = df.filter(pl.col("meta__feeder") == feeder)
    if "meta__cutoff_date" in df.columns:
        if start_date:
            df = df.filter(pl.col("meta__cutoff_date").cast(pl.String) >= start_date)
        if end_date:
            df = df.filter(pl.col("meta__cutoff_date").cast(pl.String) <= end_date)
    if df.is_empty():
        raise ValueError("Nenhum registro restante após aplicação dos filtros de alimentador/período.")
    return df


def main() -> int:
    args = parse_args()
    t_start = time.time()

    out_dir: Path = args.output_dir
    for sub in EVIDENCE_SUBDIRS:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(out_dir / "manifest" / "execution.log")
    censor = not args.no_censor
    effective_sample = 0 if args.mode == "full" else args.sample_size

    ckpt_path = out_dir / "manifest" / "checkpoints.json"
    partial_path = out_dir / "manifest" / "partial_results.json"

    completed: dict[str, bool] = {}
    results: dict[str, Any] = {}
    if args.resume and ckpt_path.exists():
        try:
            completed = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            completed = {}
    if args.resume and partial_path.exists():
        try:
            results = json.loads(partial_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    provenance = {
        "generated_at": datetime.now().isoformat(),
        "mode": args.mode,
        "sample_size": effective_sample,
        "feeder_filter": args.feeder or None,
        "start_date": args.start_date or None,
        "end_date": args.end_date or None,
        "seed": args.seed,
        "censor_uc_ids": censor,
        "git_commit": _get_git_commit(),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    results["provenance"] = provenance
    (out_dir / "manifest" / "provenance_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("=== INICIANDO GERADOR CENTRAL DE EVIDÊNCIAS DO TCC (100%% LOCAL) ===")
    classified_df = _load_classified_dataset(
        args.base_dir,
        feeder=args.feeder,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info(f"Dataset carregado: {classified_df.height:,} janelas, {classified_df.width} colunas.")

    def _save_ckpt(stage: str, payload: Any) -> None:
        completed[stage] = True
        results[stage] = payload
        ckpt_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        partial_path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # Stage 1: Data Lake Audit
    if not (args.resume and completed.get("data_quality")):
        logger.info("[1/9] Executando auditoria quantitativa do Data Lake...")
        dq_res = run_data_lake_audit(args.base_dir, out_dir, classified_df=classified_df)
        _save_ckpt("data_quality", dq_res)

    # Stage 2: Pipeline Engineering Benchmark
    if not (args.resume and completed.get("pipeline")):
        logger.info("[2/9] Executando benchmark controlado de engenharia (Polars Lazy/Sink vs Eager)...")
        pipe_res = run_pipeline_engineering_benchmark(args.base_dir, out_dir)
        _save_ckpt("pipeline", pipe_res)

    # Stage 3: Cockpit Reconciliation Audit
    if not (args.resume and completed.get("cockpit")):
        logger.info("[3/9] Executando reconciliação de granularidade e KPIs do Cockpit...")
        cockpit_res = run_cockpit_reconciliation_audit(classified_df, out_dir)
        _save_ckpt("cockpit", cockpit_res)

    # Stage 4: ML Experiments & Fusion Traceability
    scores_dict: dict[str, np.ndarray] | None = None
    if not (args.resume and completed.get("ml_methods")):
        logger.info("[4/9] Executando experimentos de ML (6 detectores, estabilidade e rastreabilidade)...")
        ml_res, scores_dict = run_ml_methods_and_fusion_audit(
            out_dir, classified_df=classified_df, sample_size=effective_sample, seed=args.seed, censor=censor
        )
        _save_ckpt("ml_methods", ml_res)

    # Stage 5: Real EDA Suite (12 Figures)
    if not (args.resume and completed.get("eda")):
        logger.info("[5/9] Gerando suíte completa de EDA real (12 figuras + CSVs + interpretações)...")
        eda_res = run_eda_suite(
            args.base_dir, out_dir, classified_df=classified_df, scores_dict=scores_dict, censor=censor
        )
        _save_ckpt("eda", eda_res)

    # Stage 6: Gating Sensitivity Study
    if not (args.resume and completed.get("gating")):
        logger.info("[6/9] Executando estudo de sensibilidade do gating (50%% a 90%%)...")
        gate_res = run_gating_sensitivity_study(
            out_dir, classified_df=classified_df, sample_size=effective_sample, seed=args.seed
        )
        _save_ckpt("gating", gate_res)

    # Stage 7: Behavioral Profiles & CUSUM CPD
    if not (args.resume and completed.get("behavior")):
        logger.info("[7/9] Executando estudo de perfis comportamentais e CPD/CUSUM...")
        beh_res = run_behavior_and_cpd_study(
            args.base_dir, out_dir, classified_df=classified_df, seed=args.seed, censor=censor
        )
        _save_ckpt("behavior", beh_res)

    # Stage 8: Ablation Study M0 -> M5
    if not (args.resume and completed.get("ablation")):
        logger.info("[8/9] Executando estudo de ablação (M0 -> M5) em coorte fixa...")
        abl_res = run_ablation_study(out_dir, classified_df=classified_df, seed=args.seed)
        _save_ckpt("ablation", abl_res)

    # Stage 9: Case Studies + Operations & Economics
    if not (args.resume and completed.get("case_studies")):
        logger.info("[9/9] Gerando estudos de caso mascarados e cenários operacionais/econômicos...")
        cs_res = run_case_studies(args.base_dir, out_dir, classified_df=classified_df, censor=censor)
        _save_ckpt("case_studies", cs_res)

    if not (args.resume and completed.get("operations_economics")):
        ops_econ_res = run_operations_and_economics(out_dir, classified_df=classified_df)
        _save_ckpt("operations_economics", ops_econ_res)

    # Final: Build the 6 mandatory Markdown deliverables
    logger.info("Compilando relatórios finais em Markdown...")
    written_reports = build_all_tcc_reports(out_dir, results)
    logger.info(f"Relatórios gerados: {len(written_reports)} arquivos em {out_dir}")
    logger.info(f"=== CONCLUÍDO EM {time.time() - t_start:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
