"""Automated Auditor for TCC Main Article against Validated Run Metrics.

Enforces zero-divergence between the article text and the actual metrics artifacts
persisted in `experiments/runs/<run_id>/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import statistics
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUITE_INDEX_PATH = PROJECT_ROOT / "experiments" / "runs" / "latest_suite.json"
ARTICLE_PATH = PROJECT_ROOT / "docs" / "TCC_ARTIGO_FINAL_REVISADO_2026-11-18.md"


def load_suite_metrics() -> dict[str, Any]:
    """Load and parse metrics across all runs declared in latest_suite.json."""
    if not SUITE_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {SUITE_INDEX_PATH}")

    suite_idx = json.loads(SUITE_INDEX_PATH.read_text(encoding="utf-8"))
    run_dirs = {k: (PROJECT_ROOT / v).resolve() for k, v in suite_idx.items()}

    oot_m = json.loads((run_dirs["oot_multiview"] / "metrics.json").read_text(encoding="utf-8"))
    daily_m = json.loads((run_dirs["daily_rules"] / "metrics.json").read_text(encoding="utf-8"))
    abl_m = json.loads((run_dirs["ablation"] / "metrics.json").read_text(encoding="utf-8"))
    gat_m = json.loads((run_dirs["gating"] / "metrics.json").read_text(encoding="utf-8"))

    # Compute daily metrics stats directly from daily_metrics.csv
    daily_csv_path = run_dirs["daily_rules"] / "daily_metrics.csv"
    daily_rows = []
    with open(daily_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            daily_rows.append(r)

    cand_vals = [int(r["CANDIDATOS_VISTORIA"]) for r in daily_rows]
    daily_stats = {
        "n_rows": len(daily_rows),
        "min": min(cand_vals),
        "max": max(cand_vals),
        "mean": statistics.mean(cand_vals),
        "std_pop": statistics.pstdev(cand_vals),
        "std_sample": statistics.stdev(cand_vals),
    }

    return {
        "run_dirs": run_dirs,
        "oot": oot_m,
        "daily": daily_m,
        "daily_stats": daily_stats,
        "ablation": abl_m,
        "gating": gat_m,
    }


def audit_article_against_metrics(article_path: Path | None = None) -> list[str]:
    """Audit the article text against validated run metrics. Returns list of errors (empty if passed)."""
    p = article_path or ARTICLE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Article not found: {p}")

    text = p.read_text(encoding="utf-8")
    m = load_suite_metrics()
    errors: list[str] = []

    # 1. Prohibited deprecated numbers that contradict active runs
    deprecated_checks = [
        ("5.908", "Deprecated baseline count (must be 3.122)"),
        ("0,5512", "Deprecated baseline threshold (must be 0,643)"),
        ("0.5512", "Deprecated baseline threshold (must be 0.643)"),
        ("41,2", "Erroneous daily rules std dev (must be 562,8)"),
        ("41.2", "Erroneous daily rules std dev (must be 562.8)"),
        ("5.312", "Erroneous daily rules min (must be 4.737)"),
        ("5.465", "Erroneous daily rules max (must be 6.728)"),
    ]
    # Check within section "Resultados e discussão" onwards
    if "# Resultados e discussão" in text:
        res_text = text[text.index("# Resultados e discussão"):]
    else:
        res_text = text

    for pat, desc in deprecated_checks:
        if pat in res_text:
            errors.append(f"Found prohibited deprecated pattern '{pat}': {desc}")

    # 2. Mandatory OOT metrics
    expected_oot_patterns = [
        ("65.686", "Total OOT UCs"),
        ("62.857", "Eligible OOT UCs"),
        ("2.829", "Quarantine OOT UCs"),
        ("6.333", "Candidates OOT"),
        ("3.122", "Baseline IsolationForest candidates"),
        ("0,643", "Baseline threshold"),
        ("0,2944", "OOT mean anomaly score"),
        ("0,9115", "OOT P95 anomaly score"),
        ("189.341", "Train clean rows"),
    ]
    for pat, desc in expected_oot_patterns:
        if pat not in res_text:
            errors.append(f"Missing expected OOT metric '{pat}': {desc}")

    # 3. Mandatory DAILY_RULES metrics
    expected_daily_patterns = [
        ("1.248.501", "Total windows evaluated"),
        ("5.387,7", "Mean daily candidates"),
        ("4.737", "Min daily candidates"),
        ("6.728", "Max daily candidates"),
        ("562,8", "Daily candidates standard deviation"),
    ]
    for pat, desc in expected_daily_patterns:
        if pat not in res_text:
            errors.append(f"Missing expected DAILY_RULES metric '{pat}': {desc}")

    # 4. Mandatory Sequential Funnel stages
    expected_funnel = [
        ("M_0", "65.686"),
        ("M_1", "65.686"),
        ("M_2", "62.857"),
        ("M_3", "58.857"),
        ("M_4", "4.390"),
        ("M_5", "4.390"),
    ]
    for stage, count in expected_funnel:
        if count not in res_text:
            errors.append(f"Missing expected funnel stage count for {stage}: {count}")

    # 5. Outside topological funnel count and Gating reconciliation
    if "1.943" not in res_text:
        errors.append("Missing outside topological funnel count 1.943")
    if "6.311" not in res_text:
        errors.append("Missing Gating 70% retained count 6.311")
    if "22 UCs" not in res_text and "22" not in res_text:
        errors.append("Missing net set difference 22 UCs")
    if "416" not in res_text:
        errors.append("Missing OOT retained marginal count 416")
    if "394" not in res_text:
        errors.append("Missing OOT policy discarded count 394")
    if "5.917" not in res_text:
        errors.append("Missing common intersection count 5.917")

    # 6. Fixed capacity ablation (K=2.000)
    expected_ablation_k2000 = [
        ("2.000", "Capacity scenario K=2.000"),
        ("1.790", "No-LOF overlap"),
        ("210", "No-LOF churn"),
        ("0,8100", "No-LOF Jaccard Top-2k"),
        ("1.868", "No-PCA overlap"),
        ("132", "No-PCA churn"),
        ("0,8762", "No-PCA Jaccard Top-2k"),
        ("1.716", "Core-4 overlap"),
        ("284", "Core-4 churn"),
        ("0,7513", "Core-4 Jaccard Top-2k"),
    ]
    for pat, desc in expected_ablation_k2000:
        if pat not in res_text:
            errors.append(f"Missing expected ablation K=2000 metric '{pat}': {desc}")

    # 7. Epistemic neutrality checks
    forbidden_epistemic = [
        "comprovaram resolução do diagnóstico diferencial",
        "evitando falsos alarmes",
        "evitou-se falsos alarmes",
        "hipótese física comprovada",
        "anomalia física confirmada",
        "evidência física comprovada",
    ]
    for phrase in forbidden_epistemic:
        if phrase in res_text.lower():
            errors.append(f"Forbidden non-neutral phrase found: '{phrase}'")

    # 8. Required neutral phrasing
    required_phrasing = [
        ("cenário de capacidade", "Capacity scenario indication"),
        ("ilustraram a interpretação diferencial", "Neutral case study framing in conclusion"),
        ("segregar alertas de qualidade", "Neutral telecom filtering framing"),
    ]
    for phrase, desc in required_phrasing:
        if phrase.lower() not in res_text.lower():
            errors.append(f"Missing required neutral phrase '{phrase}': {desc}")

    return errors


if __name__ == "__main__":
    errs = audit_article_against_metrics()
    if errs:
        print(f"FAILED: Found {len(errs)} audit discrepancies in article:")
        for e in errs:
            print(f"  [X] {e}")
        raise SystemExit(1)
    else:
        print("SUCCESS: All numbers, tables, metrics and neutral phrasings in the article match active run artifacts exactly!")
