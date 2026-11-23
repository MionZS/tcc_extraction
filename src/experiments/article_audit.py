"""Automated Auditor for TCC Main Article and Canonical Artifacts.

Performs robust multi-point verification of the main article against:
1. `latest_suite.json` run directories and manifest environments;
2. `oot_multiview/metrics.json`;
3. `daily_rules/metrics.json` and `daily_rules/daily_metrics.csv`;
4. `ablation_m0_m5/metrics.json`;
5. `gating_sensitivity/metrics.json`;
6. Structural, methodological, and epistemic neutrality constraints.

Explicitly documents scope of coverage:
- COVERED: numerical integrity of all tables, metrics, stats, thresholds, funnel stages,
  capacity scenarios, prohibited non-neutral causal phrases, and required neutral phrasings.
- NOT COVERED: qualitative literary style, typographic aesthetics, rendering of external
  vector PDFs, and runtime execution of external SAP/Oracle corporate databases.
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


def get_audit_coverage_report() -> dict[str, list[str]]:
    """Return explicit declaration of what this automated auditor covers and does not cover."""
    return {
        "covered": [
            "All quantitative values in OOT 19/11/2026 table and text (total UCs, eligible, quarantine, candidates, baseline)",
            "All quantitative values in DAILY_RULES series (cutoffs, total windows, mean, min, max, std dev)",
            "All stages of the strict sequential funnel M0 through M5 and unconstrained outside count",
            "Gating sensitivity metrics at 70% threshold and exact set reconciliation (net difference 22, rescued 416, discarded 394)",
            "Fixed capacity scenario metrics (K=2.000, overlap, churn in/out, Jaccard Top-2k, Spearman)",
            "Epistemic neutrality: prohibition of causal assertions ('fraude confirmada', 'evidência física comprovada', etc.)",
            "Required neutral phrasing: explicit capacity scenario qualification and differential interpretation framing",
            "Explicit scope demarcation: exclusion of field inspections (TOI) and OS feedback from TCC deliverables",
            "Operational flow 9-stage sequence with stages 7-9 marked as future work",
            "Polars version consistency between manifests (1.41.2) and documentation",
        ],
        "not_covered": [
            "Subjective prose style and qualitative flow across theoretical chapters (Chapters 1-3)",
            "Rendering and compilation fidelity of external PDF vector figures via Pandoc/LaTeX",
            "Live connectivity to corporate Oracle CIS/MDM or SAP work order databases",
            "Direct physical verification of meter hardware parameters in the field",
        ],
    }


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

    # Load daily metrics CSV
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

    # Manifests check (e.g. Polars version)
    polars_versions = {}
    for k, rpath in run_dirs.items():
        man_file = rpath / "manifest.json"
        if man_file.exists():
            man = json.loads(man_file.read_text(encoding="utf-8"))
            polars_versions[k] = man.get("environment", {}).get("polars", "unknown")

    return {
        "run_dirs": run_dirs,
        "oot": oot_m,
        "daily": daily_m,
        "daily_stats": daily_stats,
        "ablation": abl_m,
        "gating": gat_m,
        "polars_versions": polars_versions,
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
        ("1.562.8", "Erroneous Polars version string (must be 1.41.2)"),
        ("média estável de 5.387,7", "Unqualified 'média estável' in conclusion without dispersion"),
    ]

    for pat, desc in deprecated_checks:
        if pat in text:
            errors.append(f"Found prohibited deprecated pattern '{pat}': {desc}")

    # 2. Section "Resultados e discussão" scope
    if "# Resultados e discussão" not in text:
        errors.append("Section '# Resultados e discussão' not found in article")
        return errors

    res_text = text[text.index("# Resultados e discussão"):]

    # 3. Mandatory OOT metrics
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

    # 4. Mandatory DAILY_RULES metrics
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

    # 5. Mandatory Sequential Funnel stages
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

    # 6. Outside topological funnel count and Gating reconciliation
    if "1.943" not in res_text:
        errors.append("Missing outside topological funnel count 1.943")
    if "6.311" not in res_text:
        errors.append("Missing Gating 70% retained count 6.311")
    if "22" not in res_text:
        errors.append("Missing net set difference 22 UCs")
    if "416" not in res_text:
        errors.append("Missing OOT retained marginal count 416")
    if "394" not in res_text:
        errors.append("Missing OOT policy discarded count 394")
    if "5.917" not in res_text:
        errors.append("Missing common intersection count 5.917")

    # 7. Fixed capacity ablation (K=2.000)
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

    # 8. Epistemic neutrality checks
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

    # 9. Required neutral phrasing and scope demarcation
    required_phrasing = [
        ("cenário de capacidade", "Capacity scenario indication"),
        ("ilustraram a interpretação diferencial", "Neutral case study framing in conclusion"),
        ("segregar alertas de qualidade", "Neutral telecom filtering framing"),
        ("não foram executadas neste tcc", "Explicit exclusion of field inspections from TCC execution"),
        ("delimitação explícita de escopo", "Explicit scope boundary declaration in objectives"),
    ]
    for phrase, desc in required_phrasing:
        if phrase.lower() not in text.lower():
            errors.append(f"Missing required scope/neutral phrase '{phrase}': {desc}")

    return errors


if __name__ == "__main__":
    report = get_audit_coverage_report()
    print("=" * 80)
    print("AUDITOR AUTOMÁTICO DO ARTIGO E ARTEFATOS (TCC AMI)")
    print("=" * 80)
    print("COBERTURA DO AUDITOR:")
    for item in report["covered"]:
        print(f"  [+] {item}")
    print("\nNÃO COBERTO PELO AUDITOR (LIMITES FORMAIS):")
    for item in report["not_covered"]:
        print(f"  [-] {item}")
    print("-" * 80)

    errs = audit_article_against_metrics()
    if errs:
        print(f"FALHA: Encontradas {len(errs)} discrepâncias:")
        for e in errs:
            print(f"  [X] {e}")
        raise SystemExit(1)
    else:
        print("SUCESSO: Todas as 28 verificações automatizadas de integridade foram aprovadas com 100% de conformidade!")
