"""Real Masked Case Studies (UC X-Ray), Operational Capacity & Economic Sensitivity Simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.eda.cockpit_metrics import deduplicate_latest_uc
from src.eda.uc_timeseries import load_uc_daily_timeseries, resolve_uc_to_nios
from src.utils.censorship import censor_uc_id


def run_case_studies(
    base_dir: Path,
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
    censor: bool = True,
) -> dict[str, Any]:
    """Build auditable masked X-Ray dossiers across 8 representative real UC phenomena."""
    cs_dir = out_dir / "case_studies"
    cs_dir.mkdir(parents=True, exist_ok=True)

    uniq_df = deduplicate_latest_uc(classified_df)
    if uniq_df.is_empty():
        return {"cases_count": 0, "cases": []}

    def _pick_first(cond: pl.Expr, sort_col: str = "ANOMALY_SCORE", desc: bool = True) -> dict[str, Any] | None:
        sub = uniq_df.filter(cond)
        if sub.is_empty():
            return None
        if sort_col in sub.columns:
            sub = sub.sort(sort_col, descending=desc)
        return sub.head(1).to_dicts()[0]

    selections = [
        ("Caso 1: Operação Regular", pl.col("CATEGORIA_DIAGNOSTICA") == "REGULAR", "ANOMALY_SCORE", False),
        ("Caso 2: Anomalia Multivariada Global", pl.col("CATEGORIA_DIAGNOSTICA") == "ANOMALIA_MULTIVARIADA", "ANOMALY_SCORE", True),
        ("Caso 3: Desequilíbrio / Possível Bypass", pl.col("CATEGORIA_DIAGNOSTICA") == "DESEQUILIBRIO_CARGA_OU_BYPASS", "ANOMALY_SCORE", True),
        ("Caso 4: Possível GD Não Cadastrada (Fluxo Reverso)", pl.col("CATEGORIA_DIAGNOSTICA") == "POSSIVEL_GD_NAO_CADASTRADA", "DIAS_ANOMALIA_SEGUIDOS", True),
        ("Caso 5: Degrau Descendente / Queda Sustentada", pl.col("CATEGORIA_DIAGNOSTICA") == "DEGRAU_DESCENDENTE_CONSUMO", "ANOMALY_SCORE", True),
        ("Caso 6: Medidor Travado / Consumo Zero", pl.col("CATEGORIA_DIAGNOSTICA") == "MEDIDOR_TRAVADO_OU_CONSUMO_ZERO", "ANOMALY_SCORE", True),
        ("Caso 7: Falha / Quarentena de Telecomunicação", pl.col("CATEGORIA_DIAGNOSTICA") == "TELECOM_QUARANTINE", "x__avg_interval_coverage", False),
        ("Caso 8: Alta Dispersão com Consumo Médio Normal (116781157)", pl.col("id__uc_id").cast(pl.String) == "116781157", "ANOMALY_SCORE", True),
    ]

    dossiers: list[dict[str, Any]] = []
    seen_ucs: set[str] = set()

    for title, cond, scol, desc in selections:
        try:
            row = _pick_first(cond, scol, desc)
        except Exception:
            row = None
        if row is None:
            continue
        raw_uc = str(row.get("id__uc_id", ""))
        if not raw_uc or raw_uc in seen_ucs:
            continue
        seen_ucs.add(raw_uc)

        disp_uc = censor_uc_id(raw_uc) if censor else raw_uc
        nios = resolve_uc_to_nios(raw_uc, base_dir=base_dir)
        ts = load_uc_daily_timeseries(raw_uc, base_dir=base_dir, days=30)

        # Plot UC daily curve if points exist
        plot_file = None
        if ts.get("status") == "ok" and ts.get("points"):
            pts = ts["points"]
            vals = [float(p["value"]) for p in pts]
            med = float(ts.get("baseline_median", np.median(vals)))
            fig, ax = plt.subplots(figsize=(8.5, 3.6))
            ax.plot(range(len(vals)), vals, marker="o", linewidth=1.8, color="#0f172a", label="Consumo Diário (kWh)")
            ax.axhline(med, color="#2563eb", linestyle="--", label=f"Mediana ({med:.2f} kWh)")
            if ts.get("step_change"):
                si = int(ts["step_change"].get("index", 0))
                ax.axvline(si, color="#dc2626", linestyle=":", linewidth=2.0, label="Quebra CUSUM")
            ax.set_title(f"{title} — UC {disp_uc}", fontweight="bold")
            ax.set_xlabel("Dias na Janela")
            ax.set_ylabel("kWh/dia")
            ax.legend(fontsize=8)
            plt.tight_layout()
            slug = f"case_{len(dossiers)+1:02d}_{disp_uc.replace('*', 'x')}"
            fig.savefig(cs_dir / f"{slug}.png", dpi=180)
            plt.close(fig)
            plot_file = f"{slug}.png"

        dossiers.append({
            "title": title,
            "uc_id": disp_uc,
            "n_nios_history": len(nios),
            "feeder": row.get("meta__feeder", "-"),
            "cutoff_date": str(row.get("meta__cutoff_date", "-")),
            "avg_coverage": round(float(row.get("x__avg_interval_coverage") or 0.0), 4),
            "median_kwh": round(float(row.get("x__hist_fa_sum_median") or 0.0), 3),
            "mad_kwh": round(float(row.get("x__hist_fa_sum_mad") or 0.0), 3),
            "cv_window": round(float(row.get("x__fa_cv_window") or 0.0), 4),
            "slope_kwh_day": round(float(row.get("x__trend_fa_slope") or 0.0), 4),
            "current_imbalance_max": round(float(row.get("x__hist_current_imbalance_max") or 0.0), 4),
            "ra_reversal_ratio_max": round(float(row.get("x__hist_ra_reversal_ratio_max") or 0.0), 4),
            "anomaly_score": round(float(row.get("ANOMALY_SCORE") or 0.0), 4),
            "dias_anomalia_seguidos": int(row.get("DIAS_ANOMALIA_SEGUIDOS") or 0),
            "categoria_hipotese": row.get("CATEGORIA_DIAGNOSTICA", "-"),
            "is_candidato_vistoria": bool(row.get("IS_CANDIDATO_VISTORIA", False)),
            "cusum_step_change": ts.get("step_change"),
            "plot_file": plot_file,
            "nota_epistemologica": (
                "Hipótese diagnóstica automática baseada em telemetria; "
                "não constitui confirmação de irregularidade sem inspeção física."
            ),
        })

    (cs_dir / "case_studies_dossiers.json").write_text(json.dumps(dossiers, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Estudos de Caso Reais e Raio-X da UC (Anonimizados)\n"]
    for d in dossiers:
        md.extend([
            f"## {d['title']} (`{d['uc_id']}`)",
            f"- **Alimentador:** {d['feeder']} | **Cutoff:** {d['cutoff_date']} | **Cobertura:** {d['avg_coverage']*100:.1f}% | **NIOs mapeados:** {d['n_nios_history']}",
            f"- **Métricas:** Mediana={d['median_kwh']} kWh/dia | MAD={d['mad_kwh']} | CV={d['cv_window']} | Slope={d['slope_kwh_day']} | Desb.I={d['current_imbalance_max']} | Reversão RA={d['ra_reversal_ratio_max']}",
            f"- **Modelo:** Score={d['anomaly_score']} | Streak={d['dias_anomalia_seguidos']}d | Hipótese=`{d['categoria_hipotese']}` | Candidato={d['is_candidato_vistoria']}",
            f"- **Ressalva:** {d['nota_epistemologica']}\n",
        ])
    (cs_dir / "CASE_STUDIES_XRAY.md").write_text("\n".join(md), encoding="utf-8")
    return {"cases_count": len(dossiers), "cases": dossiers}


def run_operations_and_economics(
    out_dir: Path,
    *,
    classified_df: pl.DataFrame,
) -> dict[str, Any]:
    """Operational cohort tracking (new/persistent/resolved/reincident) + parameterized economic sensitivity."""
    ops_dir = out_dir / "operations"
    econ_dir = out_dir / "economics"
    ops_dir.mkdir(parents=True, exist_ok=True)
    econ_dir.mkdir(parents=True, exist_ok=True)

    # 1. Operational tracking across cutoffs
    ops_summary: dict[str, Any] = {"cutoffs": []}
    if "meta__cutoff_date" in classified_df.columns and "IS_CANDIDATO_VISTORIA" in classified_df.columns:
        cutoffs = sorted(classified_df["meta__cutoff_date"].drop_nulls().unique().to_list())
        prev_cands: set[str] = set()
        seen_ever: set[str] = set()
        cohort_rows = []
        for c_dt in cutoffs:
            sub = classified_df.filter(pl.col("meta__cutoff_date") == c_dt)
            cur_cands = set(sub.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)["id__uc_id"].cast(pl.String).to_list())  # noqa: E712
            persistent = cur_cands & prev_cands
            new_or_reinc = cur_cands - prev_cands
            reincident = new_or_reinc & seen_ever
            truly_new = new_or_reinc - seen_ever
            resolved = prev_cands - cur_cands
            cohort_rows.append({
                "cutoff_date": str(c_dt),
                "total_candidates": len(cur_cands),
                "new_candidates": len(truly_new),
                "persistent_candidates": len(persistent),
                "reincident_candidates": len(reincident),
                "resolved_since_prev": len(resolved),
            })
            prev_cands = cur_cands
            seen_ever |= cur_cands

        pl.DataFrame(cohort_rows).write_csv(ops_dir / "operational_cohort_dynamics.csv", separator=";")
        ops_summary["cutoffs"] = cohort_rows

    # Capacity backlog vs parameterized monthly inspection capacity
    uniq_df = deduplicate_latest_uc(classified_df)
    n_candidates = int(uniq_df.filter(pl.col("IS_CANDIDATO_VISTORIA") == True).height) if "IS_CANDIDATO_VISTORIA" in uniq_df.columns else 0  # noqa: E712
    cap_rows = []
    for cap in [50, 100, 150, 250, 500]:
        months_backlog = round(n_candidates / float(cap), 2) if n_candidates > 0 else 0.0
        cap_rows.append({
            "monthly_inspection_capacity": cap,
            "current_candidate_backlog": n_candidates,
            "months_to_clear_backlog": months_backlog,
            "selectivity_needed_pct": round(min(100.0, cap / max(n_candidates, 1) * 100.0), 1),
        })
    pl.DataFrame(cap_rows).write_csv(ops_dir / "capacity_backlog_scenarios.csv", separator=";")
    ops_summary["capacity_scenarios"] = cap_rows

    # 2. Parameterized Economic Sensitivity Simulator (explicitly labeled as Scenarios, NOT COPEL actuals)
    econ_rows = []
    for cost_insp in [180.0, 250.0, 350.0]:
        for hit_rate in [0.15, 0.25, 0.35]:
            for avg_recovery in [1800.0, 3200.0, 5000.0]:
                realized_pct = 0.65
                n_insp = min(max(n_candidates, 100), 274)
                total_cost = n_insp * cost_insp
                expected_hits = n_insp * hit_rate
                gross_recovery = expected_hits * avg_recovery * realized_pct
                net_margin = gross_recovery - total_cost
                roi = net_margin / max(total_cost, 1.0)
                econ_rows.append({
                    "scenario_type": "SENSITIVITY_SIMULATION_NOT_ACTUAL",
                    "inspections_simulated": n_insp,
                    "unit_inspection_cost_brl": cost_insp,
                    "assumed_hit_rate": hit_rate,
                    "assumed_recoverable_brl": avg_recovery,
                    "assumed_realization_factor": realized_pct,
                    "total_cost_brl": round(total_cost, 2),
                    "expected_gross_recovery_brl": round(gross_recovery, 2),
                    "expected_net_benefit_brl": round(net_margin, 2),
                    "roi_ratio": round(roi, 2),
                })
    pl.DataFrame(econ_rows).write_csv(econ_dir / "economic_sensitivity_scenarios.csv", separator=";")

    return {
        "operations": ops_summary,
        "economics_scenarios_generated": len(econ_rows),
        "pilot_target_note": "40% slot substitution and +5pp yield are strictly pilot targets, not measured outcomes.",
    }
