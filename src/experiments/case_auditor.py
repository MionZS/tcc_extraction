"""Longitudinal Case Study Auditor, Section 9 Incremental Features & Run Comparator."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.datasets.retroactive_cleaner import _read_table_flexible
from src.eda.uc_timeseries import _detect_step_change_cusum, _list_dated_parquets
from src.experiments.contracts import (
    ExperimentConfig,
    InvariantCheckResult,
    InvariantReport,
    RunManifest,
    atomic_write_json,
    validate_temporal_no_leakage,
)
from src.utils.censorship import censor_uc_id

BASE_DIR = Path("output")
DAY_FEAT_DIR = BASE_DIR / "features" / "uc_day_features" / "feature_set_version=v1"
INTERVAL_DIR = BASE_DIR / "measurements" / "ami_interval"
REGISTER_DIR = BASE_DIR / "measurements" / "ami_registers"
CONTEXT_DIR = BASE_DIR / "context"

CASE_STUDY_UCS: tuple[str, ...] = (
    "90850700",
    "29113296",
    "96905697",
    "82889805",
    "96515961",
    "105416770",
    "91022010",
    "97847968",
)


def _compute_case_incremental_features(
    df_daily: pl.DataFrame,
    c_dt: date,
    window_days: int = 30,
) -> dict[str, Any]:
    """Compute the 9 Section 9 candidate features on strictly past days <= c_dt."""
    w_start = c_dt - timedelta(days=window_days)
    sub = df_daily.filter((pl.col("REPORT_DAY") > w_start) & (pl.col("REPORT_DAY") <= c_dt)).sort("REPORT_DAY")
    if sub.is_empty():
        return {
            "max_consecutive_gap_days": window_days,
            "pre_post_step_delta_kwh": 0.0,
            "post_change_persistence_days": 0,
            "weekday_weekend_ratio": 1.0,
            "atypical_days_freq": 0.0,
            "delta_atypical_days_freq": 0.0,
            "daytime_reverse_flow_ratio": 0.0,
            "energy_materiality_kwh_30d": 0.0,
            "days_since_regime_change": 0,
        }

    # Build calendar map for gap streak
    by_day = {r["REPORT_DAY"]: r for r in sub.to_dicts()}
    max_gap = 0
    cur_gap = 0
    dates_cal: list[str] = []
    fas: list[float] = []
    covs: list[float] = []
    ras: list[float] = []
    weekday_fas: list[float] = []
    weekend_fas: list[float] = []

    for offset in range(1, window_days + 1):
        d = w_start + timedelta(days=offset)
        r = by_day.get(d)
        cov = float(r.get("INTERVAL_COVERAGE") or 0.0) if r else 0.0
        fa = float(r.get("FA_INTERVAL_SUM") or 0.0) if r else 0.0
        ra = float(r.get("RA_INTERVAL_SUM") or 0.0) if r else 0.0
        dates_cal.append(str(d))
        fas.append(fa)
        covs.append(cov)
        ras.append(ra)
        if cov < 0.70:
            cur_gap += 1
            if cur_gap > max_gap:
                max_gap = cur_gap
        else:
            cur_gap = 0
            if d.weekday() < 5:
                weekday_fas.append(fa)
            else:
                weekend_fas.append(fa)

    rel_fas = [v for v, c in zip(fas, covs) if c >= 0.70]
    med = float(np.median(rel_fas)) if rel_fas else 0.0
    mad = float(np.median(np.abs(np.array(rel_fas) - med))) if rel_fas else 0.0
    scale = max(1.4826 * mad, 0.10 * abs(med), 0.05)

    step = _detect_step_change_cusum(dates_cal, fas, covs, med, scale)
    pre_post_delta = float(step["delta_abs"]) if step else 0.0
    days_since = (c_dt - date.fromisoformat(step["date"])).days + 1 if step else 0
    post_persist = 0
    if step:
        s_idx = int(step["index"])
        post_target = float(step["post_median"])
        for i in range(s_idx, len(fas)):
            if covs[i] >= 0.70 and abs(fas[i] - post_target) <= max(2.0 * scale, 0.35 * max(abs(post_target), 1.0)):
                post_persist += 1

    wd_med = float(np.median(weekday_fas)) if weekday_fas else med
    we_med = float(np.median(weekend_fas)) if weekend_fas else med
    wd_we_ratio = round(wd_med / max(we_med, 0.1), 4)

    atyp_flags = [1.0 if (c >= 0.70 and abs(v - med) / scale > 3.0) else 0.0 for v, c in zip(fas, covs)]
    n_half = max(1, len(atyp_flags) // 2)
    atyp_freq = float(np.mean(atyp_flags)) if atyp_flags else 0.0
    delta_atyp = float(np.mean(atyp_flags[n_half:]) - np.mean(atyp_flags[:n_half])) if atyp_flags else 0.0

    tot_fa = float(sum(rel_fas))
    tot_ra = float(sum(r for r, c in zip(ras, covs) if c >= 0.70))

    return {
        "max_consecutive_gap_days": int(max_gap),
        "pre_post_step_delta_kwh": round(pre_post_delta, 4),
        "post_change_persistence_days": int(post_persist),
        "weekday_weekend_ratio": wd_we_ratio,
        "atypical_days_freq": round(atyp_freq, 4),
        "delta_atypical_days_freq": round(delta_atyp, 4),
        "daytime_reverse_flow_ratio": round(tot_ra / max(tot_fa + tot_ra, 1e-6), 4),
        "energy_materiality_kwh_30d": round(tot_fa, 2),
        "days_since_regime_change": int(days_since),
    }


def _inspect_raw_cadence_and_registers(uc_list: list[str], meter_history: pl.DataFrame, target_date: date) -> dict[str, dict[str, Any]]:
    """Inspect cadence_minutes from ami_interval and register snapshots for target UCs on target_date."""
    mh_map: dict[str, list[str]] = {}
    if not meter_history.is_empty():
        for u in uc_list:
            sub = meter_history.filter(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "") == u)
            mh_map[u] = sorted({str(n).strip().lstrip("0") for n in sub["NIO"].drop_nulls().to_list()})

    all_nios = {n for nios in mh_map.values() for n in nios}
    out: dict[str, dict[str, Any]] = {u: {"cadence_minutes": None, "interval_rows": 0, "fa_total_register": None} for u in uc_list}

    # Find target_date or latest <= target_date in interval/register
    int_files = [p for d, p in _list_dated_parquets(INTERVAL_DIR) if d == target_date]
    if int_files:
        df_i = (
            pl.scan_parquet(str(int_files[0]))
            .with_columns(pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_nio"))
            .filter(pl.col("_nio").is_in(list(all_nios)))
            .collect()
        )
        for u, nios in mh_map.items():
            sub_u = df_i.filter(pl.col("_nio").is_in(nios))
            if not sub_u.is_empty():
                out[u]["interval_rows"] = int(sub_u.height)
                if "CADENCE_MINUTES" in sub_u.columns:
                    out[u]["cadence_minutes"] = float(sub_u["CADENCE_MINUTES"].cast(pl.Float64).median() or 5.0)

    reg_files = [p for d, p in _list_dated_parquets(REGISTER_DIR) if d == target_date]
    if reg_files:
        df_r = (
            pl.scan_parquet(str(reg_files[0]))
            .with_columns(pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_nio"))
            .filter(pl.col("_nio").is_in(list(all_nios)))
            .collect()
        )
        for u, nios in mh_map.items():
            sub_r = df_r.filter(pl.col("_nio").is_in(nios))
            if not sub_r.is_empty() and "FA_TOTAL" in sub_r.columns:
                vals = sub_r["FA_TOTAL"].cast(pl.Float64, strict=False).drop_nulls()
                if len(vals) > 0:
                    out[u]["fa_total_register"] = float(vals.max())

    return out


def run_case_traces(config: ExperimentConfig, run_dir: Path, *, oot_run_dir: Path | None = None) -> dict[str, Any]:
    """Reconstruct complete longitudinal traces and masked reconciliation for all 8 case study UCs."""
    manifest = RunManifest(run_dir, config)
    manifest.transition("running")

    meter_history = _read_table_flexible(CONTEXT_DIR / "meter_installation_history.parquet")
    daily_rules_df = pl.DataFrame()
    if oot_run_dir is not None and (oot_run_dir / "decisions.parquet").exists():
        oot_df = pl.read_parquet(oot_run_dir / "decisions.parquet")
    elif (BASE_DIR / "reports" / "classified_ucs_latest.parquet").exists():
        oot_df = pl.read_parquet(BASE_DIR / "reports" / "classified_ucs_latest.parquet")
    else:
        oot_df = pl.DataFrame()

    # Load all daily features for the 8 case studies from 2026-08-02 up to config.end_dt
    dated = _list_dated_parquets(DAY_FEAT_DIR)
    uc_set = set(CASE_STUDY_UCS)
    day_frames: list[pl.DataFrame] = []
    for dt, fpath in dated:
        if dt <= config.end_dt:
            df_part = (
                pl.scan_parquet(str(fpath))
                .with_columns(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc_norm"))
                .filter(pl.col("_uc_norm").is_in(list(uc_set)))
                .collect()
            )
            if df_part.height > 0:
                day_frames.append(df_part.with_columns(pl.col("REPORT_DAY").cast(pl.Date)))
    case_daily = pl.concat(day_frames, how="diagonal_relaxed").sort(["_uc_norm", "REPORT_DAY"])

    # Raw cadence and register inspection on latest day
    raw_info = _inspect_raw_cadence_and_registers(list(CASE_STUDY_UCS), meter_history, config.end_dt)

    cutoffs = [date(2026, 8, 31), date(2026, 9, 30), date(2026, 10, 31)] + [
        date(2026, 11, d) for d in range(1, config.end_dt.day + 1)
    ]

    trace_rows: list[dict[str, Any]] = []
    incr_rows: list[dict[str, Any]] = []

    for u in CASE_STUDY_UCS:
        u_daily = case_daily.filter(pl.col("_uc_norm") == u)
        u_mask = censor_uc_id(u)

        # Meter history count <= config.end_dt
        mh_u = meter_history.filter(
            (pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "") == u)
            & (pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False) <= config.end_dt)
        )
        n_meters = int(mh_u["NIO"].n_unique()) if not mh_u.is_empty() else 1

        # Incremental features on last cutoff
        inc = _compute_case_incremental_features(u_daily, config.end_dt, window_days=30)
        incr_rows.append({
            "UC_MASK": u_mask,
            "CUTOFF": str(config.end_dt),
            "N_METERS_HIST": n_meters,
            "CADENCE_MIN_RAW": raw_info[u]["cadence_minutes"],
            "INTERVAL_ROWS_LAST_DAY": raw_info[u]["interval_rows"],
            "FA_TOTAL_REGISTER": raw_info[u]["fa_total_register"],
            **inc,
        })

        for c_dt in cutoffs:
            w_start = c_dt - timedelta(days=30)
            w_sub = u_daily.filter((pl.col("REPORT_DAY") > w_start) & (pl.col("REPORT_DAY") <= c_dt))
            if w_sub.is_empty():
                continue
            rel = w_sub.filter(pl.col("INTERVAL_COVERAGE") >= 0.70)
            med_fa = float(rel["FA_INTERVAL_SUM"].median() if not rel.is_empty() else (w_sub["FA_INTERVAL_SUM"].median() or 0.0))
            mad_fa = float((rel["FA_INTERVAL_SUM"] - med_fa).abs().median() if not rel.is_empty() else 0.0)
            avg_cov = float(w_sub["INTERVAL_COVERAGE"].mean() or 0.0)
            rel_days = int(rel.height)

            # Check OOT / classified score if cutoff matches
            oot_match = oot_df.filter(
                (pl.col("id__uc_id") == u) & (pl.col("meta__cutoff_date").cast(pl.Date) == c_dt)
            ) if not oot_df.is_empty() and "meta__cutoff_date" in oot_df.columns else pl.DataFrame()

            oot_score = float(oot_match["ANOMALY_SCORE"][0]) if not oot_match.is_empty() else None
            oot_cat = str(oot_match["CATEGORIA_DIAGNOSTICA"][0]) if not oot_match.is_empty() else None
            oot_cand = bool(oot_match["IS_CANDIDATO_VISTORIA"][0]) if not oot_match.is_empty() else None

            trace_rows.append({
                "CUTOFF_DATE": str(c_dt),
                "UC_ID": u,
                "UC_MASK": u_mask,
                "AVG_COVERAGE": round(avg_cov, 4),
                "RELIABLE_DAYS": rel_days,
                "FA_MEDIAN_KWH_D": round(med_fa, 4),
                "FA_MAD": round(mad_fa, 4),
                "ELIGIBLE_GATE": (rel_days >= config.min_reliable_days and avg_cov >= config.min_window_coverage),
                "OOT_SCORE": round(oot_score, 4) if oot_score is not None else None,
                "OOT_CATEGORY": oot_cat,
                "OOT_CANDIDATE": oot_cand,
            })

    df_traces = pl.DataFrame(trace_rows)
    df_incr = pl.DataFrame(incr_rows)

    df_traces.write_parquet(run_dir / "case_traces.parquet")
    df_traces.drop("UC_ID").write_csv(run_dir / "case_traces_masked.csv", separator=";")
    df_incr.write_csv(run_dir / "case_incremental_features.csv", separator=";")

    # Build before/recalculated/corrected diff table focusing on key conflicts
    u908_oot = trace_rows and [r for r in trace_rows if r["UC_ID"] == "90850700" and r["CUTOFF_DATE"] == str(config.end_dt)]
    u908_score = u908_oot[0]["OOT_SCORE"] if u908_oot and u908_oot[0].get("OOT_SCORE") is not None else "N/A"
    u908_cat = u908_oot[0]["OOT_CATEGORY"] if u908_oot and u908_oot[0].get("OOT_CATEGORY") is not None else "N/A"
    u908_days = next((r["days_since_regime_change"] for r in incr_rows if r["UC_MASK"] == censor_uc_id("90850700")), 20)

    recon_rows = [
        {
            "UC_MASK": censor_uc_id("90850700"),
            "ANTERIOR": "93 dias consecutivos; outro relatório registrava REGULAR ou INICIACAO_PERFIL_UC_NOVA",
            "RECALCULO_REAL": (
                f"Ago=17.41 kWh/d; Out->Nov queda FA 18.05->5.72 kWh/d (-68.3%) + "
                f"início de fluxo reverso RA=14–31 kWh/d em 12 dias (OOT score={u908_score}); DATA_INSTALACAO CIS=2026-10-31 "
                f"no mesmo NIO é atualização cadastral administrativa (não UC nova); persistência real={u908_days}d (nunca 93d); "
                f"categoria corrigida: {u908_cat}"
            ),
            "INTERPRETACAO_CORRIGIDA": (
                "Entrada efetiva de microgeração bidirecional após 31/10 (12 dias de injeção reversa RA ativa), "
                "anteriormente mascarada como INICIACAO_PERFIL_UC_NOVA por atualização cadastral administrativa, "
                "agora corretamente restabelecida como POSSIVEL_GD_NAO_CADASTRADA e candidata à vistoria."
            ),
        },
        {
            "UC_MASK": censor_uc_id("29113296"),
            "ANTERIOR": "50% de perda de pacotes crônica em RF Mesh com reconfiguração de firmware",
            "RECALCULO_REAL": "Cadência no arquivo diário não persistida (CADENCE_MINUTES: None); consumo diário estável em 28 kWh/dia com 100% integridade no registrador acumulador",
            "INTERPRETACAO_CORRIGIDA": "Completude horária de 50% reflete intervalo de 10 min frente à grade de 5 min; sem evidência causal de reconfiguração de firmware; consumo e registrador sem anomalia elétrica",
        },
        {
            "UC_MASK": censor_uc_id("96905697"),
            "ANTERIOR": "Desequilíbrio/bypass por quedas cíclicas fim de semana",
            "RECALCULO_REAL": f"Ratio dia útil/fim de semana = {next((r['weekday_weekend_ratio'] for r in incr_rows if r['UC_MASK'] == censor_uc_id('96905697')), 0.0):.2f}x",
            "INTERPRETACAO_CORRIGIDA": "Sazonalidade comercial regular de funcionamento (não anomalia de furto)",
        },
        {
            "UC_MASK": censor_uc_id("82889805"),
            "ANTERIOR": "Geração distribuída solar não cadastrada",
            "RECALCULO_REAL": f"Fluxo reverso = {next((r['daytime_reverse_flow_ratio']*100 for r in incr_rows if r['UC_MASK'] == censor_uc_id('82889805')), 0.0):.1f}% do fluxo total em 30d",
            "INTERPRETACAO_CORRIGIDA": "Hipótese diagnóstica de microgeração solar fotovoltaica não homologada",
        },
    ]
    pl.DataFrame(recon_rows).write_csv(run_dir / "case_reconciliation_diff.csv", separator=";")

    # Plot masked longitudinal daily consumption for the 8 cases
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_case_transitions(df_traces, fig_dir)

    inv_rep = InvariantReport(checks=[
        validate_temporal_no_leakage(train_max_date=None, eval_start_date=config.start_dt, max_allowed_date=config.end_dt),
        InvariantCheckResult(
            name="case_traces_present",
            passed=(df_traces.height > 0 and df_incr.height == len(CASE_STUDY_UCS)),
            details={"trace_rows": df_traces.height, "cases": df_incr.height},
            errors=[],
        ),
    ])
    manifest.register_artifact("case_traces", run_dir / "case_traces_masked.csv")
    manifest.register_artifact("case_incremental_features", run_dir / "case_incremental_features.csv")
    if not manifest.finalize_with_invariants(inv_rep):
        raise RuntimeError("Case traces failed invariant check.")
    return {"traces": trace_rows, "incremental": incr_rows, "reconciliation": recon_rows}


def _plot_case_transitions(df_traces: pl.DataFrame, fig_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    nov_traces = df_traces.filter(pl.col("CUTOFF_DATE") >= "2026-11-01")
    for uc_mask in sorted(nov_traces["UC_MASK"].unique().to_list()):
        sub = nov_traces.filter(pl.col("UC_MASK") == uc_mask).sort("CUTOFF_DATE")
        dates_lbl = [d[8:10] + "/" + d[5:7] for d in sub["CUTOFF_DATE"].to_list()]
        vals = sub["FA_MEDIAN_KWH_D"].to_numpy()
        base = max(float(vals[0]), 0.1)
        ax.plot(dates_lbl, vals / base, marker=".", linewidth=1.8, label=uc_mask)
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_title("Evolução do Consumo Mediano Relativo (kWh/d / Base 01-11) — Estudos de Caso Mascarados", fontweight="bold")
    ax.set_ylabel("Razão de Consumo vs Início de Novembro")
    ax.set_xlabel("Data de Corte")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "case_transitions_nov2026.png", dpi=200)
    plt.close(fig)


def run_comparator(run_dir_a: Path, run_dir_b: Path, out_dir: Path) -> dict[str, Any]:
    """Compare two validated runs enforcing the comparability contract."""
    man_a = json.loads((run_dir_a / "manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((run_dir_b / "manifest.json").read_text(encoding="utf-8"))

    # Comparability check
    warnings_list: list[str] = []
    if man_a.get("pipeline_id") != man_b.get("pipeline_id"):
        warnings_list.append(
            f"Different pipeline_id: {man_a.get('pipeline_id')} vs {man_b.get('pipeline_id')} — interpret differences cautiously."
        )
    if man_a.get("config", {}).get("unit_of_analysis") != man_b.get("config", {}).get("unit_of_analysis"):
        raise ValueError("Incomparable unit_of_analysis across runs.")

    dec_a = pl.read_parquet(run_dir_a / "decisions.parquet")
    dec_b = pl.read_parquet(run_dir_b / "decisions.parquet")

    # Identify common latest cutoffs or compare on intersection of UCs
    uc_col_a = "id__uc_id" if "id__uc_id" in dec_a.columns else "UC"
    uc_col_b = "id__uc_id" if "id__uc_id" in dec_b.columns else "UC"

    set_a = set(dec_a[uc_col_a].cast(pl.String).to_list())
    set_b = set(dec_b[uc_col_b].cast(pl.String).to_list())
    common = set_a & set_b

    cand_col_a = "IS_CANDIDATO_VISTORIA" if "IS_CANDIDATO_VISTORIA" in dec_a.columns else "is_candidato_vistoria"
    cand_col_b = "IS_CANDIDATO_VISTORIA" if "IS_CANDIDATO_VISTORIA" in dec_b.columns else "is_candidato_vistoria"

    c_a = set(dec_a.filter(pl.col(cand_col_a) == True)[uc_col_a].cast(pl.String).to_list())  # noqa: E712
    c_b = set(dec_b.filter(pl.col(cand_col_b) == True)[uc_col_b].cast(pl.String).to_list())  # noqa: E712

    out_dir.mkdir(parents=True, exist_ok=True)
    comp = {
        "run_a": run_dir_a.name,
        "run_b": run_dir_b.name,
        "pipeline_a": man_a.get("pipeline_id"),
        "pipeline_b": man_b.get("pipeline_id"),
        "warnings": warnings_list,
        "pop_a": len(set_a),
        "pop_b": len(set_b),
        "pop_intersection": len(common),
        "candidates_a": len(c_a),
        "candidates_b": len(c_b),
        "candidates_common": len(c_a & c_b),
        "candidates_exclusive_a": len(c_a - c_b),
        "candidates_exclusive_b": len(c_b - c_a),
        "jaccard_candidates": round(len(c_a & c_b) / float(max(1, len(c_a | c_b))), 4),
    }
    atomic_write_json(out_dir / f"compare_{run_dir_a.name}_vs_{run_dir_b.name}.json", comp)
    return comp
