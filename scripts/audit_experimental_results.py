"""P0 Experimental Audit & Sensitivity Suite for TCC AMI.

Executes Phase A (P0) strictly from validated, frozen artifacts without refitting on OOT:
- Section 4: Population & results reconciliation from single source of truth
- Section 5: Detector correlation audit (raw/norm, Pearson/Spearman, orientation, ties/zeros)
- Section 6: Frozen-model gating sensitivity (50%..90%) without refitting on OOT
- Section 7: Ensemble weights sensitivity (Protocol A Top-2000 & Protocol B fixed/train-calibrated threshold)
- Section 8: Corroboration bonus (+0.15) sensitivity (0, 0.075, 0.15, 0.225, 0.30)
- Section 9: Specialist participation audit matrix
- Section 25 (P0): Publication-grade figures 1..8 (PDF + PNG) with Portuguese formatting
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import kendalltau, spearmanr

from src.models.composite import select_pure_electrical_features
from src.utils.censorship import build_feeder_anonymizer
from src.utils.joblib_splitter import load_split_joblib

SUITE_PATH = PROJECT_ROOT / "experiments" / "runs" / "latest_suite.json"
P0_FIG_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p0_audit" / "figures"
P0_DATA_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p0_audit"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_br_int(val: int | float) -> str:
    return f"{int(round(val)):,}".replace(",", ".")


def _fmt_br_float(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _setup_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_p = stem.with_suffix(".pdf")
    png_p = stem.with_suffix(".png")
    fig.savefig(pdf_p, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_p, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_p.relative_to(PROJECT_ROOT)), "png": str(png_p.relative_to(PROJECT_ROOT))}


def _topk_indices(scores: np.ndarray, k: int) -> set[int]:
    """Stable deterministic Top-K selection breaking ties by index."""
    n = len(scores)
    k_eff = min(k, n)
    # lexsort sorts ascending; primary key = -scores, secondary = index
    order = np.lexsort((np.arange(n), -scores))
    return set(order[:k_eff].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / float(max(1, len(a | b)))


def run_p0_audit() -> dict[str, Any]:
    t0 = time.time()
    _setup_pub_style()
    commit = _git_commit()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_p0_audit_sensitivity"
    run_dir = PROJECT_ROOT / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    P0_FIG_DIR.mkdir(parents=True, exist_ok=True)
    P0_DATA_DIR.mkdir(parents=True, exist_ok=True)

    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    oot_dir = PROJECT_ROOT / suite["oot_multiview"]
    daily_dir = PROJECT_ROOT / suite["daily_rules"]
    abl_dir = PROJECT_ROOT / suite["ablation"]
    gat_dir = PROJECT_ROOT / suite["gating"]
    stab_dir = PROJECT_ROOT / suite["score_stability"]

    oot_manifest = json.loads((oot_dir / "manifest.json").read_text(encoding="utf-8"))
    daily_manifest = json.loads((daily_dir / "manifest.json").read_text(encoding="utf-8"))
    abl_manifest = json.loads((abl_dir / "manifest.json").read_text(encoding="utf-8"))
    gat_manifest = json.loads((gat_dir / "manifest.json").read_text(encoding="utf-8"))

    # Load core validated artifacts (read-only)
    uc_ctx = pl.read_parquet(PROJECT_ROOT / "output/context/uc_context.parquet")
    hier = pl.read_parquet(PROJECT_ROOT / "output/context/electrical_hierarchy.parquet")
    tr_ds = pl.read_parquet(PROJECT_ROOT / "output/model_input/v1/training_dataset.parquet")
    oot_dec = pl.read_parquet(oot_dir / "decisions.parquet")
    daily_dec = pl.read_parquet(daily_dir / "decisions.parquet")
    daily_met = pl.read_parquet(daily_dir / "daily_metrics.parquet")
    daily_19 = daily_dec.filter(pl.col("CUTOFF_DATE").cast(pl.String).str.slice(0, 10) == "2026-11-19")
    pipe = load_split_joblib(oot_dir / "frozen_multiview_model.joblib")
    det = pipe.elec_detector.detector

    # Anonymizer for feeders
    all_feeders_raw = sorted({
        str(x) for x in (
            hier["ALIMENTADOR"].drop_nulls().unique().to_list()
            + oot_dec["meta__feeder"].drop_nulls().unique().to_list()
        )
        if x and str(x).strip()
    })
    feeder_map = build_feeder_anonymizer(all_feeders_raw)

    # =========================================================================
    # SECTION 4: POPULATION & RESULTS RECONCILIATION
    # =========================================================================
    n_cad = int(uc_ctx["UC"].n_unique())
    n_hier = int(hier["UC"].n_unique())
    n_tr_rows = int(tr_ds.height)
    n_tr_ucs = int(tr_ds["id__uc_id"].n_unique())
    n_train_clean = int(tr_ds.filter((pl.col("meta__split") == "train") & pl.col("meta__is_clean_for_train").fill_null(True)).height)
    n_oot = int(oot_dec.height)
    n_oot_elig = int(oot_dec.filter(pl.col("is_reliable_elec")).height)
    n_oot_quar = int(oot_dec.filter(~pl.col("is_reliable_elec")).height)
    n_oot_clean_flag = int(oot_dec.filter(pl.col("meta__is_clean_for_train")).height)
    n_oot_cand = int(oot_dec.filter(pl.col("IS_CANDIDATO_VISTORIA")).height)
    n_oot_elec_anom = int(oot_dec.filter(pl.col("is_elec_anomaly")).height)
    n_oot_is_anom = int(oot_dec.filter(pl.col("IS_ANOMALY")).height)
    n_oot_topo_m3 = int(oot_dec.filter(pl.col("is_reliable_elec") & pl.col("meta__feeder").is_not_null()).height)
    n_oot_topo_cand = int(oot_dec.filter(pl.col("is_reliable_elec") & pl.col("meta__feeder").is_not_null() & pl.col("IS_CANDIDATO_VISTORIA")).height)
    n_daily_win = int(daily_dec.height)
    mean_daily_cand = round(float(daily_met["CANDIDATOS_VISTORIA"].mean()), 2)
    n_daily_19_cand = int(daily_19.filter(pl.col("is_candidato_vistoria")).height)

    oot_cand_set = set(oot_dec.filter(pl.col("IS_CANDIDATO_VISTORIA"))["id__uc_id"].to_list())
    daily_19_cand_set = set(daily_19.filter(pl.col("is_candidato_vistoria"))["id__uc_id"].to_list())
    n_inter = len(oot_cand_set & daily_19_cand_set)
    n_oot_excl = len(oot_cand_set - daily_19_cand_set)
    n_daily_excl = len(daily_19_cand_set - oot_cand_set)

    # Check exact difference between meta__is_clean_for_train (63,081) and is_reliable_elec (62,857)
    clean_vs_rel = oot_dec.filter(pl.col("meta__is_clean_for_train") != pl.col("is_reliable_elec"))
    clean_only = oot_dec.filter(pl.col("meta__is_clean_for_train") & (~pl.col("is_reliable_elec")))
    rel_only = oot_dec.filter((~pl.col("meta__is_clean_for_train")) & pl.col("is_reliable_elec"))

    reconciliation_rows = [
        {"indicador": "UCs cadastrais (snapshot 25/11)", "valor_reproduzido": n_cad, "valor_historico_ref": 69895, "delta": n_cad - 69895,
         "origem": "output/context/uc_context.parquet", "granularidade": "UC única", "periodo": "Cadastro vigente 25/11/2026",
         "filtros": "Nenhum", "denominador": "Universo CIS/GEO", "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "Snapshot cadastral atualizado (+12 UCs vs snapshot 20/11 de 69.895)"},
        {"indicador": "UCs únicas no dataset de modelagem (4 cortes)", "valor_reproduzido": n_tr_ucs, "valor_historico_ref": 66099, "delta": n_tr_ucs - 66099,
         "origem": "output/model_input/v1/training_dataset.parquet", "granularidade": "UC única (31/08, 30/09, 31/10, 19/11)", "periodo": "01/08–19/11/2026",
         "filtros": "UCs com telemetria em >=1 corte", "denominador": str(n_cad), "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "União dos 4 cortes (262.769 janelas); histórico 66.099 era snapshot exploratório 20/11"},
        {"indicador": "Janelas de treino limpas (Ago–Out/2026)", "valor_reproduzido": n_train_clean, "valor_historico_ref": 189341, "delta": n_train_clean - 189341,
         "origem": "output/model_input/v1/training_dataset.parquet", "granularidade": "UC x cutoff", "periodo": "31/08, 30/09, 31/10/2026",
         "filtros": "meta__split == 'train' & meta__is_clean_for_train", "denominador": "197.083 janelas de treino", "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "Fit estrito pré-novembro (sem vazamento OOT)"},
        {"indicador": "UCs OOT avaliadas (19/11/2026)", "valor_reproduzido": n_oot, "valor_historico_ref": 65686, "delta": n_oot - 65686,
         "origem": f"{suite['oot_multiview']}/decisions.parquet", "granularidade": "UC única (corte 19/11)", "periodo": "21/10–19/11/2026 (janela 30d)",
         "filtros": "meta__split == 'oot_eval'", "denominador": str(n_oot), "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "Inferência congelada OOT_MULTIVIEW"},
        {"indicador": "Elegíveis OOT (cobertura >= 70%)", "valor_reproduzido": n_oot_elig, "valor_historico_ref": 62857, "delta": n_oot_elig - 62857,
         "origem": f"{suite['oot_multiview']}/decisions.parquet", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "is_reliable_elec == True (cov >= 0,70)", "denominador": str(n_oot), "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "Gate de confiabilidade elétrica >= 70%"},
        {"indicador": "Quarentena técnica OOT (cobertura < 70%)", "valor_reproduzido": n_oot_quar, "valor_historico_ref": 2829, "delta": n_oot_quar - 2829,
         "origem": f"{suite['oot_multiview']}/decisions.parquet", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "is_reliable_elec == False (cov < 0,70)", "denominador": str(n_oot), "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "Partição exata: 62.857 + 2.829 = 65.686"},
        {"indicador": "Candidatos OOT (IS_CANDIDATO_VISTORIA)", "valor_reproduzido": n_oot_cand, "valor_historico_ref": 6037, "delta": n_oot_cand - 6037,
         "origem": f"{suite['oot_multiview']}/decisions.parquet", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "IS_ANOMALY & ~is_poor_comm & ~is_new_uc & cat_eletrica", "denominador": str(n_oot_elig), "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "Correspondência exata 1:1 com is_elec_anomaly (6.037)"},
        {"indicador": "Candidatos após restrição topológica (M4/M5)", "valor_reproduzido": n_oot_topo_cand, "valor_historico_ref": 4311, "delta": n_oot_topo_cand - 4311,
         "origem": f"{suite['ablation']}/ablation_funnel.csv", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "M2 & meta__feeder is not null & IS_CANDIDATO_VISTORIA", "denominador": str(n_oot_topo_m3), "run_id": abl_manifest["run_id"], "commit": abl_manifest["git_commit"],
         "politica": "1.726 candidatos sem alimentador cadastrado retidos fora de M3"},
        {"indicador": "Candidatos pelas regras diárias em 19/11", "valor_reproduzido": n_daily_19_cand, "valor_historico_ref": 4978, "delta": n_daily_19_cand - 4978,
         "origem": f"{suite['daily_rules']}/decisions.parquet", "granularidade": "UC única (corte 19/11)", "periodo": "19/11/2026",
         "filtros": "CUTOFF_DATE == 2026-11-19 & is_candidato_vistoria", "denominador": "62.986 elegíveis regras", "run_id": daily_manifest["run_id"], "commit": daily_manifest["git_commit"],
         "politica": "DAILY_RULES calibrado (cobertura >= 60%, >= 12d confiáveis)"},
        {"indicador": "Média diária de candidatos (regras calibradas)", "valor_reproduzido": mean_daily_cand, "valor_historico_ref": 5387.7, "delta": round(mean_daily_cand - 5387.7, 2),
         "origem": f"{suite['daily_rules']}/daily_metrics.parquet", "granularidade": "Média de 19 cortes diários", "periodo": "01/11–19/11/2026",
         "filtros": "is_candidato_vistoria por corte", "denominador": "19 cortes", "run_id": daily_manifest["run_id"], "commit": daily_manifest["git_commit"],
         "politica": "DAILY_RULES calibrado"},
        {"indicador": "Janelas totais da série diária", "valor_reproduzido": n_daily_win, "valor_historico_ref": 1248501, "delta": n_daily_win - 1248501,
         "origem": f"{suite['daily_rules']}/decisions.parquet", "granularidade": "UC x cutoff diário", "periodo": "01/11–19/11/2026",
         "filtros": "Todos os 19 cortes", "denominador": "19 cortes", "run_id": daily_manifest["run_id"], "commit": daily_manifest["git_commit"],
         "politica": "DAILY_RULES calibrado"},
        {"indicador": "Interseção OOT x regras diárias (19/11)", "valor_reproduzido": n_inter, "valor_historico_ref": 2069, "delta": n_inter - 2069,
         "origem": "Cruzamento decisions.parquet (OOT x DAILY_RULES 19/11)", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "Candidato em ambas as políticas", "denominador": f"União = {n_inter + n_oot_excl + n_daily_excl}", "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": f"Jaccard = {n_inter / (n_inter + n_oot_excl + n_daily_excl):.4f}"},
        {"indicador": "Exclusivos OOT (19/11)", "valor_reproduzido": n_oot_excl, "valor_historico_ref": 3968, "delta": n_oot_excl - 3968,
         "origem": "Cruzamento decisions.parquet (OOT \\ DAILY_RULES 19/11)", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "Candidato apenas no OOT", "denominador": str(n_oot_cand), "run_id": oot_manifest["run_id"], "commit": oot_manifest["git_commit"],
         "politica": "2.069 + 3.968 = 6.037"},
        {"indicador": "Exclusivos regras diárias (19/11)", "valor_reproduzido": n_daily_excl, "valor_historico_ref": 2909, "delta": n_daily_excl - 2909,
         "origem": "Cruzamento decisions.parquet (DAILY_RULES 19/11 \\ OOT)", "granularidade": "UC única", "periodo": "19/11/2026",
         "filtros": "Candidato apenas em DAILY_RULES", "denominador": str(n_daily_19_cand), "run_id": daily_manifest["run_id"], "commit": daily_manifest["git_commit"],
         "politica": "2.069 + 2.909 = 4.978"},
    ]
    pl.DataFrame(reconciliation_rows, strict=False).write_csv(run_dir / "reconciliation_table.csv", separator=";")
    pl.DataFrame(reconciliation_rows, strict=False).write_csv(P0_DATA_DIR / "reconciliation_table.csv", separator=";")

    # Category reconciliation (explicitly proving sum == 65,686 and 6,053 - 16 = 6,037)
    cat_df = oot_dec.group_by("CATEGORIA_DIAGNOSTICA").agg([
        pl.len().alias("total_ucs"),
        pl.col("is_reliable_elec").sum().alias("elegiveis_70"),
        pl.col("is_elec_anomaly").sum().alias("is_elec_anomaly_n"),
        pl.col("IS_ANOMALY").sum().alias("is_anomaly_n"),
        pl.col("IS_CANDIDATO_VISTORIA").sum().alias("candidatos_vistoria"),
        pl.col("rule_violation_flag").sum().alias("com_bonus_regra_015"),
        pl.col("s_elec").median().round(4).alias("s_elec_mediana"),
    ]).sort("total_ucs", descending=True)
    cat_df.write_csv(run_dir / "category_reconciliation.csv", separator=";")
    cat_df.write_csv(P0_DATA_DIR / "category_reconciliation.csv", separator=";")

    # 16 GD UCs outside candidate queue
    elec_cats = ["ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT", "POSSIVEL_GD_NAO_CADASTRADA", "FALHA_METROLOGICA_MEDIDOR_TRAVADO"]
    gd16 = oot_dec.filter(pl.col("CATEGORIA_DIAGNOSTICA").is_in(elec_cats) & (~pl.col("IS_CANDIDATO_VISTORIA")))
    gd16_summary = {
        "count": int(gd16.height),
        "categories": gd16["CATEGORIA_DIAGNOSTICA"].value_counts().to_dicts(),
        "all_reliable": bool(gd16["is_reliable_elec"].all()),
        "any_elec_anomaly": bool(gd16["is_elec_anomaly"].any()),
        "s_elec_min": round(float(gd16["s_elec"].min()), 4),
        "s_elec_max": round(float(gd16["s_elec"].max()), 4),
        "frozen_threshold": round(float(det.frozen_threshold_), 4),
        "ra_ratio_min": round(float(gd16["x__hist_ra_reversal_ratio_max"].min()), 4),
        "ra_days_min": int(gd16["x__hist_ra_reversal_days"].min()),
        "explanation": (
            "As 16 UCs pertencem integralmente a POSSIVEL_GD_NAO_CADASTRADA: atenderam à regra física "
            "de injeção reversa (ra_ratio >= 0,08 e ra_days >= 2) com telemetria confiável (cobertura >= 70%), "
            "porém obtiveram escore elétrico s_elec abaixo do limiar congelado (0,2799 a 0,5091 < 0,5131). "
            "Como IS_CANDIDATO_VISTORIA exige is_anomaly == True (equivalente a is_elec_anomaly == True nesta coorte), "
            "elas recebem hipótese diagnóstica de GD sem ingresso automático na fila de 6.037 candidatas."
        ),
    }

    # Feeder reconciliation (anonymized)
    h_by_f = hier.group_by("ALIMENTADOR").agg(pl.col("UC").n_unique().alias("ucs_cadastrais"))
    t_by_f = tr_ds.group_by("meta__feeder").agg(pl.col("id__uc_id").n_unique().alias("ucs_4cortes"))
    o_by_f = oot_dec.group_by("meta__feeder").agg([
        pl.len().alias("ucs_oot"),
        pl.col("is_reliable_elec").sum().alias("elegiveis_oot"),
        pl.col("IS_CANDIDATO_VISTORIA").sum().alias("candidatos_oot"),
    ])
    f_rec = (
        h_by_f.rename({"ALIMENTADOR": "feeder_raw"})
        .join(t_by_f.rename({"meta__feeder": "feeder_raw"}), on="feeder_raw", how="full", coalesce=True)
        .join(o_by_f.rename({"meta__feeder": "feeder_raw"}), on="feeder_raw", how="full", coalesce=True)
        .with_columns([
            pl.col("feeder_raw").map_elements(
                lambda x: "Sem alimentador cadastrado" if x is None else feeder_map.get(str(x), str(x)),
                return_dtype=pl.String
            ).alias("circuito_anonimizado"),
            pl.col("ucs_cadastrais").fill_null(0),
            pl.col("ucs_4cortes").fill_null(0),
            pl.col("ucs_oot").fill_null(0),
            pl.col("elegiveis_oot").fill_null(0),
            pl.col("candidatos_oot").fill_null(0),
        ])
        .drop("feeder_raw")
        .select(["circuito_anonimizado", "ucs_cadastrais", "ucs_4cortes", "ucs_oot", "elegiveis_oot", "candidatos_oot"])
        .sort("ucs_cadastrais", descending=True)
    )
    f_rec.write_csv(run_dir / "feeder_reconciliation.csv", separator=";")
    f_rec.write_csv(P0_DATA_DIR / "feeder_reconciliation.csv", separator=";")

    # =========================================================================
    # SECTION 5: DETECTOR CORRELATION AUDIT
    # =========================================================================
    elec_cols = select_pure_electrical_features(oot_dec)
    X_oot = oot_dec.select(elec_cols).fill_null(0.0).to_numpy()
    preds = det.predict_composite(X_oot, raw_dataframe=oot_dec)

    det_keys = ["iso", "lof", "ecod", "copod", "hbos", "pca"]
    det_labels = ["IF", "LOF", "ECOD", "COPOD", "HBOS", "PCA"]
    raw_mat = np.column_stack([preds[f"raw_{k}_score"] for k in det_keys])
    norm_mat = np.column_stack([preds[f"{k}_score"] for k in det_keys])
    elig_mask = oot_dec["is_reliable_elec"].to_numpy()
    s_elec_arr = preds["consensus_score"]
    w_curr = det.weights
    cons_no_bonus = (
        w_curr["iso_forest"] * preds["iso_score"]
        + w_curr["lof"] * preds["lof_score"]
        + w_curr["ecod"] * preds["ecod_score"]
        + w_curr["copod"] * preds["copod_score"]
        + w_curr["hbos"] * preds["hbos_score"]
        + w_curr["pca"] * preds["pca_score"]
    )

    # Exact verification against decisions.parquet
    alignment_max_diff = {
        k: float(np.max(np.abs(oot_dec[f"{k}_score"].to_numpy() - preds[f"{k}_score"])))
        for k in det_keys
    }
    alignment_max_diff["s_elec"] = float(np.max(np.abs(oot_dec["s_elec"].to_numpy() - s_elec_arr)))

    # Correlation matrices (6x6 + consensus = 7x7)
    norm_mat_7 = np.column_stack([norm_mat, s_elec_arr])
    raw_mat_7 = np.column_stack([raw_mat, s_elec_arr])
    labels_7 = det_labels + ["Consenso"]

    sp_norm_all = spearmanr(norm_mat_7).statistic
    sp_norm_elig = spearmanr(norm_mat_7[elig_mask]).statistic
    sp_raw_all = spearmanr(raw_mat_7).statistic
    sp_raw_elig = spearmanr(raw_mat_7[elig_mask]).statistic
    pe_norm_elig = np.corrcoef(norm_mat_7[elig_mask], rowvar=False)
    pe_raw_elig = np.corrcoef(raw_mat_7[elig_mask], rowvar=False)

    def _mat_to_df(mat: np.ndarray, lbls: list[str]) -> pl.DataFrame:
        data: dict[str, Any] = {"detector": lbls}
        for j, lbl in enumerate(lbls):
            data[lbl] = [round(float(mat[i, j]), 4) for i in range(len(lbls))]
        return pl.DataFrame(data)

    _mat_to_df(sp_norm_all, labels_7).write_csv(run_dir / "corr_spearman_norm_all.csv", separator=";")
    _mat_to_df(sp_norm_elig, labels_7).write_csv(run_dir / "corr_spearman_norm_eligible.csv", separator=";")
    _mat_to_df(sp_raw_elig, labels_7).write_csv(run_dir / "corr_spearman_raw_eligible.csv", separator=";")
    _mat_to_df(pe_norm_elig, labels_7).write_csv(run_dir / "corr_pearson_norm_eligible.csv", separator=";")
    _mat_to_df(sp_norm_elig, labels_7).write_csv(P0_DATA_DIR / "corr_spearman_norm_eligible.csv", separator=";")
    _mat_to_df(pe_norm_elig, labels_7).write_csv(P0_DATA_DIR / "corr_pearson_norm_eligible.csv", separator=";")

    # Score distributions, zeros, ties, saturations
    dist_rows = []
    for i, (k, lbl) in enumerate(zip(det_keys, det_labels)):
        arr_n = norm_mat[elig_mask, i]
        arr_r = raw_mat[elig_mask, i]
        _, counts_n = np.unique(arr_n, return_counts=True)
        tie_ratio_n = float(np.sum(counts_n[counts_n > 1]) / len(arr_n) * 100.0)
        _, counts_r = np.unique(arr_r, return_counts=True)
        tie_ratio_r = float(np.sum(counts_r[counts_r > 1]) / len(arr_r) * 100.0)
        dist_rows.append({
            "detector": lbl,
            "raw_min": round(float(np.min(arr_r)), 5),
            "raw_p25": round(float(np.percentile(arr_r, 25)), 5),
            "raw_p50": round(float(np.median(arr_r)), 5),
            "raw_p75": round(float(np.percentile(arr_r, 75)), 5),
            "raw_p95": round(float(np.percentile(arr_r, 95)), 5),
            "raw_p99": round(float(np.percentile(arr_r, 99)), 5),
            "raw_max": round(float(np.max(arr_r)), 5),
            "norm_mean": round(float(np.mean(arr_n)), 4),
            "norm_std": round(float(np.std(arr_n)), 4),
            "norm_p50": round(float(np.median(arr_n)), 4),
            "norm_p95": round(float(np.percentile(arr_n, 95)), 4),
            "norm_zero_pct": round(float(np.mean(arr_n == 0.0) * 100.0), 2),
            "norm_sat1_pct": round(float(np.mean(arr_n >= 0.9999) * 100.0), 2),
            "norm_tie_pct": round(tie_ratio_n, 2),
            "raw_tie_pct": round(tie_ratio_r, 2),
            "n_nan": int(np.isnan(arr_r).sum()),
            "spearman_vs_cons_all": round(float(sp_norm_all[i, 6]), 4),
            "spearman_vs_cons_elig": round(float(sp_norm_elig[i, 6]), 4),
            "spearman_vs_cons_nobonus_elig": round(float(spearmanr(arr_n, cons_no_bonus[elig_mask]).statistic), 4),
        })
    pl.DataFrame(dist_rows).write_csv(run_dir / "detector_distributions.csv", separator=";")
    pl.DataFrame(dist_rows).write_csv(P0_DATA_DIR / "detector_distributions.csv", separator=";")

    corr_checks = {
        "alignment_max_abs_diff": alignment_max_diff,
        "symmetric_spearman_elig": bool(np.allclose(sp_norm_elig, sp_norm_elig.T, atol=1e-12)),
        "diagonal_is_one": bool(np.allclose(np.diag(sp_norm_elig), 1.0, atol=1e-12)),
        "min_pairwise_spearman_elig": round(float(np.min(sp_norm_elig[:6, :6])), 4),
        "if_ecod_spearman_all": round(float(sp_norm_all[0, 2]), 4),
        "if_ecod_spearman_elig": round(float(sp_norm_elig[0, 2]), 4),
        "if_ecod_pearson_elig": round(float(pe_norm_elig[0, 2]), 4),
        "any_negative_pairwise": bool(np.any(sp_norm_elig[:6, :6] < 0.0)),
    }

    # =========================================================================
    # SECTION 6: FROZEN-MODEL GATING SENSITIVITY (NO REFIT ON OOT!)
    # =========================================================================
    cov_arr = oot_dec["x__avg_interval_coverage"].fill_null(0.0).to_numpy()
    tau_frozen = float(det.frozen_threshold_)
    raw_elec_anom = s_elec_arr >= tau_frozen
    cat_arr = oot_dec["CATEGORIA_DIAGNOSTICA"].to_numpy()
    new_uc_mask = (cat_arr == "INICIACAO_PERFIL_UC_NOVA")
    # Also check raw eligibility logic from category_reporting.py (is_poor_comm = (~is_clean) | (cov < thr))
    # For pure isolated effect of coverage threshold with frozen model:
    # We evaluate pure coverage gate `cov >= thr` keeping s_elec, tau_frozen, and new_uc_mask strictly frozen!
    ref_70_mask = (cov_arr >= 0.70)
    ref_70_cands = set(np.where(raw_elec_anom & ref_70_mask & (~new_uc_mask))[0].tolist())
    # Note: verify exact match at 70% with OOT candidates (6,037)
    oot_cand_idx = set(np.where(oot_dec["IS_CANDIDATO_VISTORIA"].to_numpy())[0].tolist())

    gating_frozen_rows = []
    gating_feeder_dist = []
    gating_profile_dist = []
    feeder_arr = oot_dec["meta__feeder"].fill_null("Sem alimentador").to_numpy()
    class_arr = oot_dec["CLASSE_COMPORTAMENTAL"].fill_null("N/A").to_numpy()

    for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
        elig_t = cov_arr >= thr
        quar_t = ~elig_t
        cands_t_mask = raw_elec_anom & elig_t & (~new_uc_mask)
        cands_t_set = set(np.where(cands_t_mask)[0].tolist())
        intercepted_t = int(np.sum(raw_elec_anom & quar_t))

        retained_vs_70 = len(cands_t_set & ref_70_cands)
        excluded_vs_70 = len(ref_70_cands - cands_t_set)
        new_elig_vs_70 = len(cands_t_set - ref_70_cands)
        jacc_vs_70 = _jaccard(cands_t_set, ref_70_cands)

        # Category changes vs 70%: UCs that flip between reliable (electrical/regular) and quarantine
        flipped_gate = int(np.sum(elig_t != ref_70_mask))

        # Top-2000 among eligible UCs at threshold thr
        elig_idx_t = np.where(elig_t)[0]
        scores_elig_t = np.where(elig_t, s_elec_arr, -1.0)
        top2k_t = _topk_indices(scores_elig_t, 2000)
        scores_elig_70 = np.where(ref_70_mask, s_elec_arr, -1.0)
        top2k_70 = _topk_indices(scores_elig_70, 2000)

        gating_frozen_rows.append({
            "coverage_threshold_pct": int(round(thr * 100)),
            "total_ucs": n_oot,
            "eligible_ucs": int(elig_t.sum()),
            "eligible_pct": round(float(elig_t.mean() * 100.0), 2),
            "quarantine_ucs": int(quar_t.sum()),
            "quarantine_pct": round(float(quar_t.mean() * 100.0), 2),
            "raw_anomalies_frozen_model": int(raw_elec_anom.sum()),
            "candidates_retained": len(cands_t_set),
            "anomalies_intercepted_quarantine": intercepted_t,
            "retained_vs_ref70": retained_vs_70,
            "excluded_vs_ref70": excluded_vs_70,
            "newly_eligible_vs_ref70": new_elig_vs_70,
            "jaccard_cands_vs_ref70": round(jacc_vs_70, 4),
            "top2k_jaccard_vs_ref70": round(_jaccard(top2k_t, top2k_70), 4),
            "top2k_substitutions_vs_ref70": len(top2k_t - top2k_70),
            "category_gate_flips_vs_ref70": flipped_gate,
            "frozen_threshold_used": round(tau_frozen, 4),
        })

        # Distribution by behavioral profile
        for cl in sorted(set(class_arr.tolist())):
            n_cl = int(np.sum(cands_t_mask & (class_arr == cl)))
            gating_profile_dist.append({
                "coverage_threshold_pct": int(round(thr * 100)),
                "classe_comportamental": cl,
                "candidatos": n_cl,
            })

        # Distribution by top feeders (anonymized)
        for f_raw in sorted(set(feeder_arr.tolist())):
            n_f = int(np.sum(cands_t_mask & (feeder_arr == f_raw)))
            gating_feeder_dist.append({
                "coverage_threshold_pct": int(round(thr * 100)),
                "circuito": "Sem alimentador" if f_raw == "Sem alimentador" else feeder_map.get(str(f_raw), str(f_raw)),
                "candidatos": n_f,
            })

    pl.DataFrame(gating_frozen_rows).write_csv(run_dir / "gating_frozen_sensitivity.csv", separator=";")
    pl.DataFrame(gating_frozen_rows).write_csv(P0_DATA_DIR / "gating_frozen_sensitivity.csv", separator=";")
    pl.DataFrame(gating_profile_dist).write_csv(run_dir / "gating_frozen_by_profile.csv", separator=";")
    pl.DataFrame(gating_feeder_dist).write_csv(run_dir / "gating_frozen_by_feeder.csv", separator=";")

    # =========================================================================
    # SECTION 7: ENSEMBLE WEIGHTS SENSITIVITY (PROTOCOL A Top-2000 & PROTOCOL B Threshold)
    # =========================================================================
    # Compute normalized scores on clean training set once so we can also compute train-calibrated P95 thresholds (NO leakage!)
    df_train_clean = tr_ds.filter((pl.col("meta__split") == "train") & pl.col("meta__is_clean_for_train").fill_null(True))
    X_tr_clean = df_train_clean.select(elec_cols).fill_null(0.0).to_numpy()
    preds_train = det.predict_composite(X_tr_clean, raw_dataframe=df_train_clean)
    tr_norm = {
        "iso_forest": preds_train["iso_score"],
        "lof": preds_train["lof_score"],
        "ecod": preds_train["ecod_score"],
        "copod": preds_train["copod_score"],
        "hbos": preds_train["hbos_score"],
        "pca": preds_train["pca_score"],
    }
    tr_rule_flag = preds_train["rule_violation_flag"]

    oot_norm_elig = {
        "iso_forest": preds["iso_score"][elig_mask],
        "lof": preds["lof_score"][elig_mask],
        "ecod": preds["ecod_score"][elig_mask],
        "copod": preds["copod_score"][elig_mask],
        "hbos": preds["hbos_score"][elig_mask],
        "pca": preds["pca_score"][elig_mask],
    }
    oot_rule_elig = preds["rule_violation_flag"][elig_mask]
    cat_elig = cat_arr[elig_mask]
    feeder_elig = np.array([
        "Sem alimentador" if f == "Sem alimentador" else feeder_map.get(str(f), str(f))
        for f in feeder_arr[elig_mask]
    ])

    base_w = dict(det.weights)  # {'iso_forest': 0.22, 'lof': 0.20, 'ecod': 0.18, 'copod': 0.18, 'hbos': 0.10, 'pca': 0.12}
    weight_configs: list[tuple[str, str, dict[str, float]]] = [
        ("Atual (código congelado)", "referencia", dict(base_w)),
        ("Tabela histórica doc.", "alternativa", {"iso_forest": 0.22, "lof": 0.10, "ecod": 0.20, "copod": 0.18, "hbos": 0.18, "pca": 0.12}),
        ("Pesos uniformes (1/6)", "alternativa", {k: 1.0 / 6.0 for k in base_w}),
    ]
    # ±20% individual perturbations (renormalized)
    short_map = {"iso_forest": "IF", "lof": "LOF", "ecod": "ECOD", "copod": "COPOD", "hbos": "HBOS", "pca": "PCA"}
    for k_det, lbl in short_map.items():
        for sign_lbl, mult in [("+20%", 1.20), ("-20%", 0.80)]:
            w_p = dict(base_w)
            w_p[k_det] = base_w[k_det] * mult
            s_tot = sum(w_p.values())
            w_p = {kk: vv / s_tot for kk, vv in w_p.items()}
            weight_configs.append((f"{lbl} {sign_lbl}", "perturbacao_20pct", w_p))
    # Leave-one-out
    for k_det, lbl in short_map.items():
        w_l = {kk: (0.0 if kk == k_det else vv) for kk, vv in base_w.items()}
        s_tot = sum(w_l.values())
        w_l = {kk: vv / s_tot for kk, vv in w_l.items()}
        weight_configs.append((f"Sem {lbl} (5 det.)", "ablacao_loo", w_l))
    # Core 4 (IF, ECOD, COPOD, HBOS) & IF alone
    w_c4 = {k: (base_w[k] if k in {"iso_forest", "ecod", "copod", "hbos"} else 0.0) for k in base_w}
    s_c4 = sum(w_c4.values())
    weight_configs.append(("Núcleo 4 (IF+ECOD+COPOD+HBOS)", "nucleo_4", {k: v / s_c4 for k, v in w_c4.items()}))
    weight_configs.append(("Isolation Forest isolado", "baseline_if", {k: (1.0 if k == "iso_forest" else 0.0) for k in base_w}))

    def _eval_weights(w_map: dict[str, float], norm_dict: dict[str, np.ndarray], rule_flag: np.ndarray, bonus: float = 0.15) -> np.ndarray:
        w_sum = sum(w_map.values())
        sc = sum((w_map[k] / w_sum) * norm_dict[k] for k in w_map)
        return np.round(np.clip(sc + np.where(rule_flag, bonus, 0.0), 0.0, 1.0), 4)

    ref_sc_elig = _eval_weights(base_w, oot_norm_elig, oot_rule_elig, bonus=0.15)
    ref_top2k = _topk_indices(ref_sc_elig, 2000)

    # Stratified subsample for fast exact Kendall tau (5,000 UCs fixed seed) + full Spearman on all 62,857
    rng = np.random.default_rng(42)
    kendall_idx = rng.choice(len(ref_sc_elig), size=min(5000, len(ref_sc_elig)), replace=False)

    weight_rows = []
    topk_cat_rows = []
    for cfg_name, cfg_group, w_map in weight_configs:
        sc_e = _eval_weights(w_map, oot_norm_elig, oot_rule_elig, bonus=0.15)
        # Train-calibrated P95 threshold (strictly on clean train, WITHOUT bonus, matching CompositeAnomalyDetector.fit!)
        sc_tr_nobonus = _eval_weights(w_map, tr_norm, tr_rule_flag, bonus=0.0)
        tau_cal_tr = float(np.percentile(sc_tr_nobonus, 95.0))

        top2k = _topk_indices(sc_e, 2000)
        rho = float(spearmanr(sc_e, ref_sc_elig).statistic)
        ktau = float(kendalltau(sc_e[kendall_idx], ref_sc_elig[kendall_idx]).statistic)
        jacc_2k = _jaccard(top2k, ref_top2k)
        common_2k = len(top2k & ref_top2k)
        excl_cfg = len(top2k - ref_top2k)
        excl_ref = len(ref_top2k - top2k)

        # Protocol B: fixed threshold 0.5131 AND train-recalibrated P95 threshold
        n_fixed_tau = int(np.sum(sc_e >= tau_frozen))
        n_cal_tau = int(np.sum(sc_e >= tau_cal_tr))

        # Top-2000 category shares
        top2k_arr = np.array(sorted(top2k), dtype=int)
        n_top_gd = int(np.sum(cat_elig[top2k_arr] == "POSSIVEL_GD_NAO_CADASTRADA"))
        n_top_pnt = int(np.sum(cat_elig[top2k_arr] == "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT"))
        n_top_met = int(np.sum(cat_elig[top2k_arr] == "FALHA_METROLOGICA_MEDIDOR_TRAVADO"))
        n_top_feeders = int(len(set(feeder_elig[top2k_arr].tolist())))

        weight_rows.append({
            "configuracao": cfg_name,
            "grupo": cfg_group,
            "w_IF": round(w_map["iso_forest"], 4),
            "w_LOF": round(w_map["lof"], 4),
            "w_ECOD": round(w_map["ecod"], 4),
            "w_COPOD": round(w_map["copod"], 4),
            "w_HBOS": round(w_map["hbos"], 4),
            "w_PCA": round(w_map["pca"], 4),
            "spearman_vs_ref": round(rho, 4),
            "kendall_5k_vs_ref": round(ktau, 4),
            "top2k_jaccard": round(jacc_2k, 4),
            "top2k_comuns": common_2k,
            "top2k_substituicoes": excl_cfg,
            "top2k_excl_ref": excl_ref,
            "score_mediana": round(float(np.median(sc_e)), 4),
            "score_p95": round(float(np.percentile(sc_e, 95)), 4),
            "score_p99": round(float(np.percentile(sc_e, 99)), 4),
            "protoB_cands_tau_fixo_05131": n_fixed_tau,
            "protoB_tau_calibrado_treino": round(tau_cal_tr, 4),
            "protoB_cands_tau_calibrado": n_cal_tau,
            "top2k_cat_pnt": n_top_pnt,
            "top2k_cat_gd": n_top_gd,
            "top2k_cat_metro": n_top_met,
            "top2k_n_circuitos": n_top_feeders,
        })
        if cfg_group in {"referencia", "alternativa", "nucleo_4", "baseline_if"} or cfg_name in {"Sem LOF (5 det.)", "Sem PCA (5 det.)"}:
            topk_cat_rows.append({
                "configuracao": cfg_name,
                "PNT_Eletrica": n_top_pnt,
                "Possivel_GD": n_top_gd,
                "Falha_Metrologica": n_top_met,
                "Outras": 2000 - (n_top_pnt + n_top_gd + n_top_met),
            })

    pl.DataFrame(weight_rows).write_csv(run_dir / "weight_sensitivity.csv", separator=";")
    pl.DataFrame(weight_rows).write_csv(P0_DATA_DIR / "weight_sensitivity.csv", separator=";")

    # =========================================================================
    # SECTION 8: CORROBORATION BONUS (+0.15) SENSITIVITY
    # =========================================================================
    v_imb_e = oot_dec["x__hist_voltage_imbalance_max"].fill_null(0.0).to_numpy()[elig_mask]
    c_imb_e = oot_dec["x__hist_current_imbalance_max"].fill_null(0.0).to_numpy()[elig_mask]
    ra_r_e = oot_dec["x__hist_ra_reversal_ratio_max"].fill_null(0.0).to_numpy()[elig_mask]
    strict_prodist_flag_e = (v_imb_e > 20.0) | (ra_r_e > 0.85)  # strictly voltage & reversal without current imbalance

    bonus_rows = []
    for mode_lbl, flag_arr in [
        ("regra_efetiva_25_11_incl_c_imb", oot_rule_elig),
        ("regra_estrita_tensao_e_reversao", strict_prodist_flag_e),
    ]:
        ref_b = _eval_weights(base_w, oot_norm_elig, flag_arr, bonus=0.15)
        ref_b_top2k = _topk_indices(ref_b, 2000)
        for b_val in [0.0, 0.075, 0.15, 0.225, 0.30]:
            sc_b = _eval_weights(base_w, oot_norm_elig, flag_arr, bonus=b_val)
            top2k_b = _topk_indices(sc_b, 2000)
            rho_b = float(spearmanr(sc_b, ref_sc_elig).statistic)
            jacc_vs_25_11 = _jaccard(top2k_b, ref_top2k)
            subs_vs_25_11 = len(top2k_b - ref_top2k)
            n_flagged_in_top2k = int(np.sum(flag_arr[np.array(sorted(top2k_b), dtype=int)]))
            mean_shift = float(np.mean(sc_b - _eval_weights(base_w, oot_norm_elig, flag_arr, bonus=0.0)))
            bonus_rows.append({
                "modo_regra": mode_lbl,
                "bonus": b_val,
                "ucs_com_flag_elig": int(flag_arr.sum()),
                "pct_elig_com_flag": round(float(flag_arr.mean() * 100.0), 2),
                "delta_medio_score_vs_b0": round(mean_shift, 4),
                "spearman_vs_ref_25_11": round(rho_b, 4),
                "top2k_jaccard_vs_ref_25_11": round(jacc_vs_25_11, 4),
                "top2k_substituicoes_vs_ref_25_11": subs_vs_25_11,
                "top2k_com_flag": n_flagged_in_top2k,
                "cands_tau_fixo_05131": int(np.sum(sc_b >= tau_frozen)),
                "score_p95": round(float(np.percentile(sc_b, 95)), 4),
            })

    pl.DataFrame(bonus_rows).write_csv(run_dir / "bonus_sensitivity.csv", separator=";")
    pl.DataFrame(bonus_rows).write_csv(P0_DATA_DIR / "bonus_sensitivity.csv", separator=";")

    # =========================================================================
    # SECTION 9: SPECIALIST PARTICIPATION AUDIT MATRIX
    # =========================================================================
    specialist_matrix = [
        {
            "componente": "S_comm (Comunicação)",
            "implementado": "Sim (CommunicationAnomalyDetector: IF + ECOD)",
            "executado_oot_25_11": "Sim (65.686 UCs)",
            "avaliado_isoladamente": "Sim (testes modulares + scores em decisions.parquet)",
            "participa_candidatura_25_11": "Indireto via gate de cobertura >= 70% (não entra na soma de s_elec)",
            "parametros_principais": "IF(100) + ECOD (50/50); threshold_comm=0,6253; gate cov>=0,70",
            "evidencia_disponivel": "decisions.parquet, frozen_multiview_model.joblib, test_specialized_models.py",
        },
        {
            "componente": "S_elec (Ensemble Elétrico 6 det.)",
            "implementado": "Sim (CompositeAnomalyDetector: IF, LOF, ECOD, COPOD, HBOS, PCA)",
            "executado_oot_25_11": "Sim (65.686 UCs; 16 features elétricas puras)",
            "avaliado_isoladamente": "Sim (auditoria individual + estabilidade + correlações)",
            "participa_candidatura_25_11": "Sim (núcleo primário: s_elec >= 0,5131 & is_reliable_elec)",
            "parametros_principais": "Pesos IF=0,22, LOF=0,20, ECOD=0,18, COPOD=0,18, HBOS=0,10, PCA=0,12; tau=0,5131",
            "evidencia_disponivel": "decisions.parquet, detector_individual_audit.csv, weight_sensitivity.csv",
        },
        {
            "componente": "S_behavior (Comportamental)",
            "implementado": "Sim (BehavioralClusterer PCA+KMeans + classes C1–C6 + transições)",
            "executado_oot_25_11": "Sim (cluster_id e CLASSE_COMPORTAMENTAL em decisions.parquet)",
            "avaliado_isoladamente": "Sim (eda_suite + ml_experiments + P1 perturbações)",
            "participa_candidatura_25_11": "Não (apenas caracterização/rotulagem de MUDANCA_CLASSE_COMPORTAMENTAL)",
            "parametros_principais": "PCA(3) + KMeans(4) em X_elec; faixas C1..C6; transição >=2 cortes",
            "evidencia_disponivel": "decisions.parquet (cluster_id, MUDOU_CLASSE_TEMPORAL), P1 behavior",
        },
        {
            "componente": "CPD (CUSUM Mudança de Regime)",
            "implementado": "Sim (_detect_step_change_cusum em src/eda/uc_timeseries.py)",
            "executado_oot_25_11": "Parcial (sob demanda no Cockpit e nos estudos de caso; não no OOT em lote)",
            "avaliado_isoladamente": "Sim (case_auditor + P1 avaliação completa em 65.686 UCs e perturbações)",
            "participa_candidatura_25_11": "Não na campanha 25/11 (integrado na variante experimental P1 V4/V5)",
            "parametros_principais": "cov>=0,70, >=8d confiáveis, margem max(2, n//6), |delta|>=max(1,25*scale, 0,25*|pre|)",
            "evidencia_disponivel": "case_traces, P1 multiview_evidence_oot.parquet e controlled_perturbations",
        },
        {
            "componente": "S_tamper (Eventos Físicos)",
            "implementado": "Sim (TamperAnomalyDetector: alarmes + slope negativo + troca)",
            "executado_oot_25_11": "Sim (calculado em decisions.parquet), porém 0 alarmes na janela OOT de Nov/2026",
            "avaliado_isoladamente": "Sim (test_specialized_models.py; histórico alarmes Jul–Set/2026)",
            "participa_candidatura_25_11": "Regra ativa (s_tamper>=0,70), mas 0 disparos efetivos no OOT (max s_tamper=0,30 só via slope)",
            "parametros_principais": "0,70*clip(alarms/3) + 0,30*clip(-slope/5) + 0,15*I(alarms>0)*troca_30d; limiar 0,70",
            "evidencia_disponivel": "decisions.parquet (s_tamper, x__alarm_count_window=0), output/events/",
        },
        {
            "componente": "PRODIST / Regras Físicas",
            "implementado": "Sim (guardrails em CompositeAnomalyDetector + category_reporting.py)",
            "executado_oot_25_11": "Sim (rule_violation_flag + regras GD/Medidor Travado)",
            "avaliado_isoladamente": "Sim (bonus_sensitivity.csv + test_composite_model.py)",
            "participa_candidatura_25_11": "Sim (+0,15 no consenso em 30.694 UCs — 30.654 via c_imb>0,20, 58 v_imb>20, 66 ra>0,85)",
            "parametros_principais": "Bonus +0,15; GD: ra_ratio>=0,08 & ra_days>=2; Travado: fa_med<0,05 & is_anom",
            "evidencia_disponivel": "decisions.parquet (rule_violation_flag), bonus_sensitivity.csv",
        },
        {
            "componente": "MoE Hierárquico (Alimentador/Global)",
            "implementado": "Sim (MoERouter + HierarchicalModelOrchestrator)",
            "executado_oot_25_11": "Não na campanha 25/11 (modelos v1 eram legado 4-det/21-feat); executado no P1 sem vazamento",
            "avaliado_isoladamente": "Sim (test_moe_router.py + P1 avaliação hierárquica congelada)",
            "participa_candidatura_25_11": "Não na campanha 25/11 (avaliado na variante P1 V3/V5)",
            "parametros_principais": "n>=1000: (0,70/0,20/0,10); 150<=n<1000: interpolação; n<150: (0,15/0,45/0,40)",
            "evidencia_disponivel": "test_moe_router.py, output/models/v1/ (legado), P1 hierarchical OOT",
        },
        {
            "componente": "Contexto Topológico / Vizinhança",
            "implementado": "Sim (contraste espacial em category_reporting.py + spurious_filter.py + funil M3)",
            "executado_oot_25_11": "Sim para Telecom (Isolada vs Gargalo Coletivo) e funil M3; SpuriousFilter avaliado no P1",
            "avaliado_isoladamente": "Sim (test_spurious_filter.py + test_neighborhood.py + P1 eventos coletivos)",
            "participa_candidatura_25_11": "Parcial: separa quarentena telecom e define funil M3 (4.311); 1.726 sem alimentador mantidas no OOT geral",
            "parametros_principais": "Peer Posto>=3 -> Geo 250m -> Feeder; déficit>=0,30 & peer>=0,80; grid_event>=60% peers",
            "evidencia_disponivel": "decisions.parquet (meta__peer_*), ablation_funnel.csv, P1 collective test",
        },
    ]
    pl.DataFrame(specialist_matrix).write_csv(run_dir / "specialist_participation_matrix.csv", separator=";")
    pl.DataFrame(specialist_matrix).write_csv(P0_DATA_DIR / "specialist_participation_matrix.csv", separator=";")

    # =========================================================================
    # P0 FIGURES (1..8)
    # =========================================================================
    figs_generated: dict[str, dict[str, str]] = {}

    # Fig 1: Population Reconciliation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    pop_labels = ["Cadastro\nCIS/GEO", "União\n4 cortes", "Coorte OOT\n(19/11)", "Elegíveis\n(>=70%)", "Com\nAlimentador", "Candidatos\nOOT", "Funil M4/M5\n(Topol.)"]
    pop_vals = [n_cad, n_tr_ucs, n_oot, n_oot_elig, n_oot_topo_m3, n_oot_cand, n_oot_topo_cand]
    colors_p = ["#5c6f7e", "#4b6b8a", "#2b5c8f", "#1f4e79", "#2a7b9b", "#d35400", "#a93226"]
    bars = ax1.bar(range(len(pop_vals)), pop_vals, color=colors_p, width=0.62, alpha=0.92)
    for b, v in zip(bars, pop_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1200, _fmt_br_int(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_xticks(range(len(pop_labels)))
    ax1.set_xticklabels(pop_labels, fontsize=7.8)
    ax1.set_ylim(0, 80000)
    ax1.set_ylabel("Número de UCs")
    ax1.set_title("(A) Reconciliação das populações (25/11/2026)", fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_br_int(x)))
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: OOT vs Daily Rules (19/11) composition
    comp_lbls = ["Exclusivos\nOOT_MULTIVIEW", "Interseção\n(Ambas)", "Exclusivos\nDAILY_RULES"]
    comp_vals = [n_oot_excl, n_inter, n_daily_excl]
    bars2 = ax2.bar(range(3), comp_vals, color=["#1f4e79", "#2e7d32", "#b76e1c"], width=0.55, alpha=0.92)
    for b, v in zip(bars2, comp_vals):
        pct = 100.0 * v / sum(comp_vals)
        ax2.text(b.get_x() + b.get_width() / 2, v + 90, f"{_fmt_br_int(v)}\n({_fmt_br_float(pct, 1)}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(comp_lbls, fontsize=8.5)
    ax2.set_ylim(0, 4800)
    ax2.set_ylabel("Número de UCs candidatas")
    ax2.set_title(f"(B) Composição das carteiras em 19/11 (União = {_fmt_br_int(sum(comp_vals))})", fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_br_int(x)))
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_generated["fig_p0_01"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_01_reconciliacao_populacional")

    # Fig 2: Revised Correlation Matrix (Spearman Eligible Norm vs Pearson Eligible Norm)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for ax, mat, ttl in [
        (axes[0], sp_norm_elig, "(A) Spearman (Elegíveis, n = 62.857)"),
        (axes[1], pe_norm_elig, "(B) Pearson (Elegíveis, n = 62.857)"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(labels_7)))
        ax.set_yticks(range(len(labels_7)))
        ax.set_xticklabels(labels_7, rotation=30, ha="right")
        ax.set_yticklabels(labels_7)
        ax.set_title(ttl, fontweight="bold", pad=10)
        for i in range(len(labels_7)):
            for j in range(len(labels_7)):
                val = mat[i, j]
                txt_c = "white" if abs(val) >= 0.65 else "#111111"
                ax.text(j, i, _fmt_br_float(val, 2), ha="center", va="center", color=txt_c, fontsize=8.2, fontweight="bold" if i == j else "normal")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Coeficiente de correlação")
    fig.suptitle("Auditoria de correlação dos detectores elétricos congelados (corte 19/11/2026)", fontweight="bold", y=1.02)
    figs_generated["fig_p0_02"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_02_correlacao_detectores_revisada")

    # Fig 3: Individual Normalized Score Distributions
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2))
    palette = ["#1f4e79", "#b83b5e", "#2e7d32", "#e67e22", "#6a1b9a", "#00838f"]
    for idx, (ax, lbl, col) in enumerate(zip(axes.ravel(), det_labels, palette)):
        arr = norm_mat[elig_mask, idx]
        ax.hist(arr, bins=40, color=col, alpha=0.82, edgecolor="white", linewidth=0.3)
        p50 = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        ax.axvline(p50, color="#222222", linestyle="--", linewidth=1.1, label=f"P50 = {_fmt_br_float(p50, 2)}")
        ax.axvline(p95, color="#c00000", linestyle=":", linewidth=1.3, label=f"P95 = {_fmt_br_float(p95, 2)}")
        z_pct = dist_rows[idx]["norm_zero_pct"]
        s1_pct = dist_rows[idx]["norm_sat1_pct"]
        ax.set_title(f"{lbl} (zeros={_fmt_br_float(z_pct, 1)}%, sat.1={_fmt_br_float(s1_pct, 1)}%)", fontsize=9.2, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Escore normalizado [0, 1]")
        ax.set_ylabel("Frequência (UCs)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_br_int(x)))
        ax.legend(fontsize=7.5, frameon=True, framealpha=0.88)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.suptitle("Distribuição dos escores normalizados congelados por detector (OOT elegíveis, n = 62.857)", fontweight="bold")
    plt.tight_layout()
    figs_generated["fig_p0_03"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_03_distribuicao_scores_individuais")

    # Fig 4: Frozen-model Gating Sensitivity (contrasting frozen vs historical refit)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    thrs = [r["coverage_threshold_pct"] for r in gating_frozen_rows]
    ret_f = [r["candidates_retained"] for r in gating_frozen_rows]
    int_f = [r["anomalies_intercepted_quarantine"] for r in gating_frozen_rows]
    x_g = np.arange(len(thrs))
    ax1.bar(x_g, ret_f, width=0.52, color="#1f4e79", label="Candidatos retidos (elegíveis)")
    ax1.bar(x_g, int_f, width=0.52, bottom=ret_f, color="#c0392b", label="Sinalizações interceptadas (quarentena)")
    for i in range(len(x_g)):
        ax1.text(x_g[i], ret_f[i] / 2, _fmt_br_int(ret_f[i]), ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")
        ax1.text(x_g[i], ret_f[i] + int_f[i] / 2, f"+{_fmt_br_int(int_f[i])}", ha="center", va="center", color="white", fontsize=7.8, fontweight="bold")
        tot = ret_f[i] + int_f[i]
        ax1.text(x_g[i], tot + 120, _fmt_br_int(tot), ha="center", va="bottom", fontsize=8.2)
    ax1.set_xticks(x_g)
    ax1.set_xticklabels([f"{t}%" for t in thrs])
    ax1.set_ylim(0, 9200)
    ax1.set_xlabel("Limiar mínimo de cobertura")
    ax1.set_ylabel("UCs sinalizadas (modelo congelado, τ = 0,5131)")
    ax1.set_title("(A) Sensibilidade do gating com modelo congelado", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Eligible % and Jaccard vs 70% reference
    elig_pcts = [r["eligible_pct"] for r in gating_frozen_rows]
    jacc_70s = [r["jaccard_cands_vs_ref70"] * 100.0 for r in gating_frozen_rows]
    top2k_70s = [r["top2k_jaccard_vs_ref70"] * 100.0 for r in gating_frozen_rows]
    ax2.plot(thrs, elig_pcts, "o-", color="#1f4e79", linewidth=2, label="UCs elegíveis (% da coorte)")
    ax2.plot(thrs, jacc_70s, "s--", color="#2e7d32", linewidth=2, label="Jaccard candidatos vs. ref. 70% (%)")
    ax2.plot(thrs, top2k_70s, "^:", color="#d35400", linewidth=2, label="Jaccard Top-2.000 vs. ref. 70% (%)")
    for t, ep, jc in zip(thrs, elig_pcts, jacc_70s):
        ax2.annotate(f"{_fmt_br_float(jc, 1)}%", (t, jc), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.8, color="#2e7d32")
    ax2.set_xlabel("Limiar mínimo de cobertura (%)")
    ax2.set_ylabel("Percentual (%)")
    ax2.set_ylim(75, 102)
    ax2.set_xticks(thrs)
    ax2.set_xticklabels([f"{t}%" for t in thrs])
    ax2.set_title("(B) Cobertura populacional e estabilidade vs. ref. 70%", fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_generated["fig_p0_04"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_04_sensibilidade_gating_congelado")

    # Fig 5: Weight Sensitivity (Top-2000 Jaccard & Substitutions + Protocol B fixed threshold)
    w_df = pl.DataFrame(weight_rows)
    sel_names = [
        "Atual (código congelado)", "Tabela histórica doc.", "Pesos uniformes (1/6)",
        "IF +20%", "LOF +20%", "PCA +20%",
        "Sem IF (5 det.)", "Sem LOF (5 det.)", "Sem ECOD (5 det.)",
        "Sem COPOD (5 det.)", "Sem HBOS (5 det.)", "Sem PCA (5 det.)",
        "Núcleo 4 (IF+ECOD+COPOD+HBOS)", "Isolation Forest isolado",
    ]
    w_sub = w_df.filter(pl.col("configuracao").is_in(sel_names))
    order_map = {n: i for i, n in enumerate(sel_names)}
    w_sub = w_sub.with_columns(pl.col("configuracao").replace_strict(order_map).alias("_ord")).sort("_ord")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True)
    y_pos = np.arange(w_sub.height)
    jaccs = w_sub["top2k_jaccard"].to_numpy()
    subs = w_sub["top2k_substituicoes"].to_numpy()
    cands_fix = w_sub["protoB_cands_tau_fixo_05131"].to_numpy()
    cands_cal = w_sub["protoB_cands_tau_calibrado"].to_numpy()

    ax1.barh(y_pos, jaccs, color="#1f4e79", alpha=0.88)
    for yp, jc, sb in zip(y_pos, jaccs, subs):
        ax1.text(jc + 0.01, yp, f"{_fmt_br_float(jc, 3)} ({sb} trocas)", va="center", fontsize=7.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(w_sub["configuracao"].to_list(), fontsize=8.3)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.28)
    ax1.set_xlabel("Jaccard Top-2.000 vs. configuração atual (Protocolo A)")
    ax1.set_title("(A) Estabilidade de carteira sob orçamento fixo (K = 2.000)", fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_br_float(x, 1)))
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    h = 0.36
    ax2.barh(y_pos - h / 2, cands_fix, height=h, color="#d35400", alpha=0.88, label="Limiar fixo τ = 0,5131")
    ax2.barh(y_pos + h / 2, cands_cal, height=h, color="#2e7d32", alpha=0.85, label="Limiar recalibrado no treino (P95)")
    ax2.axvline(n_oot_cand, color="#333333", linestyle="--", linewidth=1.0, label=f"Ref. OOT ({_fmt_br_int(n_oot_cand)})")
    ax2.set_xlabel("Número de candidatos sinalizados (Protocolo B)")
    ax2.set_title("(B) Sensibilidade volumétrica à escala do escore", fontweight="bold")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_br_int(x)))
    fig.legend(*ax2.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, fontsize=8, frameon=False)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs_generated["fig_p0_05"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_05_sensibilidade_pesos")

    # Fig 6: Corroboration Bonus Sensitivity (+0.15)
    b_df = pl.DataFrame(bonus_rows)
    b_eff = b_df.filter(pl.col("modo_regra") == "regra_efetiva_25_11_incl_c_imb")
    b_str = b_df.filter(pl.col("modo_regra") == "regra_estrita_tensao_e_reversao")
    b_vals = b_eff["bonus"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))
    ax1.plot(b_vals, b_eff["cands_tau_fixo_05131"].to_numpy(), "o-", color="#c0392b", linewidth=2, label="Regra efetiva 25/11 (incl. c_imb > 0,20; 46,7% UCs)")
    ax1.plot(b_vals, b_str["cands_tau_fixo_05131"].to_numpy(), "s--", color="#1f4e79", linewidth=2, label="Regra estrita tensão/reversão (0,19% UCs)")
    for xv, yv in zip(b_vals, b_eff["cands_tau_fixo_05131"].to_numpy()):
        ax1.annotate(_fmt_br_int(yv), (xv, yv), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#c0392b")
    ax1.axvline(0.15, color="#555555", linestyle=":", linewidth=1.1, label="Coeficiente padrão (0,15)")
    ax1.set_xlabel("Coeficiente de bônus de regra física")
    ax1.set_ylabel("UCs com escore >= 0,5131 (Elegíveis)")
    ax1.set_title("(A) Efeito do bônus no volume sob limiar fixo", fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_br_int(x)))
    ax1.legend(fontsize=7.8, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot(b_vals, b_eff["top2k_substituicoes_vs_ref_25_11"].to_numpy(), "o-", color="#c0392b", linewidth=2, label="Substituições Top-2.000 (regra efetiva)")
    ax2.plot(b_vals, b_str["top2k_substituicoes_vs_ref_25_11"].to_numpy(), "s--", color="#1f4e79", linewidth=2, label="Substituições Top-2.000 (regra estrita)")
    for xv, yv in zip(b_vals, b_eff["top2k_substituicoes_vs_ref_25_11"].to_numpy()):
        ax2.annotate(_fmt_br_int(yv), (xv, yv), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#c0392b")
    ax2.axvline(0.15, color="#555555", linestyle=":", linewidth=1.1)
    ax2.set_xlabel("Coeficiente de bônus de regra física")
    ax2.set_ylabel("Substituições no Top-2.000 vs. ref. 25/11")
    ax2.set_title("(B) Impacto na ordenação Top-2.000", fontweight="bold")
    ax2.legend(fontsize=7.8, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_generated["fig_p0_06"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_06_variacao_bonus_corroboracao")

    # Fig 7: Top-K Portfolio Comparison across Key Weight Variants
    tc_df = pl.DataFrame(topk_cat_rows)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    xc = np.arange(tc_df.height)
    p1 = tc_df["PNT_Eletrica"].to_numpy()
    p2 = tc_df["Possivel_GD"].to_numpy()
    p3 = tc_df["Falha_Metrologica"].to_numpy()
    p4 = tc_df["Outras"].to_numpy()
    ax.bar(xc, p1, width=0.58, label="Anomalia Elétrica / PNT", color="#1f4e79")
    ax.bar(xc, p2, width=0.58, bottom=p1, label="Possível GD não cadastrada", color="#2e7d32")
    ax.bar(xc, p3, width=0.58, bottom=p1 + p2, label="Falha Metrológica / Travado", color="#d35400")
    ax.bar(xc, p4, width=0.58, bottom=p1 + p2 + p3, label="Outras (abaixo do limiar OOT)", color="#95a5a6")
    ax.set_xticks(xc)
    ax.set_xticklabels(tc_df["configuracao"].to_list(), rotation=18, ha="right", fontsize=8.3)
    ax.set_ylim(0, 2350)
    ax.set_ylabel("Composição do Top-2.000 (UCs)")
    ax.set_title("Composição diagnóstica das carteiras Top-2.000 sob diferentes pesos do ensemble", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_generated["fig_p0_07"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_07_comparacao_carteiras_topk")

    # Fig 8: Specialist Participation Matrix Visual Heatmap
    comp_short = [r["componente"].split(" (")[0] for r in specialist_matrix]
    cols_m = ["Implementado", "Executado\nOOT 25/11", "Avaliado\nIsoladamente", "Participa\nCand. 25/11", "Integrado\nVariante P1"]
    # 1.0 = Full, 0.5 = Partial/Indirect, 0.0 = No
    mat_sp = np.array([
        [1.0, 1.0, 1.0, 0.5, 1.0],  # S_comm
        [1.0, 1.0, 1.0, 1.0, 1.0],  # S_elec
        [1.0, 1.0, 1.0, 0.0, 1.0],  # S_behavior
        [1.0, 0.5, 1.0, 0.0, 1.0],  # CPD
        [1.0, 0.5, 1.0, 0.0, 0.5],  # S_tamper (0 alarmes em Nov)
        [1.0, 1.0, 1.0, 1.0, 1.0],  # PRODIST
        [1.0, 0.0, 1.0, 0.0, 1.0],  # MoE
        [1.0, 0.5, 1.0, 0.5, 1.0],  # Topologia
    ])
    txt_sp = [
        ["Sim", "Sim", "Sim", "Indireto\n(gate)", "Sim"],
        ["Sim", "Sim", "Sim", "Sim\n(núcleo)", "Sim"],
        ["Sim", "Sim", "Sim", "Não", "Sim"],
        ["Sim", "Parcial\n(casos)", "Sim", "Não", "Sim"],
        ["Sim", "Parcial\n(0 alarmes)", "Sim", "0 disparos\n(OOT)", "Parcial\n(sem alarmes)"],
        ["Sim", "Sim", "Sim", "Sim\n(+0,15)", "Sim"],
        ["Sim", "Não\n(v1 legado)", "Sim", "Não", "Sim\n(re-treino limpo)"],
        ["Sim", "Parcial\n(M3/Telecom)", "Sim", "Parcial\n(funil M3)", "Sim"],
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    cmap_sp = matplotlib.colors.ListedColormap(["#f2d7d5", "#fcf3cf", "#d4efdf"])
    ax.imshow(mat_sp, cmap=cmap_sp, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols_m)))
    ax.set_xticklabels(cols_m, fontweight="bold")
    ax.set_yticks(range(len(comp_short)))
    ax.set_yticklabels(comp_short, fontweight="bold")
    for i in range(len(comp_short)):
        for j in range(len(cols_m)):
            ax.text(j, i, txt_sp[i][j], ha="center", va="center", fontsize=8.0, color="#1a1a1a")
    ax.set_xticks(np.arange(-0.5, len(cols_m), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(comp_short), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Matriz auditada de participação experimental dos especialistas (25/11 vs. P1)", fontweight="bold", pad=12)
    plt.tight_layout()
    figs_generated["fig_p0_08"] = _save_fig(fig, P0_FIG_DIR / "fig_p0_08_matriz_participacao_especialistas")

    # =========================================================================
    # SAVE MANIFEST & METRICS
    # =========================================================================
    summary = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp": datetime.now().isoformat(),
        "runtime_sec": round(time.time() - t0, 2),
        "source_suite": suite,
        "reconciliation": {
            "n_cad": n_cad,
            "n_hier": n_hier,
            "n_tr_rows": n_tr_rows,
            "n_tr_ucs": n_tr_ucs,
            "n_train_clean": n_train_clean,
            "n_oot": n_oot,
            "n_oot_elig": n_oot_elig,
            "n_oot_quar": n_oot_quar,
            "n_oot_clean_flag": n_oot_clean_flag,
            "clean_only_not_rel": int(clean_only.height),
            "rel_only_not_clean": int(rel_only.height),
            "n_oot_cand": n_oot_cand,
            "n_oot_elec_anom": n_oot_elec_anom,
            "n_oot_is_anom": n_oot_is_anom,
            "n_oot_topo_m3": n_oot_topo_m3,
            "n_oot_topo_cand": n_oot_topo_cand,
            "n_daily_win": n_daily_win,
            "mean_daily_cand": mean_daily_cand,
            "n_daily_19_cand": n_daily_19_cand,
            "n_inter_oot_daily": n_inter,
            "n_oot_excl": n_oot_excl,
            "n_daily_excl": n_daily_excl,
            "gd16_outside_queue": gd16_summary,
            "ref70_exact_match_oot_cands": (ref_70_cands == oot_cand_idx),
            "ref70_diff_count": len(ref_70_cands ^ oot_cand_idx),
        },
        "correlation_audit": corr_checks,
        "frozen_model_weights": base_w,
        "frozen_threshold_elec": round(tau_frozen, 6),
        "rule_violation_breakdown_oot": {
            "total_flagged_all": int(oot_dec["rule_violation_flag"].sum()),
            "total_flagged_elig": int(oot_rule_elig.sum()),
            "v_imb_gt_20_elig": int((v_imb_e > 20.0).sum()),
            "c_imb_gt_020_elig": int((c_imb_e > 0.20).sum()),
            "ra_ratio_gt_085_elig": int((ra_r_e > 0.85).sum()),
            "strict_v_or_ra_elig": int(strict_prodist_flag_e.sum()),
        },
        "gating_frozen_summary": gating_frozen_rows,
        "weight_sensitivity_count": len(weight_rows),
        "bonus_sensitivity_summary": bonus_rows,
        "figures": figs_generated,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (P0_DATA_DIR / "p0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "pipeline_id": "p0_audit_sensitivity",
        "git_commit": commit,
        "status": "validated",
        "created_at": datetime.now().isoformat(),
        "inputs": {k: v for k, v in suite.items()},
        "artifacts": {
            f.name: {"path": str(f.relative_to(PROJECT_ROOT)), "sha256": _sha256(f)}
            for f in sorted(run_dir.iterdir()) if f.is_file()
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_p0_audit()
    print(f"P0 Audit completed in {res['runtime_sec']}s | run_id={res['run_id']}")
    print("Reconciliation:", json.dumps(res["reconciliation"], indent=2, ensure_ascii=False)[:1200])
    print("Correlation checks:", json.dumps(res["correlation_audit"], indent=2))
    print("Rule violation breakdown:", json.dumps(res["rule_violation_breakdown_oot"], indent=2))
