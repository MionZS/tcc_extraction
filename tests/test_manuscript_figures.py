"""Verificação automatizada das figuras do manuscrito, proveniência LGPD e reconciliação de KPIs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import polars as pl
import pytest

from src.eda.cockpit_metrics import compute_reconciled_kpis, filter_cockpit_dataframe
from src.utils.censorship import build_feeder_anonymizer, censor_nio_id, censor_uc_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = PROJECT_ROOT / "docs" / "manuscript_figures"
SUITE_INDEX_PATH = PROJECT_ROOT / "experiments" / "runs" / "latest_suite.json"


def test_manuscript_figure_assets_and_mirrors_exist() -> None:
    """Garante existência e integridade não nula de todas as figuras, metadados, CSVs e snippets LaTeX."""
    expected_assets = [
        # Plataforma (Seção 4.4)
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4a_tui_menu_governanca.svg", 1000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4a_tui_menu_governanca.pdf", 1000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4a_tui_menu_governanca.png", 5000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4a_tui_menu_governanca.meta.json", 200),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4b_tui_calendario_auditoria.svg", 1000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4b_tui_calendario_auditoria.pdf", 1000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4b_tui_calendario_auditoria.png", 5000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4b_tui_calendario_auditoria.meta.json", 200),
        (MANUSCRIPT_DIR / "platform_interfaces" / "cockpit_visao_geral.png", 5000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4c_cockpit_carteira_candidatos.png", 5000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4c_cockpit_carteira_candidatos.meta.json", 200),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4d_cockpit_raiox_uc_908xxx00.png", 5000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4d_cockpit_raiox_uc_908xxx00_chart.svg", 1000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "fig_4_4d_cockpit_raiox_uc_908xxx00.meta.json", 200),
        (MANUSCRIPT_DIR / "platform_interfaces" / "cockpit_raiox_uc_685xxx20.png", 5000),
        (MANUSCRIPT_DIR / "platform_interfaces" / "cockpit_raiox_uc_291xxx96.png", 5000),
        # Painéis ML controlados (Seção 5.1)
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1a_isolation_forest_microcluster.png", 5000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1a_isolation_forest_microcluster.svg", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1a_isolation_forest_microcluster.pdf", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1a_isolation_forest_microcluster_data.csv", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1a_isolation_forest_microcluster.meta.json", 200),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1b_lof_escala_propria.png", 5000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1b_lof_escala_propria.svg", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1b_lof_escala_propria.pdf", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1b_lof_escala_propria_data.csv", 150),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1b_lof_escala_propria.meta.json", 200),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1c_contextualizacao_percentis.png", 5000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1c_contextualizacao_percentis.svg", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1c_contextualizacao_percentis.pdf", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1c_contextualizacao_percentis_data.csv", 1000),
        (MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1c_contextualizacao_percentis.meta.json", 200),
        # Estudos de Caso Reais
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_908xxx00_fa_ra_completo.png", 5000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_908xxx00_fa_ra_completo.svg", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_908xxx00_fa_ra_completo.pdf", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_908xxx00_fa_ra_completo_data.csv", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_908xxx00_fa_ra_completo.meta.json", 200),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores.png", 5000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores.svg", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores.pdf", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores_data.csv", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores.meta.json", 200),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_685xxx20_reconciliacao_cobertura.png", 5000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_685xxx20_reconciliacao_cobertura.svg", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_685xxx20_reconciliacao_cobertura.pdf", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_685xxx20_reconciliacao_cobertura_data.csv", 1000),
        (MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_685xxx20_reconciliacao_cobertura.meta.json", 200),
        # LaTeX e Catálogo
        (MANUSCRIPT_DIR / "latex_snippets" / "sec_4_4_interfaces_reais.tex", 300),
        (MANUSCRIPT_DIR / "latex_snippets" / "sec_5_1_demonstracoes_ml.tex", 300),
        (MANUSCRIPT_DIR / "latex_snippets" / "sec_estudos_caso_reconciliados.tex", 300),
        (MANUSCRIPT_DIR / "README_INCORPORACAO_LATEX.md", 200),
        (MANUSCRIPT_DIR / "figure_catalog_manifest.json", 1000),
        # Espelhos no diretório do artigo
        (PROJECT_ROOT / "docs" / "figures" / "ml_method_demo" / "05_isolation_forest.png", 5000),
        (PROJECT_ROOT / "docs" / "figures" / "ml_method_demo" / "06_lof.png", 5000),
        (PROJECT_ROOT / "docs" / "figures" / "ml_method_demo" / "09_ablation_funnel.png", 5000),
        (PROJECT_ROOT / "docs" / "figures" / "eda" / "exemplos" / "acompanhamento_semanal_outlier.png", 5000),
    ]

    for fpath, min_bytes in expected_assets:
        assert fpath.exists(), f"Artefato esperado ausente: {fpath}"
        assert fpath.stat().st_size >= min_bytes, f"Artefato suspeitamente pequeno ({fpath.stat().st_size} bytes): {fpath}"


def test_lgpd_anonymization_across_generated_manuscript_artifacts() -> None:
    """Varre .svg, .csv, .json, .tex e .md em docs/manuscript_figures contra vazamento de IDs e nomes reais."""
    assert censor_uc_id("90850700") == "908***00"
    assert censor_nio_id("0043570463") == "NIO-435***63"
    fmap = build_feeder_anonymizer(["Polonia", "Oleos Brasil"])
    assert all("circuito f-" in v.lower() for v in fmap.values())

    forbidden_tokens = [
        "90850700",
        "29113296",
        "68546220",
        "90807600",
        "43570463",
        "42312054",
        "43703386",
        "copel",
        "araucaria",
        "araucária",
        "oleos brasil",
        "óleos brasil",
        "polonia",
        "polônia",
        "fonte nova",
        "csn",
    ]
    text_exts = {".svg", ".csv", ".json", ".tex", ".md"}
    scanned = 0
    for fpath in MANUSCRIPT_DIR.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in text_exts:
            scanned += 1
            content = fpath.read_text(encoding="utf-8", errors="ignore").lower()
            for tok in forbidden_tokens:
                assert tok not in content, f"Vazamento LGPD detectado ({tok}) em {fpath}"
    assert scanned >= 20


def test_oot_cockpit_kpi_reconciliation_matches_2829_quarantine() -> None:
    """Valida reconciliação exata dos KPIs do Cockpit sobre o artifact OOT de latest_suite.json."""
    suite_idx = json.loads(SUITE_INDEX_PATH.read_text(encoding="utf-8"))
    oot_df = pl.read_parquet(PROJECT_ROOT / suite_idx["oot_multiview"] / "decisions.parquet")
    kpis = compute_reconciled_kpis(oot_df)

    assert kpis["total_unique_ucs"] == 65686
    assert kpis["candidatos_vistoria_ucs"] == 6037
    assert kpis["possivel_gd_ucs"] == 1451
    assert kpis["quarentena_telecom_ucs"] == 2829

    q_df, _ = filter_cockpit_dataframe(oot_df, granularity="unique_ucs", category="TELECOM_ALL")
    assert q_df.height == 2829


def test_real_case_and_controlled_ml_numerical_invariants() -> None:
    """Valida invariantes matemáticos nos CSVs dos casos reais e dos 3 painéis ML controlados."""
    # 1. Reconciliação exata UC 685***20: 50,0%, 100,0% e 99,49%
    csv_685 = MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_685xxx20_reconciliacao_cobertura_data.csv"
    with open(csv_685, "r", encoding="utf-8") as f:
        rows_685 = list(csv.DictReader(f, delimiter=";"))
    assert len(rows_685) == 198  # 168 horas + 30 dias
    hourly_rows = [r for r in rows_685 if r["granularity"] == "hourly_168h"]
    assert len(hourly_rows) == 168
    assert all(float(r["hourly_5min_grid_cov"]) == pytest.approx(0.5, abs=1e-4) for r in hourly_rows)
    assert all(float(r["hourly_10min_cadence_cov"]) == pytest.approx(1.0, abs=1e-4) for r in hourly_rows)
    assert any(abs(float(r["window_30d_cov"]) - 0.99491) < 1e-4 for r in rows_685 if r["window_30d_cov"])

    # 2. UC 291***96: 314 leituras empíricas em ami_registers
    csv_291 = MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores_data.csv"
    with open(csv_291, "r", encoding="utf-8") as f:
        rows_291 = list(csv.DictReader(f, delimiter=";"))
    assert len(rows_291) == 314
    meta_291 = json.loads((MANUSCRIPT_DIR / "real_case_studies" / "fig_caso_291xxx96_horarios_registradores.meta.json").read_text(encoding="utf-8"))
    assert meta_291["origin_data"]["ami_registers_rows"] == 314

    # 3. Seção 5.1 — Masking no Isolation Forest e resgate pelo LOF em escala própria
    csv_if = MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1a_isolation_forest_microcluster_data.csv"
    with open(csv_if, "r", encoding="utf-8") as f:
        rows_if = list(csv.DictReader(f, delimiter=";"))
    mc_if_scores = [float(r["if_norm_score"]) for r in rows_if if r["role"] == "Microagrupamento (n=18)"]
    p95_if = float(rows_if[0]["p95_threshold"])
    assert len(mc_if_scores) == 18
    assert statistics.median(mc_if_scores) < p95_if

    csv_lof = MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1b_lof_escala_propria_data.csv"
    with open(csv_lof, "r", encoding="utf-8") as f:
        rows_lof = {r["role"]: r for r in csv.DictReader(f, delimiter=";")}
    assert float(rows_lof["Outlier Local"]["lof_raw_median"]) > float(rows_lof["Outlier Local"]["p95_lof_raw"])
    assert float(rows_lof["Microagrupamento (n=18)"]["lof_raw_median"]) > float(rows_lof["Microagrupamento (n=18)"]["p95_lof_raw"])

    # 4. Contextualização em percentis estritamente em [0, 100]%
    csv_ctx = MANUSCRIPT_DIR / "ml_controlled_panels" / "fig_5_1c_contextualizacao_percentis_data.csv"
    with open(csv_ctx, "r", encoding="utf-8") as f:
        rows_ctx = list(csv.DictReader(f, delimiter=";"))
    assert len(rows_ctx) == 1000
    assert all(0.0 <= float(r["pct_global"]) <= 100.0 for r in rows_ctx)
    assert all(0.0 <= float(r["pct_local"]) <= 100.0 for r in rows_ctx)
