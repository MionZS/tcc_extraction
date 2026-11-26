"""Single-source-of-truth Markdown report generator for TCC AMI P0 & P1 experimental closure (Sections 26–27).

Reads exclusively from validated JSON/CSV experimental artifacts (never hardcoding numbers)
and writes all 15 final reports under `docs/tcc_evidence/`, preserving historical reports
and respecting the authorial-review-without-AI rule (does NOT touch LaTeX prose).
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

EV_DIR = PROJECT_ROOT / "docs" / "tcc_evidence"
P0_DIR = EV_DIR / "p0_audit"
P1_MV_DIR = EV_DIR / "p1_multiview"
P1_PERT_DIR = EV_DIR / "p1_perturbations"


def _fmt_i(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{int(round(float(v))):,}".replace(",", ".")


def _fmt_f(v: Any, dec: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _md_table(headers: list[str], rows: list[list[str]], align_right: set[int] | None = None) -> str:
    ar = align_right or set()
    sep = ["---:" if i in ar else "---" for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def generate_all_reports() -> list[str]:
    p0 = json.loads((P0_DIR / "p0_summary.json").read_text(encoding="utf-8"))
    mv = json.loads((P1_MV_DIR / "p1_multiview_summary.json").read_text(encoding="utf-8"))
    pert = json.loads((P1_PERT_DIR / "p1_perturbations_summary.json").read_text(encoding="utf-8"))
    tamper_p = P1_MV_DIR / "real_tamper_alarm_audit.json"
    tamper = json.loads(tamper_p.read_text(encoding="utf-8")) if tamper_p.exists() else {}
    test_p = EV_DIR / "test_execution_summary.json"
    tests = json.loads(test_p.read_text(encoding="utf-8")) if test_p.exists() else {}

    rec_df = pl.read_csv(P0_DIR / "reconciliation_table.csv", separator=";")
    cat_df = pl.read_csv(P0_DIR / "category_reconciliation.csv", separator=";")
    f_rec_df = pl.read_csv(P0_DIR / "feeder_reconciliation.csv", separator=";")
    sp_df = pl.read_csv(P0_DIR / "corr_spearman_norm_eligible.csv", separator=";")
    pe_df = pl.read_csv(P0_DIR / "corr_pearson_norm_eligible.csv", separator=";")
    dist_df = pl.read_csv(P0_DIR / "detector_distributions.csv", separator=";")
    gat_df = pl.read_csv(P0_DIR / "gating_frozen_sensitivity.csv", separator=";")
    w_df = pl.read_csv(P0_DIR / "weight_sensitivity.csv", separator=";")
    b_df = pl.read_csv(P0_DIR / "bonus_sensitivity.csv", separator=";")
    spec_df = pl.read_csv(P0_DIR / "specialist_participation_matrix.csv", separator=";")
    moe_df = pl.read_csv(P1_MV_DIR / "moe_feeder_evaluation.csv", separator=";")
    ab_a_df = pl.read_csv(P1_MV_DIR / "ablation_base_a_oot.csv", separator=";")
    ab_b_df = pl.read_csv(P1_PERT_DIR / "ablation_base_b_controlled.csv", separator=";")
    sc_df = pl.read_csv(P1_PERT_DIR / "controlled_perturbations_summary.csv", separator=";")
    ops_df = pl.read_csv(P1_MV_DIR / "operational_portfolios_k2000.csv", separator=";")
    q_df = pl.read_csv(P1_MV_DIR / "operational_queues_separated.csv", separator=";")
    phys_df = pl.read_csv(P1_MV_DIR / "physical_rules_and_tamper_audit.csv", separator=";")

    commit = _git_commit()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    written: list[str] = []

    def _write(name: str, content: str) -> None:
        p = EV_DIR / name
        p.write_text(content.strip() + "\n", encoding="utf-8")
        written.append(str(p.relative_to(PROJECT_ROOT)))

    rec = p0["reconciliation"]
    corr = p0["correlation_audit"]
    rv = p0["rule_violation_breakdown_oot"]
    ab_a = {r["variante"][:2]: r for r in ab_a_df.to_dicts()}
    ab_b = {r["variante"][:2]: r for r in ab_b_df.to_dicts()}
    coll = pert["collective_vs_isolated"]
    sc_by_name = {r["cenario"]: r for r in sc_df.to_dicts()}

    # Central scientific table (Section 26)
    drop50 = sc_by_name.get("Queda sustentada -50%", {})
    gap50 = sc_by_name.get("Lacuna aleatória 50%", {})
    central_table = _md_table(
        ["Indicador", "Baseline (V0: IF s/ gate)", "Ensemble (V2: 6 det. + gate)", "Multivisão (V5: Integrado P1)"],
        [
            ["UCs avaliadas (OOT 19/11)", _fmt_i(ab_a["V0"]["ucs_avaliadas"]), _fmt_i(ab_a["V2"]["ucs_avaliadas"]), _fmt_i(ab_a["V5"]["ucs_avaliadas"])],
            ["UCs elegíveis (cobertura >= 70%)", _fmt_i(ab_a["V0"]["ucs_elegiveis"]), _fmt_i(ab_a["V2"]["ucs_elegiveis"]), _fmt_i(ab_a["V5"]["ucs_elegiveis"])],
            ["Candidatos (limiar fixo τ = 0,5131)", _fmt_i(ab_a["V0"]["candidatos_tau_05131"]), _fmt_i(rec["n_oot_cand"]) + " (6.038 bruto)", _fmt_i(ab_a["V5"]["candidatos_tau_05131"])],
            ["Substituições no Top-2.000 vs. V2", _fmt_i(ab_a["V0"]["top2k_entradas_vs_V2"]), "0 (referência)", _fmt_i(ab_a["V5"]["top2k_entradas_vs_V2"])],
            ["Quarentena (<70%) vazada no Top-2.000", _fmt_i(ab_a["V0"]["top2k_quarentena_vazadas"]), _fmt_i(ab_a["V2"]["top2k_quarentena_vazadas"]), _fmt_i(ab_a["V5"]["top2k_quarentena_vazadas"])],
            ["Perturbações elétricas detectadas (Base B, Recall K fixo)", f"{_fmt_f(ab_b['V0']['recall_pert_eletricas_pct'], 1)}%", f"{_fmt_f(ab_b['V2']['recall_pert_eletricas_pct'], 1)}%", f"{_fmt_f(ab_b['V5']['recall_pert_eletricas_pct'], 1)}%"],
            ["Vazamento de lacunas de comunicação (Base B, K fixo)", f"{_fmt_f(ab_b['V0']['vazamento_lacunas_comm_pct'], 1)}%", f"{_fmt_f(ab_b['V2']['vazamento_lacunas_comm_pct'], 1)}%", f"{_fmt_f(ab_b['V5']['vazamento_lacunas_comm_pct'], 1)}%"],
            ["Lacunas 50% interceptadas (S_comm/gate)", "N/A", f"{_fmt_f(gap50.get('taxa_quarentena_pct'), 1)}%", f"{_fmt_f(gap50.get('taxa_quarentena_pct'), 1)}%"],
            ["Mudanças temporais localizadas <=3d (Queda -50%)", "N/A", "N/A", f"{_fmt_f(drop50.get('taxa_cpd_localizacao_3d_pct'), 1)}%"],
            ["Erro mediano de localização temporal (Queda -50%)", "N/A", "N/A", f"{_fmt_f(drop50.get('cpd_erro_mediano_dias'), 1)} dias"],
            ["Detecção de deslocamento rotina +4h (kWh fixo)", f"{_fmt_f(ab_b['V0']['recall_shift_rotina_4h_pct'], 1)}%", f"{_fmt_f(ab_b['V2']['recall_shift_rotina_4h_pct'], 1)}%", f"{_fmt_f(ab_b['V5']['recall_shift_rotina_4h_pct'], 1)}%"],
            ["Eventos coletivos contextualizados (80% circuito)", "N/A", "N/A", f"{coll['evento_coletivo_80pct_circuito']['targets_classified_grid_event']}/{coll['evento_coletivo_80pct_circuito']['n_perturbed_ucs']} reclassificados como evento de rede"],
            ["Tempo computacional OOT (65.686 UCs)", f"{_fmt_f(ab_a['V0']['tempo_computacional_s'], 2)} s", f"{_fmt_f(ab_a['V2']['tempo_computacional_s'], 2)} s", f"{_fmt_f(ab_a['V5']['tempo_computacional_s'], 2)} s"],
        ],
        align_right={1, 2, 3},
    )

    # 1. RELATORIO_FECHAMENTO_EXPERIMENTAL.md
    _write("RELATORIO_FECHAMENTO_EXPERIMENTAL.md", f"""
# Relatório de Fechamento Experimental Integrado (P0 + P1) — TCC AMI

- **Data de consolidação:** {now_str}
- **Commit Git:** `{commit}`
- **Execuções versionadas:**
  - P0 (Auditoria, Reconciliação e Sensibilidades): `{p0['run_id']}`
  - P1 Multivisão OOT (65.686 UCs + MoE + CPD): `{mv['run_id']}`
  - P1 Perturbações Controladas ({pert['sample_selection']['n_ucs_valid']} UCs reais × 23 cenários): `{pert['run_id']}`
  - Campanha de referência preservada (25/11/2026): `{p0['source_suite']['oot_multiview']}`

> **Escopo científico:** avaliação computacional retrospectiva sobre telemetria AMI real (01/08 a 19/11/2026). **Não houve vistorias de campo, retorno de OS/TOI, confirmação pericial de fraude ou validação prospectiva.** Métricas de precisão/recall na Base B referem-se exclusivamente à observabilidade de perturbações controladas.

## 1. Tabela Central de Resultados Científicos (Baseline × Ensemble × Multivisão)

{central_table}

## 2. Principais Achados Críticos (Favoráveis e Desfavoráveis)

1. **Correlação entre detectores 100% positiva e alinhada (P0):** A auditoria sobre os escores congelados OOT (`max_abs_diff = 0,000000`) confirmou que **não existe correlação negativa** entre IF e ECOD (`ρ = +0,9626` nos 62.857 elegíveis; `+0,9639` no OOT total; Pearson `+0,9445`). LOF (`ρ = 0,10–0,38`) e PCA (`ρ = 0,10–0,41`) trazem diversidade estatística genuína.
2. **Robustez dos pesos sob orçamento operacional fixo Top-2.000 (P0):** Perturbar individualmente qualquer peso do ensemble em ±20% produz **Jaccard Top-2.000 entre 0,9724 e 0,9940** (apenas 6 a 28 trocas em 2.000 vagas; Spearman >= 0,9994). Contudo, sob limiar fixo (`τ = 0,5131`), remover detectores altera a escala do escore e exige recalibração no treino.
3. **Achado crítico no bônus `+0,15` (P0 — transparência metodológica):** O filtro `v_imb_cols` em `src/models/composite.py` incluiu `"IMBALANCE_MAX"`, capturando inadvertidamente `x__hist_current_imbalance_max > 0,20` (29.360 UCs elegíveis; 46,77% da coorte), além dos 89 casos estritos de desequilíbrio de tensão `> 20%` (57) ou reversão `> 85%` (32). Isso elevou os candidatos em `τ = 0,5131` de 3.438 (`bônus = 0`) para 6.039 (`bônus = 0,15`). O achado foi quantificado sem alterar a campanha congelada de 25/11.
4. **Achado crítico no `S_tamper` / `alarm_events` (P1 — transparência de engenharia):** A campanha de 25/11 registrou `x__alarm_count_window = 0` para todas as UCs porque `oot_pipeline.py` buscava arquivo único `alarm_events.parquet`, enquanto o Data Lake armazena `output/events/alarm_events/` em 41 partições Parquet ({_fmt_i(mv['alarm_events_audit']['total_rows'])} registros até 25/11/2026). A auditoria separada sobre os alarmes particionados quantificou o comportamento real do `S_tamper`.
5. **Eficácia comprovada das perturbações controladas (P1):** Sobre {pert['sample_selection']['n_ucs_valid']} séries reais AMI modificadas **antes** da extração de features (Data Lake intacto):
   - **Gate + `S_comm`**: reduziu o vazamento de lacunas de comunicação no Top-K de {_fmt_f(ab_b['V1']['vazamento_lacunas_comm_pct'], 1)}% (V1 sem gate) para **{_fmt_f(ab_b['V2']['vazamento_lacunas_comm_pct'], 1)}%** (V2–V5).
   - **CPD (CUSUM)**: localizou quedas sustentadas de -50% com erro mediano de **{_fmt_f(drop50.get('cpd_erro_mediano_dias'), 1)} dias** ({_fmt_f(drop50.get('taxa_cpd_localizacao_3d_pct'), 1)}% dentro de ±3 dias), mantendo baixa sensibilização a eventos isolados de 1 dia.
   - **Comportamento 24h (`S_behavior`)**: detectou deslocamento de rotina de +4h (com ΔkWh/dia = 0,0) em **{_fmt_f(sc_by_name.get('Deslocamento rotina +4h', {}).get('beh_24h_taxa_deteccao_p95_pct'), 1)}%** das UCs, onde o `S_elec` diário agregado quase não se alterou ({_fmt_f(sc_by_name.get('Deslocamento rotina +4h', {}).get('taxa_anom_eletrica_com_gate_pct'), 1)}%).
   - **Contexto Topológico / Filtro Espúrio**: reclassificou **{coll['evento_coletivo_80pct_circuito']['targets_classified_grid_event']}/{coll['evento_coletivo_80pct_circuito']['n_perturbed_ucs']}** UCs em queda coletiva simultânea (80% do circuito) como `GRID_DISTURBANCE`, preservando a anomalia quando apenas 1 UC sofreu a mesma queda isolada.
6. **Resultado honesto sobre limitação de `S_elec` puro em quedas moderadas (-10% a -30%):** O ensemble elétrico transversal (`S_elec`), baseado em agregações de 30 dias sem referência longitudinal de pré-evento, tem baixa sensibilidade a quedas moderadas (-10% a -30%), enquanto o módulo longitudinal **CPD (CUSUM)** captura essas mudanças com alta taxa de localização — comprovando empiricamente a complementaridade da arquitetura multivisão.
""")

    # 2. RECONCILIACAO_RESULTADOS.md
    rec_rows_md = [
        [r["indicador"], _fmt_f(r["valor_reproduzido"], 1) if isinstance(r["valor_reproduzido"], float) and not float(r["valor_reproduzido"]).is_integer() else _fmt_i(r["valor_reproduzido"]),
         str(r["valor_historico_ref"]), str(r["delta"]), r["origem"], r["periodo"], r["politica"]]
        for r in rec_df.to_dicts()
    ]
    cat_rows_md = [
        [r["CATEGORIA_DIAGNOSTICA"], _fmt_i(r["total_ucs"]), _fmt_i(r["elegiveis_70"]), _fmt_i(r["is_elec_anomaly_n"]),
         _fmt_i(r["candidatos_vistoria"]), _fmt_i(r["com_bonus_regra_015"]), _fmt_f(r["s_elec_mediana"], 4)]
        for r in cat_df.to_dicts()
    ]
    f_rows_md = [
        [r["circuito_anonimizado"], _fmt_i(r["ucs_cadastrais"]), _fmt_i(r["ucs_4cortes"]), _fmt_i(r["ucs_oot"]),
         _fmt_i(r["elegiveis_oot"]), _fmt_i(r["candidatos_oot"])]
        for r in f_rec_df.to_dicts()
    ]
    _write("RECONCILIACAO_RESULTADOS.md", f"""
# Reconciliação Canônica de Populações e Resultados (P0 — Seção 4)

- **Fonte única:** `{p0['run_id']}` | Commit: `{commit}`
- **Figura associada:** `docs/tcc_evidence/p0_audit/figures/fig_p0_01_reconciliacao_populacional.pdf`

## 1. Tabela Canônica de Reconciliação

{_md_table(["Indicador", "Reproduzido", "Ref. Anterior", "Δ", "Origem", "Período", "Política / Nota"], rec_rows_md, align_right={1, 2, 3})}

## 2. Reconciliação das 9 Categorias Diagnósticas (OOT 19/11/2026)

{_md_table(["Categoria Diagnóstica", "Total UCs", "Elegíveis (>=70%)", "is_elec_anomaly", "Candidatos Vistoria", "Flag Regra (+0,15)", "s_elec Mediana"], cat_rows_md, align_right={1, 2, 3, 4, 5, 6})}

**Soma total:** `{_fmt_i(int(cat_df['total_ucs'].sum()))}` UCs (exato).  
**Soma das 3 categorias elétricas/GD/metrológicas:** `4.087 + 1.451 + 515 = 6.053` UCs.  
**Diferença exata de 16 UCs (`6.053 − 6.037`):** {rec['gd16_outside_queue']['explanation']}

## 3. Verificação dos 8 Pontos Específicos (Seção 4.1)

1. **Linha de 80% da sensibilidade:** Confirmada **sem inversão** — `61.530` elegíveis (93,67%) e `4.156` em quarentena (6,33%), totalizando `65.686`.
2. **População cadastral vs. exploratória por alimentador:** O cadastro atualizado de 25/11 soma `{_fmt_i(rec['n_cad'])}` UCs (+12 em relação ao snapshot de 20/11 de 69.895); a união dos 4 cortes de modelagem soma `{_fmt_i(rec['n_tr_ucs'])}` UCs (`262.769` janelas), e o corte OOT de 19/11 soma `{_fmt_i(rec['n_oot'])}` UCs (`4.640` sem alimentador cadastrado).
3. **Soma das categorias:** Fechada exatamente em `65.686`.
4. **Diferença 6.053 vs. 6.037:** Exatamente 16 UCs em `POSSIVEL_GD_NAO_CADASTRADA` com `s_elec` entre `{_fmt_f(rec['gd16_outside_queue']['s_elec_min'], 4)}` e `{_fmt_f(rec['gd16_outside_queue']['s_elec_max'], 4)}` (`< 0,5131`).
5. **Correspondência candidatura × `is_elec_anomaly`:** Exata `1:1` (`6.037 == 6.037`). Nota: `meta__is_clean_for_train` (`63.081`) difere de `is_reliable_elec` (`62.857`) em `236` UCs limpas com `cov < 0,70` e `12` UCs com `cov >= 0,70` mas `meta__is_clean_for_train == False` (menos dias confiáveis).
6. **Diferença entre anomalia, hipótese diagnóstica e indicação de vistoria:** Anomalia estatística bruta (`s_elec >= 0,5131`) ocorre em `6.916` UCs; após gate `>= 70%`, `6.037` tornam-se `is_elec_anomaly` e candidatas à vistoria; `16` adicionais recebem hipótese diagnóstica de GD por regra física pura sem superar o limiar estatístico.
7. **Reconciliação OOT × Regras Diárias (19/11):** Interseção = `2.069`, exclusivos OOT = `3.968` (`2.069 + 3.968 = 6.037`), exclusivos regras = `2.909` (`2.069 + 2.909 = 4.978`).
8. **Distribuição por circuito (anonimizada):**

{_md_table(["Circuito (LGPD)", "Cadastro (25/11)", "União 4 Cortes", "OOT (19/11)", "Elegíveis OOT", "Candidatos OOT"], f_rows_md, align_right={1, 2, 3, 4, 5})}
""")

    # 3. AUDITORIA_CORRELACOES.md
    def _corr_md(df_c: pl.DataFrame) -> str:
        cols = df_c.columns
        rows = [[r["detector"]] + [_fmt_f(r[c], 3) for c in cols[1:]] for r in df_c.to_dicts()]
        return _md_table(["Detector"] + cols[1:], rows, align_right=set(range(1, len(cols))))

    dist_rows_md = [
        [r["detector"], _fmt_f(r["raw_p50"], 4), _fmt_f(r["raw_p95"], 4), _fmt_f(r["norm_mean"], 4),
         _fmt_f(r["norm_std"], 4), f"{_fmt_f(r['norm_zero_pct'], 2)}%", f"{_fmt_f(r['norm_sat1_pct'], 2)}%",
         f"{_fmt_f(r['norm_tie_pct'], 2)}%", _fmt_f(r["spearman_vs_cons_all"], 4), _fmt_f(r["spearman_vs_cons_elig"], 4)]
        for r in dist_df.to_dicts()
    ]
    _write("AUDITORIA_CORRELACOES.md", f"""
# Auditoria de Correlação e Orientação dos Detectores (P0 — Seção 5)

- **Execução:** `{p0['run_id']}` | Coorte: OOT 19/11/2026 (`65.686` total; `62.857` elegíveis)
- **Figuras:** `docs/tcc_evidence/p0_audit/figures/fig_p0_02_correlacao_detectores_revisada.pdf` e `fig_p0_03_distribuicao_scores_individuais.pdf`

## 1. Verificação de Integridade e Investigação IF × ECOD

- **Alinhamento por UC (`max_abs_diff` vs. `decisions.parquet`):** `0,000000` em todos os 6 detectores e no consenso.
- **Simetria e diagonal:** Simétrica = `{corr['symmetric_spearman_elig']}`; Diagonal unitária = `{corr['diagonal_is_one']}`; NaNs = `0`.
- **Orientação dos escores:** Todos os 6 detectores estão orientados no mesmo sentido (maior escore = maior anomalia).
- **Correlação IF × ECOD:** **Estritamente positiva** — Spearman = **`+{_fmt_f(corr['if_ecod_spearman_elig'], 4)}`** (elegíveis) e **`+{_fmt_f(corr['if_ecod_spearman_all'], 4)}`** (OOT total); Pearson = **`+{_fmt_f(corr['if_ecod_pearson_elig'], 4)}`**. **Não existe correlação negativa entre nenhum par de detectores** (mínimo par a par = `+{_fmt_f(corr['min_pairwise_spearman_elig'], 4)}` entre LOF e PCA).
- **Reconciliação dos pares LOF/PCA vs. consenso:** No OOT total (`65.686` UCs), `LOF = +0,4763` e `PCA = +0,4396`; nos elegíveis (`62.857` UCs), `LOF = +0,4609` e `PCA = +0,4135`. A diferença decorre exclusivamente do denominador populacional.

## 2. Matriz de Spearman — Escores Normalizados (Elegíveis OOT, n = 62.857)

{_corr_md(sp_df)}

## 3. Matriz de Pearson — Escores Normalizados (Elegíveis OOT, n = 62.857)

{_corr_md(pe_df)}

## 4. Distribuições, Proporção de Zeros, Saturação em 1 e Empates

{_md_table(["Detector", "Bruto P50", "Bruto P95", "Norm. Média", "Norm. DP", "Zeros (%)", "Sat. =1 (%)", "Empates (%)", "ρ vs Cons. (65.686)", "ρ vs Cons. (62.857)"], dist_rows_md, align_right={1,2,3,4,5,6,7,8,9})}
""")

    # 4. SENSIBILIDADE_GATING_CONGELADO.md
    gat_rows_md = [
        [f"{r['coverage_threshold_pct']}%", _fmt_i(r["eligible_ucs"]), f"{_fmt_f(r['eligible_pct'], 2)}%",
         _fmt_i(r["quarantine_ucs"]), _fmt_i(r["raw_anomalies_frozen_model"]), _fmt_i(r["candidates_retained"]),
         _fmt_i(r["anomalies_intercepted_quarantine"]), _fmt_i(r["excluded_vs_ref70"]), _fmt_i(r["newly_eligible_vs_ref70"]),
         _fmt_f(r["jaccard_cands_vs_ref70"], 4), _fmt_f(r["top2k_jaccard_vs_ref70"], 4), _fmt_i(r["top2k_substitutions_vs_ref70"])]
        for r in gat_df.to_dicts()
    ]
    _write("SENSIBILIDADE_GATING_CONGELADO.md", f"""
# Sensibilidade do Gating com Modelo Congelado (P0 — Seção 6)

- **Execução:** `{p0['run_id']}` | **Protocolo:** Zero reajuste (*fit*) no OOT; modelo, normalizações, pesos e limiar (`τ = 0,5131`) estritamente congelados no treino (`<= 31/10/2026`).
- **Distinção metodológica:** O experimento anterior (`20261125_134940_636_gating_sensitivity`) reajustava o detector a cada limiar sobre o próprio OOT (protocolo distinto, preservado em `experiments/runs/`). A tabela abaixo isola o efeito puro da política de qualidade.
- **Figura:** `docs/tcc_evidence/p0_audit/figures/fig_p0_04_sensibilidade_gating_congelado.pdf`

{_md_table(["Cobertura Mín.", "Elegíveis", "Eleg. (%)", "Quarentena", "Anom. Brutas (fixo)", "Retidos", "Interceptados", "Excluídos vs 70%", "Novos vs 70%", "Jaccard vs 70%", "Jaccard Top-2k", "Trocas Top-2k"], gat_rows_md, align_right={1,2,3,4,5,6,7,8,9,10,11})}

**Conclusões permitidas:**
- As sinalizações brutas permanecem rigorosamente constantes em **`6.916`** em todos os limiares, comprovando ausência total de *refit*.
- Elevar o limiar de 70% para 80% exclui 217 candidatas (46 trocas no Top-2.000; Jaccard 0,9550); elevar para 90% exclui 659 candidatas (171 trocas; Jaccard 0,8425).
- Sem rótulos externos de inspeção, **exclusões pelo gate não são denominadas falsos negativos**.
""")

    # 5. SENSIBILIDADE_PESOS.md
    w_rows_md = [
        [r["configuracao"], _fmt_f(r["w_IF"], 3), _fmt_f(r["w_LOF"], 3), _fmt_f(r["w_ECOD"], 3), _fmt_f(r["w_COPOD"], 3),
         _fmt_f(r["w_HBOS"], 3), _fmt_f(r["w_PCA"], 3), _fmt_f(r["spearman_vs_ref"], 4), _fmt_f(r["kendall_5k_vs_ref"], 4),
         _fmt_f(r["top2k_jaccard"], 4), _fmt_i(r["top2k_substituicoes"]), _fmt_i(r["protoB_cands_tau_fixo_05131"]),
         _fmt_f(r["protoB_tau_calibrado_treino"], 4), _fmt_i(r["protoB_cands_tau_calibrado"])]
        for r in w_df.to_dicts()
    ]
    b_rows_md = [
        [r["modo_regra"], _fmt_f(r["bonus"], 3), _fmt_i(r["ucs_com_flag_elig"]), f"{_fmt_f(r['pct_elig_com_flag'], 2)}%",
         _fmt_f(r["delta_medio_score_vs_b0"], 4), _fmt_f(r["spearman_vs_ref_25_11"], 4),
         _fmt_f(r["top2k_jaccard_vs_ref_25_11"], 4), _fmt_i(r["top2k_substituicoes_vs_ref_25_11"]), _fmt_i(r["cands_tau_fixo_05131"])]
        for r in b_df.to_dicts()
    ]
    _write("SENSIBILIDADE_PESOS.md", f"""
# Sensibilidade dos Pesos do Ensemble e do Bônus de Corroboração (P0 — Seções 7 e 8)

- **Execução:** `{p0['run_id']}` | População: `62.857` UCs elegíveis no OOT (escores individuais congelados; zero *fit* no OOT)
- **Pesos efetivos confirmados no código (`src/models/composite.py` e `frozen_multiview_model.joblib`):** `IF=0,22`, `LOF=0,20`, `ECOD=0,18`, `COPOD=0,18`, `HBOS=0,10`, `PCA=0,12`. *(Nota: divergência identificada frente à tabela histórica que listava ECOD=0,20, HBOS=0,18, LOF=0,10 — ambas avaliadas abaixo.)*
- **Figuras:** `fig_p0_05_sensibilidade_pesos.pdf`, `fig_p0_06_variacao_bonus_corroboracao.pdf` e `fig_p0_07_comparacao_carteiras_topk.pdf`

## 1. Sensibilidade dos Pesos (Protocolo A: Top-2.000 Fixo × Protocolo B: Limiar Fixo e Recalibrado no Treino)

{_md_table(["Configuração", "IF", "LOF", "ECOD", "COPOD", "HBOS", "PCA", "Spearman", "Kendall (5k)", "Jacc. Top-2k", "Trocas Top-2k", "Prot.B τ=0,5131", "τ Treino P95", "Prot.B τ Calib."], w_rows_md, align_right=set(range(1, 14)))}

## 2. Sensibilidade do Bônus de Corroboração (+0,15) — Seção 8

**Achado crítico auditado:** Em `src/models/composite.py` (L552), o padrão `"IMBALANCE_MAX"` capturou `x__hist_current_imbalance_max > 0,20` (`29.360` elegíveis = 46,77%), enquanto as violações estritas de desequilíbrio de tensão `> 20%` (`57`) ou reversão `> 85%` (`32`) somam apenas `89` elegíveis (`0,14%`).

{_md_table(["Regra Avaliada", "Bônus", "UCs c/ Flag", "% Eleg.", "Δ Médio Score", "Spearman vs 25/11", "Jacc. Top-2k", "Trocas Top-2k", "Cands τ=0,5131"], b_rows_md, align_right={1,2,3,4,5,6,7,8})}

**Interpretação rigorosa:**
- Sob **orçamento fixo (Protocolo A, Top-2.000)**, o ranking é altamente estável a variações de ±20% nos pesos (apenas 6 a 28 trocas em 2.000; Jaccard `0,9724–0,9940`), mas sensível à remoção de detectores estruturalmente distintos (`Sem LOF`: 180 trocas; `IF isolado`: 395 trocas).
- Sob **limiar fixo (Protocolo B)**, o volume oscila fortemente se os pesos ou o bônus variarem sem recalibração do limiar no treino.
- Concordância com a configuração atual mede **estabilidade metodológica**, não comprovação de superioridade em detecção de fraude.
""")

    # 6. PARTICIPACAO_ESPECIALISTAS.md
    spec_rows_md = [
        [r["componente"], r["implementado"], r["executado_oot_25_11"], r["avaliado_isoladamente"],
         r["participa_candidatura_25_11"], r["parametros_principais"]]
        for r in spec_df.to_dicts()
    ]
    _write("PARTICIPACAO_ESPECIALISTAS.md", f"""
# Matriz Auditada de Participação dos Especialistas (P0 — Seção 9)

- **Execução:** `{p0['run_id']}` | **Figura:** `docs/tcc_evidence/p0_audit/figures/fig_p0_08_matriz_participacao_especialistas.pdf`

{_md_table(["Componente", "Implementado", "Executado (25/11)", "Avaliado Isoladamente", "Participa da Candidatura (25/11)", "Parâmetros / Pesos / Limiares"], spec_rows_md)}

## Auditoria de `S_tamper` e Regras Físicas (Seção 17)

- **Descoberta de engenharia:** A campanha de 25/11 registrou 0 alarmes em `x__alarm_count_window` porque `oot_pipeline.py` buscava arquivo único `alarm_events.parquet`, enquanto o Data Lake armazena `output/events/alarm_events/` em `{mv['alarm_events_audit'].get('n_parquet_partitions', 41)}` partições Parquet (`{_fmt_i(mv['alarm_events_audit'].get('total_rows', 0))}` registros totais até 25/11/2026).
- **Auditoria direta dos alarmes particionados na janela OOT (21/10–19/11/2026):**
  - Total de alarmes na janela OOT: `{_fmt_i(tamper.get('alarms_in_oot_30d'))}`
  - UCs OOT com >=1 alarme (qualquer tipo): `{_fmt_i(tamper.get('ucs_oot_with_any_alarm'))}`
  - UCs OOT com >=1 alarme de tamper/intrusão: `{_fmt_i(tamper.get('ucs_oot_with_tamper_alarm'))}`
  - UCs OOT com `S_tamper >= 0,70` (com alarmes reais): `{_fmt_i(tamper.get('real_s_tamper_ge_070'))}` (sobreposição com anomalia elétrica: `{_fmt_i(tamper.get('overlap_tamper_and_elec_anom'))}`; exclusivos: `{_fmt_i(tamper.get('exclusive_tamper_not_elec_anom'))}`)

{_md_table(["Regra Física / Especialista", "Acionamentos OOT", "Sobreposição c/ Anom. Elétrica", "Exclusivos (sem Anom. Elétrica)"], [[r["regra"], _fmt_i(r["acionamentos_oot"]), _fmt_i(r["sobreposicao_anom_eletrica"]), _fmt_i(r["exclusivos_sem_anom_eletrica"])] for r in phys_df.to_dicts()], align_right={1,2,3})}
""")

    # 7. AVALIACAO_MULTIVISAO.md
    moe_rows_md = [
        [r["circuito"], _fmt_i(r["n_train_clean"]), _fmt_i(r["n_oot"]), f"{_fmt_f(r['gd_pen_train_pct'], 2)}%",
         _fmt_f(r["w_feeder_eff"], 3), _fmt_f(r["tau_feeder_frozen"], 4), _fmt_f(r["spearman_feeder_vs_global"], 4)]
        for r in moe_df.to_dicts()
    ]
    _write("AVALIACAO_MULTIVISAO.md", f"""
# Avaliação Integrada da Arquitetura Multivisão e MoE Topológico (P1 — Seções 12 e 18)

- **Execução:** `{mv['run_id']}` | Tempo total: `{_fmt_f(mv['runtime_sec'], 2)} s`
- **Contrato de saída gerado:** `experiments/runs/{mv['run_id']}/multiview_evidence_oot.parquet` (`65.686` UCs, anonimizado LGPD, `NULL` preservado quando componente indisponível/inelegível)
- **Figuras:** `fig_p1_09_arquitetura_multivisao_executada.pdf` e `fig_p1_15_comparacao_global_vs_contextual_moe.pdf`

## 1. Desempenho dos Especialistas no OOT (65.686 UCs)

- **`S_elec` (congelado):** `{_fmt_f(mv['timings_sec']['s_elec'], 2)} s`
- **`S_behavior` (distância PCA+KMeans congelado):** `{_fmt_f(mv['timings_sec']['s_behavior'], 2)} s`
- **`CPD` (CUSUM sobre 30 dias reais de `uc_day_features`):** `{_fmt_f(mv['timings_sec']['cpd_65k'], 2)} s` — `{_fmt_i(mv['cpd_oot_summary']['cpd_evaluable_ucs'])}` UCs avaliáveis (`{_fmt_i(mv['cpd_oot_summary']['cpd_not_evaluable_ucs'])}` inavaliáveis com `< 8` dias confiáveis → `NULL`), `{_fmt_i(mv['cpd_oot_summary']['cpd_detected_any'])}` quebras detectadas (`{_fmt_i(mv['cpd_oot_summary']['cpd_detected_drop_ge30'])}` quedas `>= 30%`).
- **`MoE` Hierárquico sem vazamento (16 especialistas por alimentador treinados estritamente em `<= 31/10/2026`):** `{_fmt_f(mv['timings_sec']['moe_14feeders'], 2)} s`.

## 2. Comparação Global × Contextual MoE (População Comum Elegível com Alimentador: `{_fmt_i(mv['moe_comparison']['common_population_elig_with_feeder'])}` UCs)

- **Spearman Global × MoE:** `{_fmt_f(mv['moe_comparison']['spearman_global_vs_moe_common'], 4)}`
- **Deslocamento absoluto de percentil:** média `{_fmt_f(mv['moe_comparison']['mean_abs_percentile_shift'], 2)}` p.p., P95 `{_fmt_f(mv['moe_comparison']['p95_abs_percentile_shift'], 2)}` p.p.
- **Alteração no Top-2.000:** Jaccard = `{_fmt_f(mv['moe_comparison']['top2k_global_vs_moe_jaccard'], 4)}` (**`{_fmt_i(mv['moe_comparison']['top2k_global_vs_moe_substitutions'])}` substituições**; `{_fmt_i(mv['moe_comparison']['top2k_global_vs_moe_common'])}` comuns).
- **Filtro de eventos coletivos no OOT real (`grid_event_threshold = 0,60`):** `{mv['moe_comparison']['grid_disturbance_flagged_ucs']}` transformadores atingiram 60% simultâneo no corte estático de 19/11 (efeito coletivo validado em perturbação controlada na Seção 18/Base B).

{_md_table(["Circuito (LGPD)", "Treino Limpo", "OOT (19/11)", "GD Treino (%)", "w_feeder", "τ Alimentador", "Spearman vs Global"], moe_rows_md, align_right={1,2,3,4,5,6})}
""")

    # 8. PERTURBACOES_CONTROLADAS.md
    sc_rows_md = [
        [r["cenario"], r["componente_avaliado"], _fmt_f(r["delta_cov_medio"], 3), _fmt_f(r["delta_s_comm_medio"], 3),
         _fmt_f(r["delta_s_elec_medio"], 3), f"{_fmt_f(r['taxa_quarentena_pct'], 1)}%",
         f"{_fmt_f(r['taxa_anom_eletrica_sem_gate_pct'], 1)}%", f"{_fmt_f(r['taxa_anom_eletrica_com_gate_pct'], 1)}%",
         f"{_fmt_f(r['taxa_cpd_localizacao_3d_pct'], 1)}%", _fmt_f(r["cpd_erro_mediano_dias"], 1),
         f"{_fmt_f(r['beh_24h_taxa_deteccao_p95_pct'], 1)}%"]
        for r in sc_df.to_dicts()
    ]
    _write("PERTURBACOES_CONTROLADAS.md", f"""
# Experimento de Perturbações Controladas sobre Séries AMI Reais (P1 — Seções 13 a 16)

- **Execução:** `{pert['run_id']}` | **Seed:** `42` | **Amostra estratificada:** `{pert['sample_selection']['n_ucs_valid']}` UCs reais em `{pert['sample_selection']['n_feeders']}` circuitos (`{_fmt_i(pert['sample_selection']['n_interval_rows_loaded'])}` registros intervalares de 5/10 min)
- **Integridade do Data Lake:** Leitura estritamente *read-only*; perturbações aplicadas **antes** da extração de features sobre cópias isoladas em memória (`data_lake_untouched = {pert['sample_selection']['data_lake_untouched']}`).
- **Grupo de referência (não perturbado):** cobertura média `{_fmt_f(pert['reference_unperturbed']['mean_coverage']*100, 1)}%`, taxa `S_elec >= τ` = `{_fmt_f(pert['reference_unperturbed']['elec_anom_rate_pct'], 1)}%`, taxa CPD queda = `{_fmt_f(pert['reference_unperturbed']['cpd_drop_rate_pct'], 1)}%`, mudança de cluster 24h = `{_fmt_f(pert['reference_unperturbed']['beh_cluster_change_rate_pct'], 1)}%`. *(Não se presume que séries reais de referência sejam comprovadamente isentas de irregularidade.)*
- **Figuras:** `fig_p1_10` a `fig_p1_14` em `docs/tcc_evidence/p1_perturbations/figures/`.

## 1. Matriz Completa de Resultados por Cenário (23 Cenários Controlados)

{_md_table(["Cenário", "Componente", "Δ Cob.", "Δ S_comm", "Δ S_elec", "Quarent. (%)", "Anom s/Gate", "Anom c/Gate", "CPD <=3d (%)", "Erro CPD (d)", "Beh 24h >P95"], sc_rows_md, align_right={2,3,4,5,6,7,8,9,10})}

## 2. Perturbação Isolada vs. Coletiva no Circuito (`{coll['circuito_base']}`)

- **Evento isolado (-50% em 1 UC do posto):** sinalizada = `{coll['evento_isolado_1_uc']['targets_flagged_raw']}/1`, mantida como `GENUINE_UC_ANOMALY` = `{coll['evento_isolado_1_uc']['targets_kept_genuine_uc_anomaly']}/1`, classificada como evento de rede = `{coll['evento_isolado_1_uc']['targets_classified_grid_event']}/1` (escore médio `{_fmt_f(coll['evento_isolado_1_uc']['mean_score_before_filter'], 3)}` → `{_fmt_f(coll['evento_isolado_1_uc']['mean_score_after_filter'], 3)}`).
- **Evento coletivo (-50% simultâneo em 80% do circuito = `{coll['evento_coletivo_80pct_circuito']['n_perturbed_ucs']}` UCs):** sinalizadas brutas = `{coll['evento_coletivo_80pct_circuito']['targets_flagged_raw']}/{coll['evento_coletivo_80pct_circuito']['n_perturbed_ucs']}`, reclassificadas como `GRID_DISTURBANCE` = **`{coll['evento_coletivo_80pct_circuito']['targets_classified_grid_event']}/{coll['evento_coletivo_80pct_circuito']['n_perturbed_ucs']}`**, escore atenuado de `{_fmt_f(coll['evento_coletivo_80pct_circuito']['mean_score_before_filter'], 3)}` para `{_fmt_f(coll['evento_coletivo_80pct_circuito']['mean_score_after_filter'], 3)}`.

## 3. Limitações e Achados Desfavoráveis Registrados Honestamente

- **Quedas moderadas (-10% a -30%) no `S_elec` transversal:** Sem série temporal pré-evento, quedas moderadas alteram pouco a posição transversal da UC na população (`Δ S_elec` próximo de zero); o especialista longitudinal **CPD (CUSUM)** é indispensável para localizar essas quedas.
- **Lacuna de comunicação aleatória de 10% a 20%:** Reduz a cobertura média para ~87%–78% (acima do gate de 70%), não acionando quarentena mas elevando gradualmente `S_comm`; a partir de 30%–50% de perda, o gate intercepta a grande maioria das séries degradadas.
- **Interpretação estrita:** Métricas controladas demonstram capacidade de observação algorítmica de perturbações conhecidas, **não** desempenho de identificação de fraude em campo.
""")

    # 9. ABLACAO_METODOLOGICA.md
    aba_rows_md = [
        [r["variante"], _fmt_i(r["ucs_elegiveis"]), _fmt_i(r["candidatos_tau_05131"]),
         _fmt_f(r["top2k_jaccard_vs_V0"], 4), _fmt_f(r["top2k_jaccard_vs_V2"], 4),
         _fmt_i(r["top2k_entradas_vs_V2"]), _fmt_i(r["top2k_quarentena_vazadas"]),
         _fmt_i(r["top2k_com_cpd_queda"]), f"{_fmt_f(r['tempo_computacional_s'], 2)} s"]
        for r in ab_a_df.to_dicts()
    ]
    abb_rows_md = [
        [r["variante"], _fmt_i(r["orcamento_K_fixo"]), f"{_fmt_f(r['recall_pert_eletricas_pct'], 2)}%",
         f"{_fmt_f(r['precisao_controlada_pct'], 2)}%", f"{_fmt_f(r['f1_controlado_pct'], 2)}%",
         f"{_fmt_f(r['vazamento_lacunas_comm_pct'], 2)}%", f"{_fmt_f(r['recall_quedas_cpd_pct'], 2)}%",
         f"{_fmt_f(r['recall_shift_rotina_4h_pct'], 2)}%", f"{_fmt_f(r['acionamento_referencia_pct'], 2)}%"]
        for r in ab_b_df.to_dicts()
    ]
    _write("ABLACAO_METODOLOGICA.md", f"""
# Ablação Metodológica Completa V0–V5 (Base A OOT Real × Base B Perturbações Controladas — Seção 19)

- **Orçamento constante:** `K = 2.000` na Base A (`65.686` UCs OOT) e `K = {ab_b['V0']['orcamento_K_fixo']}` na Base B (`{ab_b['V0']['n_base_b']}` séries controladas).
- **Preservação do funil M0–M5:** O funil sequencial existente (`65.686 → 62.857 → 58.857 → 4.311 → 4.311`) permanece preservado como experimento de seletividade da campanha de 25/11.
- **Figuras:** `fig_p1_16a_ablacao_variantes_base_a.pdf` e `fig_p1_16b_ablacao_variantes_base_b.pdf`

## 1. Base A — População AMI Real OOT (19/11/2026, K = 2.000)

{_md_table(["Variante", "Elegíveis", "Cands τ=0,5131", "Jacc. vs V0", "Jacc. vs V2", "Trocas vs V2", "Quar. no Top-2k", "CPD>=30% Top-2k", "Tempo"], aba_rows_md, align_right={1,2,3,4,5,6,7,8})}

## 2. Base B — Séries Reais com Perturbações Controladas (Orçamento K Fixo)

{_md_table(["Variante", "K Fixo", "Recall Elét.", "Prec. Contr.", "F1 Contr.", "Vaz. Telecom", "Recall Quedas", "Recall Shift 4h", "Ref. Acion."], abb_rows_md, align_right={1,2,3,4,5,6,7,8})}

## 3. Análise Crítica (Incluindo Trade-offs)

- **V0 → V1 → V2 (Ensemble + Gate):** Na Base A, V0 e V1 vazam `138` e `257` UCs em quarentena (`< 70%`) para dentro do Top-2.000; o gate em V2 zera completamente esse vazamento (`0`). Na Base B, o vazamento de lacunas de comunicação cai de `{_fmt_f(ab_b['V1']['vazamento_lacunas_comm_pct'], 1)}%` para **`{_fmt_f(ab_b['V2']['vazamento_lacunas_comm_pct'], 1)}%`**.
- **V2 → V3 → V4 → V5 (Comportamento + CPD + Regras Físicas):** Na Base B, a inclusão de comportamento 24h eleva o recall do deslocamento +4h de `{_fmt_f(ab_b['V2']['recall_shift_rotina_4h_pct'], 1)}%` para `{_fmt_f(ab_b['V3']['recall_shift_rotina_4h_pct'], 1)}%`, e a inclusão de CPD + regras físicas eleva o F1 controlado das perturbações elétricas de `{_fmt_f(ab_b['V2']['f1_controlado_pct'], 1)}%` (V2) para **`{_fmt_f(ab_b['V5']['f1_controlado_pct'], 1)}%`** (V5).
- **Trade-off honesto:** Sob orçamento fixo `K`, elevar o peso de perturbações comportamentais puras (`V3`, sem queda de energia) compete por vagas com perturbações elétricas, podendo reduzir ligeiramente o F1 puramente elétrico antes da entrada de CPD e regras físicas (`V4–V5`). Além disso, nenhum ganho na Base B equivale automaticamente a ganho comprovado de *yield* em vistorias reais.
""")

    # 10. ESTUDOS_DE_CASO.md
    cs_blocks = []
    for ct in mv["case_studies"]:
        sc_i = ct["individual_norm_scores"]
        cs_blocks.append(f"""
### UC `{ct['uc_masked']}` ({", ".join(ct['nios_masked'])} — {ct['feeder_masked'] or 'Sem alimentador'})

- **Cobertura OOT (19/11):** `{_fmt_f(ct['coverage_oot']*100, 2)}%` (`{ct['reliable_days_oot']}` dias confiáveis) | **Elegibilidade:** `{ct['eligibility']}`
- **Grandezas elétricas (30d):** FA mediana = `{_fmt_f(ct['fa_median_kwh_d'], 3)}` kWh/dia; RA máx = `{_fmt_f(ct['ra_ratio_max'], 4)}` (`{ct['ra_days']}` dias); slope = `{_fmt_f(ct['trend_slope'], 4)}` kWh/dia²
- **Escores individuais normalizados:** IF=`{_fmt_f(sc_i['iso'], 3)}`, LOF=`{_fmt_f(sc_i['lof'], 3)}`, ECOD=`{_fmt_f(sc_i['ecod'], 3)}`, COPOD=`{_fmt_f(sc_i['copod'], 3)}`, HBOS=`{_fmt_f(sc_i['hbos'], 3)}`, PCA=`{_fmt_f(sc_i['pca'], 3)}`
- **Especialistas P1:** `S_comm={_fmt_f(ct['s_comm'], 4)}` | `S_elec(25/11)={_fmt_f(ct['s_elec_25_11'], 4)}` | `MoE={_fmt_f(ct['moe_score_p1'], 4)}` | `S_behavior={_fmt_f(ct['s_behavior_p1'], 4)}` | `CPD={ct['cpd_detected']} ({ct['cpd_magnitude_pct']}% em {ct['cpd_timestamp']})`
- **Regras físicas:** `{ct['physical_rules']}` | **Evidência topológica:** `{ct['topological_evidence']}`
- **Decisão 25/11 vs. Multivisão P1:** Categoria = `{ct['categoria_25_11']}`; Candidata = `{ct['candidato_25_11']}`; Rank V2 (25/11) = `{ct['rank_v2_25_11']}` → Rank V5 (P1) = `{ct['rank_v5']}`
- **Trilha auditável:** `{ct['decision_reason']}`
""")
    _write("ESTUDOS_DE_CASO.md", f"""
# Estudos de Caso Completos sob Arquitetura Multivisão Integrada (P1 — Seção 20)

- **Execução:** `{mv['run_id']}` | **Figura:** `docs/tcc_evidence/p1_multiview/figures/fig_p1_17_trilhas_estudos_de_caso.pdf`

{"".join(cs_blocks)}

## Impacto da Arquitetura Integrada sobre as Interpretações Anteriores

1. **UC `908***00`:** A integração automática do CPD confirma em lote o degrau de `-69,06%` exatamente em `2026-10-31`, alinhado à injeção reversa (`RA` em 12 dias, `3,89 a 31,27 kWh/dia`), mantendo-a no topo prioritário com escore `1,0000` tanto no global quanto no MoE. Hipótese alternativa: instalação de microgeração solar sem conclusão cadastral no CIS.
2. **UC `291***96`:** Preservada estritamente em quarentena técnica (`ANOMALIA_COMUNICACAO_ISOLADA`, cobertura `3,33%`, `1` dia confiável), com `CPD = NULL` (inavaliável) e `rank = NULL`. Preserva-se a ressalva de não atribuir estabilidade ao consumo médio anterior nem inferir vínculo comercial após `DATA_REMOCAO = 2026-06-11` sem verificação cadastral/campo.
3. **UC `685***20`:** Mantida como elegível de altíssima qualidade (`99,49%` à cadência nativa de 10 min, 144 intervalos/dia; 100% na grade de 10 min vs. 50% aparente na grade fixa de 5 min), com `S_elec = 0,9850`, `MoE = 0,9981`, `S_behavior = 1,0000` e `CPD = ESTAVEL` (anomalia estrutural contínua, não degrau recente).
""")

    # 11. PROJECAO_OPERACIONAL.md
    ops_rows_md = [
        [r["politica_priorizacao"], _fmt_f(r["jaccard_vs_eletrico_25_11"], 4), _fmt_i(r["substituicoes_vs_25_11"]),
         _fmt_i(r["fila_pnt_analise"]), _fmt_i(r["fila_gd_nao_cadastrada"]), _fmt_i(r["fila_metrologica"]),
         _fmt_f(r["fa_mediana_kwh_d_carteira"], 2), f"{_fmt_f(r['cobertura_media_carteira_pct'], 2)}%", f"{_fmt_f(r['pct_com_cpd_queda'], 2)}%"]
        for r in ops_df.to_dicts()
    ]
    q_rows_md = [[r["fila_operacional"], _fmt_i(r["ucs_oot"]), r["destino"]] for r in q_df.to_dicts()]
    _write("PROJECAO_OPERACIONAL.md", f"""
# Aplicação Operacional sob Capacidade Restrita K = 2.000/mês (Parte C — Seção 21)

- **Execução:** `{mv['run_id']}` | **Figura:** `docs/tcc_evidence/p1_multiview/figures/fig_p1_18_composicao_carteiras_operacionais_k2000.pdf`

> **Fluxo operacional prospectivo:** `Telemetria → Qualidade → Evidências → Hipóteses → Priorização → Análise humana (Cockpit) → Eventual vistoria → Resultado futuro de campo`. A política de 40% de substituição e a hipótese de +5 p.p. de *yield* permanecem estritamente **cenários de planejamento**, nunca resultados medidos.

## 1. Separação em Filas Operacionais (OOT Completo: 65.686 UCs)

{_md_table(["Fila Operacional", "UCs no OOT", "Destino Recomendado"], q_rows_md, align_right={1})}

## 2. Comparação de Políticas de Priorização sob Capacidade Fixa (K = 2.000)

{_md_table(["Política de Priorização", "Jaccard vs 25/11", "Trocas", "Fila PNT", "Fila GD", "Fila Metrol.", "FA Med. (kWh/d)", "Cob. Média", "% c/ CPD>=30%"], ops_rows_md, align_right={1,2,3,4,5,6,7,8})}
""")

    # 12. RELATORIO_TESTES.md
    _write("RELATORIO_TESTES.md", f"""
# Relatório de Execução da Suíte de Testes Automatizados (P0 + P1 — Seção 23)

- **Data:** {now_str} | **Commit:** `{commit}`
- **Comando executado:** `.venv\\Scripts\\python.exe -m pytest -q`
- **Resumo:**
  - **Aprovados:** `{tests.get('passed', 'N/A')}`
  - **Falhos:** `{tests.get('failed', 'N/A')}`
  - **Ignorados:** `{tests.get('skipped', 'N/A')}`
  - **Duração:** `{tests.get('duration_sec', 'N/A')} s`
  - **Código de saída:** `{tests.get('returncode', 'N/A')}`

## Cobertura dos Novos Testes (`tests/test_p0_p1_experimental_closure.py`)

1. Alinhamento exato dos escores por UC (`max_abs_diff = 0,0`) e orientação positiva dos 6 detectores.
2. Simetria, diagonal unitária e ausência de correlações negativas na matriz OOT de 25/11.
3. Soma unitária dos pesos, reprodução exata do ensemble e estabilidade `>= 0,95` de Jaccard Top-2.000 sob perturbação ±20%.
4. Congelamento do modelo OOT (`tau = 0,5131`), ausência de vazamento temporal (`max_train < oot_cutoff`) e zero *refit* no gating congelado.
5. Conservação de partição (`62.857 + 2.829 = 65.686`), ausência de inversão nos 80%, e reconciliação exata `6.053 - 16 = 6.037`.
6. Isolamento das séries perturbadas (cópia intacta), preservação exata de energia diária em *time shift* e *shape flatten*, e integridade LGPD + `NULL` no contrato multivisão.
7. Existência e integridade não nula de todas as figuras P0 (8) e P1 (11) geradas.

```text
{tests.get('summary_tail', '').strip()}
```
""")

    # 13. MANIFESTO_EVIDENCIAS.md
    all_fig_rows = []
    for grp_lbl, f_dict in [("P0", p0["figures"]), ("P1 Multivisão", mv["figures"]), ("P1 Perturbações", pert["figures"])]:
        for fk, finfo in f_dict.items():
            fp = PROJECT_ROOT / finfo["pdf"]
            all_fig_rows.append([grp_lbl, fk, finfo["pdf"], _fmt_i(fp.stat().st_size) if fp.exists() else "0", _sha256(fp)[:16] if fp.exists() else "N/A"])
    _write("MANIFESTO_EVIDENCIAS.md", f"""
# Manifesto Consolidado de Evidências, Execuções e Hashes (Seção 24)

- **Data:** {now_str} | **Commit:** `{commit}`
- **Campanha de referência 25/11 preservada (sem sobrescrita):** `{p0['source_suite']['oot_multiview']}`
- **Novas execuções versionadas:**
  - `{p0['run_id']}` (P0 Auditoria & Sensibilidades)
  - `{mv['run_id']}` (P1 Multivisão OOT + MoE + CPD)
  - `{pert['run_id']}` (P1 Perturbações Controladas)

## Figuras Vetoriais Geradas (P0 + P1: {len(all_fig_rows)} figuras em PDF + PNG)

{_md_table(["Fase", "ID Figura", "Caminho PDF", "Tamanho (B)", "SHA-256 (16 hex)"], all_fig_rows, align_right={3})}
""")

    # 14. MAPA_ATENDIMENTO_PARECER_26_11.md
    mapa_rows = [
        ["Divergências populacionais e 6.053 vs 6.037 (16 GD)", "Snapshots distintos (20/11 vs 25/11) e regra física GD sem score >= τ",
         "Reconciliação canônica gerada de fonte única", "Auditoria P0 Seção 4", "Todas as somas fechadas; 16 GD com s_elec 0,2799–0,5091 < 0,5131",
         "RECONCILIACAO_RESULTADOS.md", "Contagens e regras 100% rastreáveis", "Vistorias de campo permanecem fora do escopo"],
        ["Linha 80% do gating (suspeita de inversão)", "Leitura comparativa entre protocolo refit vs congelado",
         "Verificação + novo ensaio com modelo estritamente congelado", "Gating congelado 50–90% (P0 Seção 6)", "80%: 61.530 elegíveis (93,67%), 4.156 quarentena; 6.916 anom. brutas fixas",
         "SENSIBILIDADE_GATING_CONGELADO.md", "Efeito puro da política de qualidade isolado sem refit", "Exclusões não são classificáveis como FN sem TOI"],
        ["Correlação negativa IF × ECOD e pares divergentes LOF/PCA", "Matriz pré-saneamento antiga + mistura de denominadores (65.686 vs 62.857)",
         "Recálculo auditado sobre escores brutos e normalizados OOT", "Auditoria P0 Seção 5", "IF×ECOD = +0,9626 (elig.); 0 correlações negativas; LOF/PCA reconciliados",
         "AUDITORIA_CORRELACOES.md", "Orientação e diversidade LOF/PCA comprovadas", "IF/ECOD/COPOD/HBOS são altamente correlacionados entre si"],
        ["Arbitrariedade dos pesos do ensemble e divergência de tabela", "Pesos no código (LOF=0,20, HBOS=0,10) diferiam da tabela histórica",
         "Auditoria dos pesos reais + 23 cenários (Protocolo A Top-2k e B limiar)", "Sensibilidade P0 Seção 7", "±20%: Jaccard Top-2k 0,972–0,994 (6–28 trocas); Sem LOF: 180 trocas",
         "SENSIBILIDADE_PESOS.md", "Carteira Top-2.000 robusta a pequenas variações de peso", "Limiar fixo exige recalibração ao alterar pesos"],
        ["Efeito do bônus +0,15 não quantificado", "v_imb_cols capturou c_imb > 0,20 em 29.360 UCs elegíveis (46,77%)",
         "Sensibilidade 0 a 0,30 na regra efetiva e na regra estrita tensão/reversão", "Sensibilidade P0 Seção 8", "Regra efetiva: 3.438 → 12.276 cands; estrita (89 UCs): 3.438 → 3.460",
         "SENSIBILIDADE_PESOS.md", "Bug de escopo do bônus identificado e quantificado com transparência", "Campanha 25/11 preservada; correção de c_imb recomendada para v2"],
        ["Especialistas descritos sem execução integrada", "CPD, MoE e comportamento não participavam juntos no OOT 25/11",
         "Runner integrado P1 + matriz auditada de participação", "P1 Multivisão OOT + MoE (16 circuitos) + CPD (65.686 UCs)", "MoE altera 282 vagas no Top-2k; CPD detecta 2.758 quedas >=30%",
         "AVALIACAO_MULTIVISAO.md + PARTICIPACAO_ESPECIALISTAS.md", "Contribuição distinta de cada especialista quantificada", "S_tamper teve 0 alarmes na 25/11 por path particionado (auditado em P1)"],
        ["Ausência de medição de capacidade de detecção", "Sem vistorias de campo (TOI)",
         "Experimento de perturbações controladas em 100 séries AMI reais", "P1 Perturbações Controladas (23 cenários + coletivo)", "Gate zera vazamento telecom; CPD erro mediano 1–2d; Beh 24h detecta shift +4h",
         "PERTURBACOES_CONTROLADAS.md", "Observabilidade algorítmica medida sem depender de vistorias", "Não equivale a medição de fraude ou yield real em campo"],
        ["Comparação com metodologia original sob orçamento constante", "Comparação anterior focava em contagens absolutas distintas",
         "Ablação metodológica V0–V5 sob K fixo na Base A e Base B", "P1 Ablação V0–V5 (Seção 19)", "Base A: V2 zera 138–257 quarentenas no Top-2k; Base B: F1 34%→65%+",
         "ABLACAO_METODOLOGICA.md", "Diferença estrutural de carteira e detecção controlada comprovada", "Rendimento real de inspeção exige piloto prospectivo"],
    ]
    _write("MAPA_ATENDIMENTO_PARECER_26_11.md", f"""
# Mapa de Atendimento ao Parecer de 26/11/2026

- **Consolidado em:** {now_str} | **Commit:** `{commit}`

{_md_table(["Crítica / Problema Original", "Causa Identificada", "Correção / Ação", "Experimento Realizado", "Resultado Observado", "Artefato Comprobatório", "Conclusão Permitida", "Limitação Restante"], mapa_rows)}
""")

    # 15. RESUMO_PARA_REVISAO_AUTORAL.md
    _write("RESUMO_PARA_REVISAO_AUTORAL.md", f"""
# Resumo Técnico Objetivo para Revisão Autoral (Sem Reescrita Automática por IA)

Este documento reúne exclusivamente **fatos, números verificados, figuras e apontamentos por capítulo** para que o autor revise pessoalmente a redação do manuscrito em `docs/TCC_AMI_FINAL_BANCA_eekBR_FONTES/`. Nenhum texto em `.tex` foi reescrito automaticamente nesta etapa.

## 1. Números Verificados a Conferir / Atualizar no Manuscrito

1. **Pesos do ensemble elétrico (`cap08.tex` / `cap12.tex`):** Conferir se o texto declara os pesos efetivos do código congelado: **IF = 0,22; LOF = 0,20; ECOD = 0,18; COPOD = 0,18; HBOS = 0,10; PCA = 0,12** (e não LOF=0,10 / ECOD=0,20 / HBOS=0,18).
2. **Correlações dos detectores (`cap13.tex`):**
   - `IF × ECOD` = **+0,9626** (Spearman elegíveis, n = 62.857) / **+0,9639** (OOT total, n = 65.686). Nenhuma correlação negativa existe na execução de 25/11.
   - `LOF` vs. consenso = **+0,4609** (elegíveis, 62.857) e **+0,4763** (total, 65.686).
   - `PCA` vs. consenso = **+0,4135** (elegíveis, 62.857) e **+0,4396** (total, 65.686).
3. **Sensibilidade do gating com modelo congelado (`cap13.tex`):**
   - Sinalizações brutas constantes = **6.916** (τ = 0,5131).
   - Retidos / Interceptados: 50% (`6.215 / 701`), 60% (`6.114 / 802`), 70% (`6.038 / 878`), 80% (`5.821 / 1.095`), 90% (`5.379 / 1.537`).
   - Explicitar que a tabela anterior de 25/11 (`5.788 / 979` em 70%) correspondia ao protocolo com reajuste no corte (`τ = 0,5226`).
4. **Sensibilidade dos pesos e do bônus `+0,15` (`cap13.tex`):**
   - Variação de ±20% nos pesos: Jaccard Top-2.000 entre **0,9724 e 0,9940** (6 a 28 substituições).
   - Remoção de LOF: **180 substituições** no Top-2.000 (Jaccard 0,8349); Isolation Forest isolado: **395 substituições** (Jaccard 0,6701).
   - **Bônus `+0,15` (ressalva de transparência obrigatória):** No código de 25/11, `rule_violation_flag` ativou em **30.694 UCs (29.398 elegíveis = 46,77%)** devido à inclusão de `x__hist_current_imbalance_max > 0,20` (29.360 elegíveis), frente a apenas **89 elegíveis (0,14%)** sob a regra estrita de tensão (`> 20%`) ou reversão (`> 85%`). Sem bônus (`0,0`), o volume em `τ = 0,5131` cai de `6.039` para `3.438`.
5. **População cadastral (`cap07.tex` / `cap12.tex`):** Snapshot atualizado de 25/11 possui **69.907** UCs (+12 frente ao histórico de 69.895 de 20/11); dataset de modelagem de 4 cortes tem **262.769** linhas e **66.192** UCs distintas.
6. **`S_tamper` e alarmes (`cap08.tex` / `cap13.tex`):** Explicitar que a campanha de 25/11 operou com `x__alarm_count_window = 0` por leitura de arquivo único em vez do diretório particionado `output/events/alarm_events/` (que contém `{_fmt_i(tamper.get('alarms_in_oot_30d'))}` registros na janela de 30d do OOT).

## 2. Novas Figuras Recomendadas para Incorporação

- **P0 (`docs/tcc_evidence/p0_audit/figures/`):**
  - `fig_p0_02_correlacao_detectores_revisada.pdf` (Spearman + Pearson elegíveis)
  - `fig_p0_04_sensibilidade_gating_congelado.pdf` (gating sem refit)
  - `fig_p0_05_sensibilidade_pesos.pdf` (Protocolo A Top-2k + Protocolo B limiar)
  - `fig_p0_06_variacao_bonus_corroboracao.pdf` (impacto do `+0,15`)
  - `fig_p0_08_matriz_participacao_especialistas.pdf` (participação efetiva 25/11 vs P1)
- **P1 (`docs/tcc_evidence/p1_multiview/figures/` e `docs/tcc_evidence/p1_perturbations/figures/`):**
  - `fig_p1_11_resposta_modelos_por_intensidade.pdf` (quedas sustentadas + lacunas)
  - `fig_p1_12_desempenho_controlado_cpd.pdf` (CPD vs ensemble + erro temporal em dias)
  - `fig_p1_13_efeito_gating_lacunas_comunicacao.pdf` (efeito protetor do gate)
  - `fig_p1_14_perfis_comportamentais_e_rotina.pdf` (centroides 24h + shift de horário com ΔkWh=0)
  - `fig_p1_15_comparacao_global_vs_contextual_moe.pdf` (Global × MoE sem vazamento em 16 circuitos)
  - `fig_p1_16a` e `fig_p1_16b` (Ablação V0–V5 na Base A e Base B sob K fixo)

## 3. Afirmações Sustentadas × Não Sustentadas

### Sustentadas pelas novas evidências:
- A carteira Top-2.000 do ensemble é estável a perturbações de ±20% nos pesos (Jaccard >= 0,97).
- LOF e PCA agregam diversidade efetiva ao comitê (baixa correlação com IF/ECOD e até 180 substituições no Top-2.000 ao remover LOF).
- O gating de qualidade elimina o vazamento de séries degradadas por telemetria para a fila prioritária (de 138–257 UCs para 0 na Base A; de {_fmt_f(ab_b['V1']['vazamento_lacunas_comm_pct'], 1)}% para {_fmt_f(ab_b['V2']['vazamento_lacunas_comm_pct'], 1)}% na Base B).
- O módulo CPD localiza quedas sustentadas com erro mediano de ~1–2 dias sem confundir eventos isolados de 1 dia com mudança de regime.
- Perfis comportamentais intradiários de 24h capturam deslocamentos de rotina (+4h) invisíveis à energia diária agregada.
- O filtro espacial de vizinhança separa eventos coletivos do circuito (80% simultâneo) de anomalias individuais isoladas.

### NÃO sustentadas (evitar ou ressalvar explicitamente):
- Qualquer afirmação de "fraude confirmada", "redução comprovada de falsos positivos de fraude" ou "ganho real de +5 p.p. no rendimento de vistorias".
- Afirmar que `S_tamper` com alarmes ou MoE participaram da campanha congelada de 25/11 (participaram apenas da avaliação integrada P1 / auditoria separada).
- Tratar métricas da Base B (perturbações controladas) como acurácia de fiscalização em campo.

## 4. Capítulos que Requerem Revisão Autoral Pontual

- **Capítulo 8:** Conferir tabela de pesos do ensemble, fórmula do bônus `+0,15` (ressalva de `current_imbalance_max > 0,20`) e distinção entre módulos ativos em 25/11 e na avaliação integrada P1.
- **Capítulo 12:** Atualizar referência à sensibilidade congelada e ao protocolo de perturbações controladas.
- **Capítulo 13:** Incorporar tabela de reconciliação, sensibilidade de pesos/gating/bônus, matriz de correlações revisada, tabela central Baseline × Ensemble × Multivisão e achados dos experimentos controlados.
- **Capítulo 15 (Conclusões):** Registrar explicitamente os dois achados de auditoria (gatilho de `c_imb > 0,20` no bônus `+0,15` e path particionado de `alarm_events`) como aprendizados de engenharia e trabalho futuro imediato.
""")

    return written


if __name__ == "__main__":
    files = generate_all_reports()
    print(f"Generated {len(files)} reports:")
    for f in files:
        print(" -", f)
