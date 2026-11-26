# Reconciliação Canônica de Populações e Resultados (P0 — Seção 4)

- **Fonte única:** `20261126_172200_p0_audit_sensitivity` | Commit: `06f7f71`
- **Figura associada:** `docs/tcc_evidence/p0_audit/figures/fig_p0_01_reconciliacao_populacional.pdf`

## 1. Tabela Canônica de Reconciliação

| Indicador | Reproduzido | Ref. Anterior | Δ | Origem | Período | Política / Nota |
| --- | ---: | ---: | ---: | --- | --- | --- |
| UCs cadastrais (snapshot 25/11) | 69.907 | 69895.0 | 12.0 | output/context/uc_context.parquet | Cadastro vigente 25/11/2026 | Snapshot cadastral atualizado (+12 UCs vs snapshot 20/11 de 69.895) |
| UCs únicas no dataset de modelagem (4 cortes) | 66.192 | 66099.0 | 93.0 | output/model_input/v1/training_dataset.parquet | 01/08–19/11/2026 | União dos 4 cortes (262.769 janelas); histórico 66.099 era snapshot exploratório 20/11 |
| Janelas de treino limpas (Ago–Out/2026) | 189.341 | 189341.0 | 0.0 | output/model_input/v1/training_dataset.parquet | 31/08, 30/09, 31/10/2026 | Fit estrito pré-novembro (sem vazamento OOT) |
| UCs OOT avaliadas (19/11/2026) | 65.686 | 65686.0 | 0.0 | experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet | 21/10–19/11/2026 (janela 30d) | Inferência congelada OOT_MULTIVIEW |
| Elegíveis OOT (cobertura >= 70%) | 62.857 | 62857.0 | 0.0 | experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet | 19/11/2026 | Gate de confiabilidade elétrica >= 70% |
| Quarentena técnica OOT (cobertura < 70%) | 2.829 | 2829.0 | 0.0 | experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet | 19/11/2026 | Partição exata: 62.857 + 2.829 = 65.686 |
| Candidatos OOT (IS_CANDIDATO_VISTORIA) | 6.037 | 6037.0 | 0.0 | experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet | 19/11/2026 | Correspondência exata 1:1 com is_elec_anomaly (6.037) |
| Candidatos após restrição topológica (M4/M5) | 4.311 | 4311.0 | 0.0 | experiments/runs/20261125_134849_083_ablation_m0_m5/ablation_funnel.csv | 19/11/2026 | 1.726 candidatos sem alimentador cadastrado retidos fora de M3 |
| Candidatos pelas regras diárias em 19/11 | 4.978 | 4978.0 | 0.0 | experiments/runs/20261125_134535_754_daily_rules/decisions.parquet | 19/11/2026 | DAILY_RULES calibrado (cobertura >= 60%, >= 12d confiáveis) |
| Média diária de candidatos (regras calibradas) | 5.387,7 | 5387.7 | 0.04 | experiments/runs/20261125_134535_754_daily_rules/daily_metrics.parquet | 01/11–19/11/2026 | DAILY_RULES calibrado |
| Janelas totais da série diária | 1.248.501 | 1248501.0 | 0.0 | experiments/runs/20261125_134535_754_daily_rules/decisions.parquet | 01/11–19/11/2026 | DAILY_RULES calibrado |
| Interseção OOT x regras diárias (19/11) | 2.069 | 2069.0 | 0.0 | Cruzamento decisions.parquet (OOT x DAILY_RULES 19/11) | 19/11/2026 | Jaccard = 0.2313 |
| Exclusivos OOT (19/11) | 3.968 | 3968.0 | 0.0 | Cruzamento decisions.parquet (OOT \ DAILY_RULES 19/11) | 19/11/2026 | 2.069 + 3.968 = 6.037 |
| Exclusivos regras diárias (19/11) | 2.909 | 2909.0 | 0.0 | Cruzamento decisions.parquet (DAILY_RULES 19/11 \ OOT) | 19/11/2026 | 2.069 + 2.909 = 4.978 |

## 2. Reconciliação das 9 Categorias Diagnósticas (OOT 19/11/2026)

| Categoria Diagnóstica | Total UCs | Elegíveis (>=70%) | is_elec_anomaly | Candidatos Vistoria | Flag Regra (+0,15) | s_elec Mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OPERACAO_REGULAR | 44.674 | 44.674 | 0 | 0 | 18.510 | 0,2223 |
| MUDANCA_CLASSE_COMPORTAMENTAL | 12.084 | 12.084 | 0 | 0 | 5.375 | 0,2391 |
| ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT | 4.087 | 4.087 | 4.087 | 4.087 | 3.894 | 0,6187 |
| GARGALO_COLETIVO_RF_MESH | 1.829 | 12 | 0 | 0 | 750 | 0,4078 |
| POSSIVEL_GD_NAO_CADASTRADA | 1.451 | 1.451 | 1.435 | 1.435 | 1.370 | 0,8692 |
| ANOMALIA_COMUNICACAO_ISOLADA | 910 | 0 | 0 | 0 | 526 | 0,4207 |
| FALHA_METROLOGICA_MEDIDOR_TRAVADO | 515 | 515 | 515 | 515 | 227 | 0,5808 |
| INICIACAO_PERFIL_UC_NOVA | 71 | 0 | 0 | 0 | 15 | 0,3845 |
| TRANSICAO_TROCA_MEDIDOR | 65 | 34 | 0 | 0 | 27 | 0,3422 |

**Soma total:** `65.686` UCs (exato).  
**Soma das 3 categorias elétricas/GD/metrológicas:** `4.087 + 1.451 + 515 = 6.053` UCs.  
**Diferença exata de 16 UCs (`6.053 − 6.037`):** As 16 UCs pertencem integralmente a POSSIVEL_GD_NAO_CADASTRADA: atenderam à regra física de injeção reversa (ra_ratio >= 0,08 e ra_days >= 2) com telemetria confiável (cobertura >= 70%), porém obtiveram escore elétrico s_elec abaixo do limiar congelado (0,2799 a 0,5091 < 0,5131). Como IS_CANDIDATO_VISTORIA exige is_anomaly == True (equivalente a is_elec_anomaly == True nesta coorte), elas recebem hipótese diagnóstica de GD sem ingresso automático na fila de 6.037 candidatas.

## 3. Verificação dos 8 Pontos Específicos (Seção 4.1)

1. **Linha de 80% da sensibilidade:** Confirmada **sem inversão** — `61.530` elegíveis (93,67%) e `4.156` em quarentena (6,33%), totalizando `65.686`.
2. **População cadastral vs. exploratória por alimentador:** O cadastro atualizado de 25/11 soma `69.907` UCs (+12 em relação ao snapshot de 20/11 de 69.895); a união dos 4 cortes de modelagem soma `66.192` UCs (`262.769` janelas), e o corte OOT de 19/11 soma `65.686` UCs (`4.640` sem alimentador cadastrado).
3. **Soma das categorias:** Fechada exatamente em `65.686`.
4. **Diferença 6.053 vs. 6.037:** Exatamente 16 UCs em `POSSIVEL_GD_NAO_CADASTRADA` com `s_elec` entre `0,2799` e `0,5091` (`< 0,5131`).
5. **Correspondência candidatura × `is_elec_anomaly`:** Exata `1:1` (`6.037 == 6.037`). Nota: `meta__is_clean_for_train` (`63.081`) difere de `is_reliable_elec` (`62.857`) em `236` UCs limpas com `cov < 0,70` e `12` UCs com `cov >= 0,70` mas `meta__is_clean_for_train == False` (menos dias confiáveis).
6. **Diferença entre anomalia, hipótese diagnóstica e indicação de vistoria:** Anomalia estatística bruta (`s_elec >= 0,5131`) ocorre em `6.916` UCs; após gate `>= 70%`, `6.037` tornam-se `is_elec_anomaly` e candidatas à vistoria; `16` adicionais recebem hipótese diagnóstica de GD por regra física pura sem superar o limiar estatístico.
7. **Reconciliação OOT × Regras Diárias (19/11):** Interseção = `2.069`, exclusivos OOT = `3.968` (`2.069 + 3.968 = 6.037`), exclusivos regras = `2.909` (`2.069 + 2.909 = 4.978`).
8. **Distribuição por circuito (anonimizada):**

| Circuito (LGPD) | Cadastro (25/11) | União 4 Cortes | OOT (19/11) | Elegíveis OOT | Candidatos OOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circuito F-04 | 9.278 | 8.817 | 8.753 | 8.667 | 434 |
| Circuito F-06 | 7.912 | 7.628 | 7.575 | 7.497 | 289 |
| Circuito F-08 | 6.916 | 6.585 | 6.545 | 6.482 | 464 |
| Circuito F-05 | 5.679 | 5.253 | 5.181 | 5.054 | 517 |
| Circuito F-02 | 5.355 | 5.128 | 5.098 | 5.033 | 726 |
| None | 5.285 | 0 | 0 | 0 | 0 |
| Circuito F-13 | 4.888 | 4.675 | 4.641 | 4.570 | 313 |
| Circuito F-14 | 4.267 | 4.104 | 4.083 | 4.063 | 229 |
| Circuito F-03 | 4.182 | 4.059 | 4.045 | 3.984 | 198 |
| Circuito F-11 | 3.886 | 3.569 | 3.531 | 3.480 | 454 |
| Circuito F-09 | 3.741 | 3.593 | 3.573 | 3.521 | 203 |
| Circuito F-19 | 2.522 | 2.408 | 2.400 | 2.360 | 154 |
| Circuito F-16 | 2.167 | 2.056 | 2.040 | 1.990 | 129 |
| Circuito F-22 | 1.686 | 1.613 | 1.606 | 1.551 | 120 |
| Circuito F-18 | 1.602 | 1.542 | 1.537 | 354 | 21 |
| Circuito F-17 | 249 | 215 | 210 | 168 | 32 |
| Circuito F-07 | 220 | 189 | 187 | 46 | 13 |
| Circuito F-21 | 40 | 25 | 21 | 21 | 11 |
| Circuito F-20 | 26 | 20 | 20 | 16 | 4 |
| Circuito F-12 | 3 | 0 | 0 | 0 | 0 |
| Circuito F-01 | 1 | 0 | 0 | 0 | 0 |
| Circuito F-15 | 1 | 0 | 0 | 0 | 0 |
| Circuito F-10 | 1 | 0 | 0 | 0 | 0 |
| None | 0 | 4.713 | 0 | 0 | 0 |
| None | 0 | 0 | 4.640 | 4.000 | 1.726 |


## 4. Auditoria Numérica Detalhada: 6.038 (Ensaio Congelado) vs. 6.037 (Carteira OOT)

O indicador `ref70_diff_count = 1` em `docs/tcc_evidence/p0_audit/p0_summary.json` isola a única unidade consumidora com comportamento diferenciado entre a partição numérica direta e a carteira de vistoria pós-regras de negócio:

- **UC:** `109855540`
- **Telemetria:** Cobertura de $93,61\%$ (29 dias confiáveis de 30).
- **Decomposição do Escore:**
  - Ensemble não supervisionado ponderado: $0{,}363082$
  - Bônus por flag de violação de regra: $+0{,}150000$
  - Escore elétrico consolidado não arredondado: $S_{elec} = 0{,}513082$
  - Limiar de corte congelado exato no treino: $\tau_{exact} = 0{,}513072$
  - Diferença sobre o limiar: $+0{,}000010$ ($10^{-5}$)
- **Motivo da Diferenciação:**
  - No artefato OOT (`experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet`), a UC foi atribuída à categoria `MUDANCA_CLASSE_COMPORTAMENTAL` (trajetória de queda abrupta de consumo de C3 para C1) e, sob a árvore de decisão operacional, recebeu `IS_CANDIDATO_VISTORIA = False` (retendo 6.037 candidatas).
  - No ensaio de sensibilidade do gating congelado, que avalia estritamente a partição física do filtro de qualidade sobre os escores numéricos elegíveis ($S_{elec} \ge \tau_{frozen}$ com cobertura $\ge 70\%$ e excluindo apenas novas UCs), a unidade satisfez a desigualdade numérica ($0{,}513082 \ge 0{,}513072$) e foi contabilizada entre os 6.038 casos retidos.
  - As demais 6.037 UCs coincidem estritamente entre os dois conjuntos (`ref70_exact_match_oot_cands` com simetria 100%).
