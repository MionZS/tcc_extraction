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
6. **`S_tamper` e alarmes (`cap08.tex` / `cap13.tex`):** Explicitar que a campanha de 25/11 operou com `x__alarm_count_window = 0` por leitura de arquivo único em vez do diretório particionado `output/events/alarm_events/` (que contém `1.706.242` registros na janela de 30d do OOT).

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
- O gating de qualidade elimina o vazamento de séries degradadas por telemetria para a fila prioritária (de 138–257 UCs para 0 na Base A; de 32,0% para 22,0% na Base B).
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
