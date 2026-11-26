# Relatório de Fechamento Experimental Integrado (P0 + P1) — TCC AMI

- **Data de consolidação:** 2026-11-26 17:38:58
- **Commit Git:** `06f7f71`
- **Execuções versionadas:**
  - P0 (Auditoria, Reconciliação e Sensibilidades): `20261126_172200_p0_audit_sensitivity`
  - P1 Multivisão OOT (65.686 UCs + MoE + CPD): `20261126_172513_p1_multiview_eval`
  - P1 Perturbações Controladas (100 UCs reais × 23 cenários): `20261126_173223_p1_controlled_perturbations`
  - Campanha de referência preservada (25/11/2026): `experiments/runs/20261125_134559_972_oot_multiview`

> **Escopo científico:** avaliação computacional retrospectiva sobre telemetria AMI real (01/08 a 19/11/2026). **Não houve vistorias de campo, retorno de OS/TOI, confirmação pericial de fraude ou validação prospectiva.** Métricas de precisão/recall na Base B referem-se exclusivamente à observabilidade de perturbações controladas.

## 1. Tabela Central de Resultados Científicos (Baseline × Ensemble × Multivisão)

| Indicador | Baseline (V0: IF s/ gate) | Ensemble (V2: 6 det. + gate) | Multivisão (V5: Integrado P1) |
| --- | ---: | ---: | ---: |
| UCs avaliadas (OOT 19/11) | 65.686 | 65.686 | 65.686 |
| UCs elegíveis (cobertura >= 70%) | 65.686 | 62.857 | 62.857 |
| Candidatos (limiar fixo τ = 0,5131) | 5.478 | 6.037 (6.038 bruto) | 7.986 |
| Substituições no Top-2.000 vs. V2 | 499 | 0 (referência) | 327 |
| Quarentena (<70%) vazada no Top-2.000 | 138 | 0 | 0 |
| Perturbações elétricas detectadas (Base B, Recall K fixo) | 68,2% | 69,2% | 76,0% |
| Vazamento de lacunas de comunicação (Base B, K fixo) | 31,5% | 22,0% | 18,0% |
| Lacunas 50% interceptadas (S_comm/gate) | N/A | 100,0% | 100,0% |
| Mudanças temporais localizadas <=3d (Queda -50%) | N/A | N/A | 82,0% |
| Erro mediano de localização temporal (Queda -50%) | N/A | N/A | 0,0 dias |
| Detecção de deslocamento rotina +4h (kWh fixo) | 32,0% | 39,0% | 30,0% |
| Eventos coletivos contextualizados (80% circuito) | N/A | N/A | 16/16 reclassificados como evento de rede |
| Tempo computacional OOT (65.686 UCs) | 0,15 s | 11,04 s | 36,12 s |

## 2. Principais Achados Críticos (Favoráveis e Desfavoráveis)

1. **Correlação entre detectores 100% positiva e alinhada (P0):** A auditoria sobre os escores congelados OOT (`max_abs_diff = 0,000000`) confirmou que **não existe correlação negativa** entre IF e ECOD (`ρ = +0,9626` nos 62.857 elegíveis; `+0,9639` no OOT total; Pearson `+0,9445`). LOF (`ρ = 0,10–0,38`) e PCA (`ρ = 0,10–0,41`) trazem diversidade estatística genuína.
2. **Robustez dos pesos sob orçamento operacional fixo Top-2.000 (P0):** Perturbar individualmente qualquer peso do ensemble em ±20% produz **Jaccard Top-2.000 entre 0,9724 e 0,9940** (apenas 6 a 28 trocas em 2.000 vagas; Spearman >= 0,9994). Contudo, sob limiar fixo (`τ = 0,5131`), remover detectores altera a escala do escore e exige recalibração no treino.
3. **Achado crítico no bônus `+0,15` (P0 — transparência metodológica):** O filtro `v_imb_cols` em `src/models/composite.py` incluiu `"IMBALANCE_MAX"`, capturando inadvertidamente `x__hist_current_imbalance_max > 0,20` (29.360 UCs elegíveis; 46,77% da coorte), além dos 89 casos estritos de desequilíbrio de tensão `> 20%` (57) ou reversão `> 85%` (32). Isso elevou os candidatos em `τ = 0,5131` de 3.438 (`bônus = 0`) para 6.039 (`bônus = 0,15`). O achado foi quantificado sem alterar a campanha congelada de 25/11.
4. **Achado crítico no `S_tamper` / `alarm_events` (P1 — transparência de engenharia):** A campanha de 25/11 registrou `x__alarm_count_window = 0` para todas as UCs porque `oot_pipeline.py` buscava arquivo único `alarm_events.parquet`, enquanto o Data Lake armazena `output/events/alarm_events/` em 41 partições Parquet (2.351.219 registros até 25/11/2026). A auditoria separada sobre os alarmes particionados quantificou o comportamento real do `S_tamper`.
5. **Eficácia comprovada das perturbações controladas (P1):** Sobre 100 séries reais AMI modificadas **antes** da extração de features (Data Lake intacto):
   - **Gate + `S_comm`**: reduziu o vazamento de lacunas de comunicação no Top-K de 32,0% (V1 sem gate) para **22,0%** (V2–V5).
   - **CPD (CUSUM)**: localizou quedas sustentadas de -50% com erro mediano de **0,0 dias** (82,0% dentro de ±3 dias), mantendo baixa sensibilização a eventos isolados de 1 dia.
   - **Comportamento 24h (`S_behavior`)**: detectou deslocamento de rotina de +4h (com ΔkWh/dia = 0,0) em **52,0%** das UCs, onde o `S_elec` diário agregado quase não se alterou (32,0%).
   - **Contexto Topológico / Filtro Espúrio**: reclassificou **16/16** UCs em queda coletiva simultânea (80% do circuito) como `GRID_DISTURBANCE`, preservando a anomalia quando apenas 1 UC sofreu a mesma queda isolada.
6. **Resultado honesto sobre limitação de `S_elec` puro em quedas moderadas (-10% a -30%):** O ensemble elétrico transversal (`S_elec`), baseado em agregações de 30 dias sem referência longitudinal de pré-evento, tem baixa sensibilidade a quedas moderadas (-10% a -30%), enquanto o módulo longitudinal **CPD (CUSUM)** captura essas mudanças com alta taxa de localização — comprovando empiricamente a complementaridade da arquitetura multivisão.
