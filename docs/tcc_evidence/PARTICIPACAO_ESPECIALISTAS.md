# Matriz Auditada de Participação dos Especialistas (P0 — Seção 9)

- **Execução:** `20261126_172200_p0_audit_sensitivity` | **Figura:** `docs/tcc_evidence/p0_audit/figures/fig_p0_08_matriz_participacao_especialistas.pdf`

| Componente | Implementado | Executado (25/11) | Avaliado Isoladamente | Participa da Candidatura (25/11) | Parâmetros / Pesos / Limiares |
| --- | --- | --- | --- | --- | --- |
| S_comm (Comunicação) | Sim (CommunicationAnomalyDetector: IF + ECOD) | Sim (65.686 UCs) | Sim (testes modulares + scores em decisions.parquet) | Indireto via gate de cobertura >= 70% (não entra na soma de s_elec) | IF(100) + ECOD (50/50); threshold_comm=0,6253; gate cov>=0,70 |
| S_elec (Ensemble Elétrico 6 det.) | Sim (CompositeAnomalyDetector: IF, LOF, ECOD, COPOD, HBOS, PCA) | Sim (65.686 UCs; 16 features elétricas puras) | Sim (auditoria individual + estabilidade + correlações) | Sim (núcleo primário: s_elec >= 0,5131 & is_reliable_elec) | Pesos IF=0,22, LOF=0,20, ECOD=0,18, COPOD=0,18, HBOS=0,10, PCA=0,12; tau=0,5131 |
| S_behavior (Comportamental) | Sim (BehavioralClusterer PCA+KMeans + classes C1–C6 + transições) | Sim (cluster_id e CLASSE_COMPORTAMENTAL em decisions.parquet) | Sim (eda_suite + ml_experiments + P1 perturbações) | Não (apenas caracterização/rotulagem de MUDANCA_CLASSE_COMPORTAMENTAL) | PCA(3) + KMeans(4) em X_elec; faixas C1..C6; transição >=2 cortes |
| CPD (CUSUM Mudança de Regime) | Sim (_detect_step_change_cusum em src/eda/uc_timeseries.py) | Parcial (sob demanda no Cockpit e nos estudos de caso; não no OOT em lote) | Sim (case_auditor + P1 avaliação completa em 65.686 UCs e perturbações) | Não na campanha 25/11 (integrado na variante experimental P1 V4/V5) | cov>=0,70, >=8d confiáveis, margem max(2, n//6), |delta|>=max(1,25*scale, 0,25*|pre|) |
| S_tamper (Eventos Físicos) | Sim (TamperAnomalyDetector: alarmes + slope negativo + troca) | Sim (calculado em decisions.parquet), porém 0 alarmes na janela OOT de Nov/2026 | Sim (test_specialized_models.py; histórico alarmes Jul–Set/2026) | Regra ativa (s_tamper>=0,70), mas 0 disparos efetivos no OOT (max s_tamper=0,30 só via slope) | 0,70*clip(alarms/3) + 0,30*clip(-slope/5) + 0,15*I(alarms>0)*troca_30d; limiar 0,70 |
| PRODIST / Regras Físicas | Sim (guardrails em CompositeAnomalyDetector + category_reporting.py) | Sim (rule_violation_flag + regras GD/Medidor Travado) | Sim (bonus_sensitivity.csv + test_composite_model.py) | Sim (+0,15 no consenso em 30.694 UCs — 30.654 via c_imb>0,20, 58 v_imb>20, 66 ra>0,85) | Bonus +0,15; GD: ra_ratio>=0,08 & ra_days>=2; Travado: fa_med<0,05 & is_anom |
| MoE Hierárquico (Alimentador/Global) | Sim (MoERouter + HierarchicalModelOrchestrator) | Não na campanha 25/11 (modelos v1 eram legado 4-det/21-feat); executado no P1 sem vazamento | Sim (test_moe_router.py + P1 avaliação hierárquica congelada) | Não na campanha 25/11 (avaliado na variante P1 V3/V5) | n>=1000: (0,70/0,20/0,10); 150<=n<1000: interpolação; n<150: (0,15/0,45/0,40) |
| Contexto Topológico / Vizinhança | Sim (contraste espacial em category_reporting.py + spurious_filter.py + funil M3) | Sim para Telecom (Isolada vs Gargalo Coletivo) e funil M3; SpuriousFilter avaliado no P1 | Sim (test_spurious_filter.py + test_neighborhood.py + P1 eventos coletivos) | Parcial: separa quarentena telecom e define funil M3 (4.311); 1.726 sem alimentador mantidas no OOT geral | Peer Posto>=3 -> Geo 250m -> Feeder; déficit>=0,30 & peer>=0,80; grid_event>=60% peers |

## Auditoria de `S_tamper` e Regras Físicas (Seção 17)

- **Descoberta de engenharia:** A campanha de 25/11 registrou 0 alarmes em `x__alarm_count_window` porque `oot_pipeline.py` buscava arquivo único `alarm_events.parquet`, enquanto o Data Lake armazena `output/events/alarm_events/` em `41` partições Parquet (`2.351.219` registros totais até 25/11/2026).
- **Auditoria direta dos alarmes particionados na janela OOT (21/10–19/11/2026):**
  - Total de alarmes na janela OOT: `1.706.242`
  - UCs OOT com >=1 alarme (qualquer tipo): `31.338`
  - UCs OOT com >=1 alarme de tamper/intrusão: `1.777`
  - UCs OOT com `S_tamper >= 0,70` (com alarmes reais): `1.377` (sobreposição com anomalia elétrica: `496`; exclusivos: `881`)

| Regra Física / Especialista | Acionamentos OOT | Sobreposição c/ Anom. Elétrica | Exclusivos (sem Anom. Elétrica) |
| --- | ---: | ---: | ---: |
| PRODIST Desequilíbrio Tensão (>20%) | 58 | 57 | 1 |
| PRODIST Reversão Extrema (>85%) | 66 | 32 | 34 |
| Regra Física GD (RA>=8% & >=2d) | 1.706 | 1.434 | 272 |
| Desequilíbrio Corrente (>20% — gatilho atual +0,15) | 30.654 | 5.453 | 25.201 |
| Desequilíbrio Corrente Severo (>45%) | 29.904 | 5.163 | 24.741 |
| Medidor Travado / Consumo Nulo (FA<0,05 & Anom.) | 546 | 545 | 0 |
| Tamper (s_tamper >= 0,70; 0 alarmes em Nov/2026) | 0 | 0 | 0 |
