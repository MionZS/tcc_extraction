# Validação Temporal Multi-Corte (`19/11 → 25/11/2026`) e Auditoria `v2` Saneada (Opção A)

- **Execução:** `20261127_102118_p1_v2_temporal_validation` | **Commit:** `87e6146` | **Tempo:** `195,00 s`
- **Referência `v1` (19/11) preservada integralmente:** Sim (zero sobrescrita dos artefatos de 25/11).
- **Treino congelado (sem vazamento):** `<= 31/10/2026` (`189.341` janelas limpas). Limiar congelado `τ_v1 = 0,5131` vs. `τ_v2 = 0,5131`.

## 1. Comparação `v1` (Ref. 25/11) × `v2` Saneada nos Dois Cortes OOT (`19/11` e `25/11`)

Correções aplicadas exclusivamente na variante `v2`:
1. `strict_voltage_imbalance_rule=True` (elimina captura indevida de `current_imbalance_max > 0,20` no bônus `+0,15`).
2. `min_reliable_days=18` combinado a `avg_interval_coverage >= 0,70` (fecha brecha de blocos consecutivos ausentes).
3. Leitura particionada real de `output/events/alarm_events/` (ativa `S_tamper` sobre alarmes físicos de tampa/intrusão/corrente reversa).

| Execução | Total UCs | Elegíveis | Quarentena | Flag Regra (+0,15) | is_elec_anomaly | S_tamper >= 0,70 | Candidatos Vistoria |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 (19/11 — Ref. 25/11) | 65.686 | 62.857 | 2.829 | 30.694 (46,73%) | 6.037 | 0 | 6.037 |
| v1 (25/11 — +6 dias) | 65.640 | 62.274 | 3.366 | 30.649 (46,69%) | 6.059 | 0 | 6.059 |
| v2 (19/11 — Saneada) | 65.686 | 62.686 | 3.000 | 124 (0,19%) | 3.455 | 1.377 | 4.298 |
| v2 (25/11 — Saneada +6d) | 65.640 | 62.185 | 3.455 | 122 (0,19%) | 3.466 | 1.375 | 4.300 |

## 2. Persistência Temporal Real em +6 Dias (`19/11 → 25/11/2026`)

| Comparação Temporal | Elegíveis Comuns | Spearman | Cands. Persist. | Persist. Cands (%) | Jacc. Cands | Top-2k Persist. | Persist. Top-2k (%) | Jacc. Top-2k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 congelada (19/11 -> 25/11) | 62.207 | 0,9688 | 5.358 | 88,8% | 0,7952 | 1.797 | 89,8% | 0,8157 |
| v2 saneada (19/11 -> 25/11) | 62.115 | 0,9484 | 3.933 | 91,5% | 0,8431 | 1.760 | 88,0% | 0,7857 |
| v1 vs v2 no mesmo corte (19/11) | 62.686 | 0,9041 | 3.526 | 58,4% | 0,5178 | 1.848 | 92,4% | 0,8587 |

## 3. Verificação Controlada do Gate Endurecido `v2` (100 UCs reais)

| Cenário de Lacunas (n = 100 UCs reais) | v1 Quarentena (%) | v2 Quarentena (%) | v1 Vazamento Elét. (%) | v2 Vazamento Elét. (%) |
| --- | ---: | ---: | ---: | ---: |
| Lacuna aleatória 30% | 61,0% | 78,0% | 9,0% | 3,0% |
| Lacuna aleatória 50% | 100,0% | 100,0% | 0,0% | 0,0% |
| Bloco consecutivo 20% (6d) | 0,0% | 0,0% | 28,0% | 18,0% |
| Bloco consecutivo 30% (9d) | 0,0% | 0,0% | 27,0% | 17,0% |
| Bloco consecutivo 50% (15d) | 0,0% | 100,0% | 31,0% | 0,0% |

## 4. Evolução Longitudinal dos 3 Estudos de Caso Reais até `25/11/2026`

| UC (LGPD) | Dias Obs. (20–25/11) | FA Média 20–25/11 | RA Média 20–25/11 | Alarmes Tamper 30d | Score v1 (25/11) | Score v2 (25/11) | Categoria v2 (25/11) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 908***00 | 6 | 4,55 kWh/d | 14,72 kWh/d | 0 | 0,9327 | 0,7827 | POSSIVEL_GD_NAO_CADASTRADA |
| 291***96 | 0 | 0,00 kWh/d | 0,00 kWh/d | 2 | 1,0000 | 1,0000 | ANOMALIA_COMUNICACAO_ISOLADA |
| 685***20 | 6 | 71,59 kWh/d | 0,00 kWh/d | 0 | 1,0000 | 0,8937 | ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT |

## 5. Figuras Geradas

- `docs/tcc_evidence/v2_temporal_validation/figures/fig_v2_01_persistencia_temporal_19_25_nov.pdf`
- `docs/tcc_evidence/v2_temporal_validation/figures/fig_v2_02_comparacao_v1_vs_v2_auditoria.pdf`
- `docs/tcc_evidence/v2_temporal_validation/figures/fig_v2_03_evolucao_casos_reais_ate_25_nov.pdf`
