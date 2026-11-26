# Sensibilidade do Gating com Modelo Congelado (P0 — Seção 6)

- **Execução:** `20261126_172200_p0_audit_sensitivity` | **Protocolo:** Zero reajuste (*fit*) no OOT; modelo, normalizações, pesos e limiar (`τ = 0,5131`) estritamente congelados no treino (`<= 31/10/2026`).
- **Distinção metodológica:** O experimento anterior (`20261125_134940_636_gating_sensitivity`) reajustava o detector a cada limiar sobre o próprio OOT (protocolo distinto, preservado em `experiments/runs/`). A tabela abaixo isola o efeito puro da política de qualidade.
- **Figura:** `docs/tcc_evidence/p0_audit/figures/fig_p0_04_sensibilidade_gating_congelado.pdf`

| Cobertura Mín. | Elegíveis | Eleg. (%) | Quarentena | Anom. Brutas (fixo) | Retidos | Interceptados | Excluídos vs 70% | Novos vs 70% | Jaccard vs 70% | Jaccard Top-2k | Trocas Top-2k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 63.371 | 96,48% | 2.315 | 6.916 | 6.215 | 701 | 0 | 177 | 0,9715 | 0,9380 | 64 |
| 60% | 63.111 | 96,08% | 2.575 | 6.916 | 6.114 | 802 | 0 | 76 | 0,9876 | 0,9802 | 20 |
| 70% | 62.857 | 95,69% | 2.829 | 6.916 | 6.038 | 878 | 0 | 0 | 1,0000 | 1,0000 | 0 |
| 80% | 61.530 | 93,67% | 4.156 | 6.916 | 5.821 | 1.095 | 217 | 0 | 0,9641 | 0,9550 | 46 |
| 90% | 59.091 | 89,96% | 6.595 | 6.916 | 5.379 | 1.537 | 659 | 0 | 0,8909 | 0,8425 | 171 |

**Conclusões permitidas:**
- As sinalizações brutas permanecem rigorosamente constantes em **`6.916`** em todos os limiares, comprovando ausência total de *refit*.
- Elevar o limiar de 70% para 80% exclui 217 candidatas (46 trocas no Top-2.000; Jaccard 0,9550); elevar para 90% exclui 659 candidatas (171 trocas; Jaccard 0,8425).
- Sem rótulos externos de inspeção, **exclusões pelo gate não são denominadas falsos negativos**.


## Reconciliação Exata: 6.038 Retidos no Ensaio Congelado vs. 6.037 Candidatos OOT

Na auditoria P0 (`20261126_172200_p0_audit_sensitivity`), o indicador `ref70_diff_count` registrou exatamente `1` unidade consumidora de discrepância entre os escores elegíveis retidos a 70% (`6.038`) e a carteira OOT de vistoria (`6.037`, `IS_CANDIDATO_VISTORIA = True`).

### Registro Direto de Auditoria — UC 109855540
- **Artefato de Auditoria Dedicado:** `docs/tcc_evidence/p0_audit/uc_boundary_audit_109855540.json`
- **Identificador:** `id__uc_id = '109855540'` (Alimentador Chapada, SE Araucária)
- **Cobertura:** $93,61\%$ (`meta__data_quality = 0.9361`, `x__coverage_reliable_days = 29` de 30)
- **Escores dos 6 Detectores (Não Arredondados vs. Arredondados a 4 decimais no Parquet):**
  - IF: $0{,}456487$ (Parquet: $0{,}4565$, peso $0{,}22 	o 0{,}100427$)
  - LOF: $0{,}000000$ (Parquet: $0{,}0000$, peso $0{,}20 	o 0{,}000000$)
  - ECOD: $0{,}574060$ (Parquet: $0{,}5741$, peso $0{,}18 	o 0{,}103331$)
  - COPOD: $0{,}474655$ (Parquet: $0{,}4747$, peso $0{,}18 	o 0{,}085438$)
  - HBOS: $0{,}343435$ (Parquet: $0{,}3434$, peso $0{,}10 	o 0{,}034344$)
  - PCA: $0{,}329389$ (Parquet: $0{,}3294$, peso $0{,}12 	o 0{,}039527$)
- **Soma Base Ponderada:** $S_{base} = 0{,}363066$ (quando somados os valores de 4 casas: $0{,}363082$)
- **Bônus de Regra Física:** `rule_violation_flag = True` ($+0{,}150000$)
- **Escore Elétrico Consolidado:** $S_{elec} = 0{,}363066 + 0{,}150000 = 0{,}513066$ (armazenado com `round(consensus, 4)` como **$0{,}5131$** em `decisions.parquet`)
- **Limiar de Treino Congelado Exato:** $\tau_{exact} = 0{,}513072445231479$ (registrado como `0.513072` em `p0_summary.json`)
- **Trilha da Diferença:**
  1. No artefato OOT (`decisions.parquet`), a serialização arredonda o escore a 4 casas decimais ($0{,}5131$). Além disso, na árvore diagnóstica, a unidade foi classificada como `MUDANCA_CLASSE_COMPORTAMENTAL` (`IS_CANDIDATO_VISTORIA = False`).
  2. No ensaio de sensibilidade do gating congelado (`audit_experimental_results.py`), a partição avalia numericamente `s_elec_arr >= tau_frozen` sobre a coluna serializada ($0{,}5131 \ge 0{,}513072 \implies 	ext{True}$), retendo a UC entre os 6.038 casos.
  3. Com isso, a discrepância de exatamente 1 unidade (`ref70_diff_count = 1`) fica matematicamente e documentalmente auditada.
