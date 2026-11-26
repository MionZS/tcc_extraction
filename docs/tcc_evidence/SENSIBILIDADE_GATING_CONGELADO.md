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
