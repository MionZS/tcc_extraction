# Ablação Metodológica Completa V0–V5 (Base A OOT Real × Base B Perturbações Controladas — Seção 19)

- **Orçamento constante:** `K = 2.000` na Base A (`65.686` UCs OOT) e `K = 400` na Base B (`800` séries controladas).
- **Preservação do funil M0–M5:** O funil sequencial existente (`65.686 → 62.857 → 58.857 → 4.311 → 4.311`) permanece preservado como experimento de seletividade da campanha de 25/11.
- **Figuras:** `fig_p1_16a_ablacao_variantes_base_a.pdf` e `fig_p1_16b_ablacao_variantes_base_b.pdf`

## 1. Base A — População AMI Real OOT (19/11/2026, K = 2.000)

| Variante | Elegíveis | Cands τ=0,5131 | Jacc. vs V0 | Jacc. vs V2 | Trocas vs V2 | Quar. no Top-2k | CPD>=30% Top-2k | Tempo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0: Baseline IF (sem gate) | 65.686 | 5.478 | 1,0000 | 0,6006 | 499 | 138 | 134 | 0,15 s |
| V1: Ensemble 6 det. (sem gate) | 65.686 | 6.916 | 0,5905 | 0,7723 | 257 | 257 | 159 | 11,04 s |
| V2: V1 + Gate Qualidade (>=70%) | 62.857 | 6.038 | 0,6006 | 1,0000 | 0 | 0 | 152 | 11,04 s |
| V3: V2 + Comportamento + MoE/Topol. | 62.857 | 6.255 | 0,5968 | 0,7241 | 320 | 0 | 156 | 26,85 s |
| V4: V3 + CPD (CUSUM 30d) | 62.857 | 6.450 | 0,5905 | 0,7167 | 330 | 0 | 198 | 35,82 s |
| V5: V4 + Regras Físicas/Multivisão | 62.857 | 7.986 | 0,5860 | 0,7190 | 327 | 0 | 198 | 36,12 s |

## 2. Base B — Séries Reais com Perturbações Controladas (Orçamento K Fixo)

| Variante | K Fixo | Recall Elét. | Prec. Contr. | F1 Contr. | Vaz. Telecom | Recall Quedas | Recall Shift 4h | Ref. Acion. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0: Baseline IF (sem gate) | 400 | 68,25% | 68,25% | 68,25% | 31,50% | 43,00% | 32,00% | 32,00% |
| V1: Ensemble 6 det. (sem gate) | 400 | 66,75% | 66,75% | 66,75% | 32,00% | 45,00% | 34,00% | 35,00% |
| V2: V1 + Gate Qualidade (>=70%) | 400 | 69,25% | 69,25% | 69,25% | 22,00% | 50,00% | 39,00% | 40,00% |
| V3: V2 + Comportamento 24h | 400 | 68,50% | 68,50% | 68,50% | 23,50% | 46,50% | 44,00% | 35,00% |
| V4: V3 + CPD (CUSUM) | 400 | 76,00% | 76,00% | 76,00% | 18,00% | 54,00% | 30,00% | 30,00% |
| V5: V4 + Regras Físicas (Multivisão) | 400 | 76,00% | 76,00% | 76,00% | 18,00% | 54,00% | 30,00% | 30,00% |

## 3. Análise Crítica (Incluindo Trade-offs)

- **V0 → V1 → V2 (Ensemble + Gate):** Na Base A, V0 e V1 vazam `138` e `257` UCs em quarentena (`< 70%`) para dentro do Top-2.000; o gate em V2 zera completamente esse vazamento (`0`). Na Base B, o vazamento de lacunas de comunicação cai de `32,0%` para **`22,0%`**.
- **V2 → V3 → V4 → V5 (Comportamento + CPD + Regras Físicas):** Na Base B, a inclusão de comportamento 24h eleva o recall do deslocamento +4h de `39,0%` para `44,0%`, e a inclusão de CPD + regras físicas eleva o F1 controlado das perturbações elétricas de `69,2%` (V2) para **`76,0%`** (V5).
- **Trade-off honesto:** Sob orçamento fixo `K`, elevar o peso de perturbações comportamentais puras (`V3`, sem queda de energia) compete por vagas com perturbações elétricas, podendo reduzir ligeiramente o F1 puramente elétrico antes da entrada de CPD e regras físicas (`V4–V5`). Além disso, nenhum ganho na Base B equivale automaticamente a ganho comprovado de *yield* em vistorias reais.
