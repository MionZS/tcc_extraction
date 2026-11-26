# Sensibilidade dos Pesos do Ensemble e do Bônus de Corroboração (P0 — Seções 7 e 8)

- **Execução:** `20261126_172200_p0_audit_sensitivity` | População: `62.857` UCs elegíveis no OOT (escores individuais congelados; zero *fit* no OOT)
- **Pesos efetivos confirmados no código (`src/models/composite.py` e `frozen_multiview_model.joblib`):** `IF=0,22`, `LOF=0,20`, `ECOD=0,18`, `COPOD=0,18`, `HBOS=0,10`, `PCA=0,12`. *(Nota: divergência identificada frente à tabela histórica que listava ECOD=0,20, HBOS=0,18, LOF=0,10 — ambas avaliadas abaixo.)*
- **Figuras:** `fig_p0_05_sensibilidade_pesos.pdf`, `fig_p0_06_variacao_bonus_corroboracao.pdf` e `fig_p0_07_comparacao_carteiras_topk.pdf`

## 1. Sensibilidade dos Pesos (Protocolo A: Top-2.000 Fixo × Protocolo B: Limiar Fixo e Recalibrado no Treino)

| Configuração | IF | LOF | ECOD | COPOD | HBOS | PCA | Spearman | Kendall (5k) | Jacc. Top-2k | Trocas Top-2k | Prot.B τ=0,5131 | τ Treino P95 | Prot.B τ Calib. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atual (código congelado) | 0,220 | 0,200 | 0,180 | 0,180 | 0,100 | 0,120 | 1,0000 | 1,0000 | 1,0000 | 0 | 6.039 | 0,5131 | 6.039 |
| Tabela histórica doc. | 0,220 | 0,100 | 0,200 | 0,180 | 0,180 | 0,120 | 0,9954 | 0,9502 | 0,9203 | 83 | 6.890 | 0,5553 | 5.583 |
| Pesos uniformes (1/6) | 0,167 | 0,167 | 0,167 | 0,167 | 0,167 | 0,167 | 0,9986 | 0,9709 | 0,9361 | 66 | 5.754 | 0,4984 | 6.241 |
| IF +20% | 0,253 | 0,192 | 0,172 | 0,172 | 0,096 | 0,115 | 0,9999 | 0,9926 | 0,9861 | 14 | 6.291 | 0,5189 | 5.948 |
| IF -20% | 0,184 | 0,209 | 0,188 | 0,188 | 0,105 | 0,126 | 0,9998 | 0,9918 | 0,9841 | 16 | 5.955 | 0,5070 | 6.171 |
| LOF +20% | 0,211 | 0,231 | 0,173 | 0,173 | 0,096 | 0,115 | 0,9995 | 0,9840 | 0,9753 | 25 | 5.862 | 0,5064 | 6.100 |
| LOF -20% | 0,229 | 0,167 | 0,188 | 0,188 | 0,104 | 0,125 | 0,9994 | 0,9827 | 0,9724 | 28 | 6.405 | 0,5247 | 6.016 |
| ECOD +20% | 0,212 | 0,193 | 0,208 | 0,174 | 0,097 | 0,116 | 0,9999 | 0,9925 | 0,9871 | 13 | 6.345 | 0,5181 | 6.019 |
| ECOD -20% | 0,228 | 0,207 | 0,149 | 0,187 | 0,104 | 0,124 | 0,9999 | 0,9918 | 0,9930 | 7 | 5.878 | 0,5076 | 6.051 |
| COPOD +20% | 0,212 | 0,193 | 0,174 | 0,208 | 0,097 | 0,116 | 0,9999 | 0,9910 | 0,9881 | 12 | 6.121 | 0,5157 | 6.031 |
| COPOD -20% | 0,228 | 0,207 | 0,187 | 0,149 | 0,104 | 0,124 | 0,9998 | 0,9902 | 0,9831 | 17 | 5.952 | 0,5097 | 6.062 |
| HBOS +20% | 0,216 | 0,196 | 0,176 | 0,176 | 0,118 | 0,118 | 1,0000 | 0,9951 | 0,9940 | 6 | 6.050 | 0,5136 | 6.028 |
| HBOS -20% | 0,225 | 0,204 | 0,184 | 0,184 | 0,082 | 0,122 | 1,0000 | 0,9949 | 0,9910 | 9 | 6.053 | 0,5121 | 6.083 |
| PCA +20% | 0,215 | 0,195 | 0,176 | 0,176 | 0,098 | 0,141 | 0,9999 | 0,9942 | 0,9782 | 22 | 5.886 | 0,5060 | 6.136 |
| PCA -20% | 0,225 | 0,205 | 0,184 | 0,184 | 0,102 | 0,098 | 0,9999 | 0,9939 | 0,9782 | 22 | 6.337 | 0,5199 | 5.966 |
| Sem IF (5 det.) | 0,000 | 0,256 | 0,231 | 0,231 | 0,128 | 0,154 | 0,9928 | 0,9446 | 0,9093 | 95 | 5.515 | 0,4846 | 6.637 |
| Sem LOF (5 det.) | 0,275 | 0,000 | 0,225 | 0,225 | 0,125 | 0,150 | 0,9822 | 0,9012 | 0,8349 | 180 | 7.580 | 0,5904 | 5.291 |
| Sem ECOD (5 det.) | 0,268 | 0,244 | 0,000 | 0,220 | 0,122 | 0,146 | 0,9953 | 0,9471 | 0,9389 | 63 | 5.129 | 0,4827 | 6.136 |
| Sem COPOD (5 det.) | 0,268 | 0,244 | 0,220 | 0,000 | 0,122 | 0,146 | 0,9948 | 0,9396 | 0,8859 | 121 | 5.561 | 0,4971 | 6.096 |
| Sem HBOS (5 det.) | 0,244 | 0,222 | 0,200 | 0,200 | 0,000 | 0,133 | 0,9986 | 0,9710 | 0,9569 | 44 | 6.023 | 0,5097 | 6.134 |
| Sem PCA (5 det.) | 0,250 | 0,227 | 0,204 | 0,204 | 0,114 | 0,000 | 0,9978 | 0,9671 | 0,8984 | 107 | 7.116 | 0,5617 | 5.627 |
| Núcleo 4 (IF+ECOD+COPOD+HBOS) | 0,324 | 0,000 | 0,265 | 0,265 | 0,147 | 0,000 | 0,9778 | 0,8858 | 0,8399 | 174 | 8.511 | 0,6535 | 4.990 |
| Isolation Forest isolado | 1,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,9657 | 0,8482 | 0,6701 | 395 | 8.479 | 0,7207 | 4.454 |

## 2. Sensibilidade do Bônus de Corroboração (+0,15) — Seção 8

**Achado crítico auditado:** Em `src/models/composite.py` (L552), o padrão `"IMBALANCE_MAX"` capturou `x__hist_current_imbalance_max > 0,20` (`29.360` elegíveis = 46,77%), enquanto as violações estritas de desequilíbrio de tensão `> 20%` (`57`) ou reversão `> 85%` (`32`) somam apenas `89` elegíveis (`0,14%`).

| Regra Avaliada | Bônus | UCs c/ Flag | % Eleg. | Δ Médio Score | Spearman vs 25/11 | Jacc. Top-2k | Trocas Top-2k | Cands τ=0,5131 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regra_efetiva_25_11_incl_c_imb | 0,000 | 29.398 | 46,77% | 0,0000 | 0,9223 | 0,8824 | 125 | 3.438 |
| regra_efetiva_25_11_incl_c_imb | 0,075 | 29.398 | 46,77% | 0,0350 | 0,9835 | 0,9598 | 41 | 4.392 |
| regra_efetiva_25_11_incl_c_imb | 0,150 | 29.398 | 46,77% | 0,0699 | 1,0000 | 1,0000 | 0 | 6.039 |
| regra_efetiva_25_11_incl_c_imb | 0,225 | 29.398 | 46,77% | 0,1045 | 0,9875 | 0,9782 | 22 | 8.612 |
| regra_efetiva_25_11_incl_c_imb | 0,300 | 29.398 | 46,77% | 0,1384 | 0,9714 | 0,9685 | 32 | 12.276 |
| regra_estrita_tensao_e_reversao | 0,000 | 89 | 0,14% | 0,0000 | 0,9223 | 0,8824 | 125 | 3.438 |
| regra_estrita_tensao_e_reversao | 0,075 | 89 | 0,14% | 0,0001 | 0,9223 | 0,8771 | 131 | 3.449 |
| regra_estrita_tensao_e_reversao | 0,150 | 89 | 0,14% | 0,0002 | 0,9223 | 0,8674 | 142 | 3.460 |
| regra_estrita_tensao_e_reversao | 0,225 | 89 | 0,14% | 0,0002 | 0,9223 | 0,8579 | 153 | 3.460 |
| regra_estrita_tensao_e_reversao | 0,300 | 89 | 0,14% | 0,0003 | 0,9223 | 0,8519 | 160 | 3.460 |

**Interpretação rigorosa:**
- Sob **orçamento fixo (Protocolo A, Top-2.000)**, o ranking é altamente estável a variações de ±20% nos pesos (apenas 6 a 28 trocas em 2.000; Jaccard `0,9724–0,9940`), mas sensível à remoção de detectores estruturalmente distintos (`Sem LOF`: 180 trocas; `IF isolado`: 395 trocas).
- Sob **limiar fixo (Protocolo B)**, o volume oscila fortemente se os pesos ou o bônus variarem sem recalibração do limiar no treino.
- Concordância com a configuração atual mede **estabilidade metodológica**, não comprovação de superioridade em detecção de fraude.
