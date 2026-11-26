# Auditoria de Correlação e Orientação dos Detectores (P0 — Seção 5)

- **Execução:** `20261126_172200_p0_audit_sensitivity` | Coorte: OOT 19/11/2026 (`65.686` total; `62.857` elegíveis)
- **Figuras:** `docs/tcc_evidence/p0_audit/figures/fig_p0_02_correlacao_detectores_revisada.pdf` e `fig_p0_03_distribuicao_scores_individuais.pdf`

## 1. Verificação de Integridade e Investigação IF × ECOD

- **Alinhamento por UC (`max_abs_diff` vs. `decisions.parquet`):** `0,000000` em todos os 6 detectores e no consenso.
- **Simetria e diagonal:** Simétrica = `True`; Diagonal unitária = `True`; NaNs = `0`.
- **Orientação dos escores:** Todos os 6 detectores estão orientados no mesmo sentido (maior escore = maior anomalia).
- **Correlação IF × ECOD:** **Estritamente positiva** — Spearman = **`+0,9626`** (elegíveis) e **`+0,9639`** (OOT total); Pearson = **`+0,9445`**. **Não existe correlação negativa entre nenhum par de detectores** (mínimo par a par = `+0,1022` entre LOF e PCA).
- **Reconciliação dos pares LOF/PCA vs. consenso:** No OOT total (`65.686` UCs), `LOF = +0,4763` e `PCA = +0,4396`; nos elegíveis (`62.857` UCs), `LOF = +0,4609` e `PCA = +0,4135`. A diferença decorre exclusivamente do denominador populacional.

## 2. Matriz de Spearman — Escores Normalizados (Elegíveis OOT, n = 62.857)

| Detector | IF | LOF | ECOD | COPOD | HBOS | PCA | Consenso |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IF | 1,000 | 0,380 | 0,963 | 0,883 | 0,875 | 0,413 | 0,886 |
| LOF | 0,380 | 1,000 | 0,364 | 0,308 | 0,372 | 0,102 | 0,461 |
| ECOD | 0,963 | 0,364 | 1,000 | 0,913 | 0,852 | 0,396 | 0,881 |
| COPOD | 0,883 | 0,308 | 0,913 | 1,000 | 0,778 | 0,373 | 0,872 |
| HBOS | 0,875 | 0,372 | 0,852 | 0,778 | 1,000 | 0,379 | 0,876 |
| PCA | 0,413 | 0,102 | 0,396 | 0,373 | 0,379 | 1,000 | 0,413 |
| Consenso | 0,886 | 0,461 | 0,881 | 0,872 | 0,876 | 0,413 | 1,000 |

## 3. Matriz de Pearson — Escores Normalizados (Elegíveis OOT, n = 62.857)

| Detector | IF | LOF | ECOD | COPOD | HBOS | PCA | Consenso |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IF | 1,000 | 0,374 | 0,945 | 0,898 | 0,893 | 0,468 | 0,871 |
| LOF | 0,374 | 1,000 | 0,414 | 0,364 | 0,401 | 0,255 | 0,498 |
| ECOD | 0,945 | 0,414 | 1,000 | 0,942 | 0,882 | 0,463 | 0,901 |
| COPOD | 0,898 | 0,364 | 0,942 | 1,000 | 0,846 | 0,445 | 0,901 |
| HBOS | 0,893 | 0,401 | 0,882 | 0,846 | 1,000 | 0,507 | 0,901 |
| PCA | 0,468 | 0,255 | 0,463 | 0,445 | 0,507 | 1,000 | 0,531 |
| Consenso | 0,871 | 0,498 | 0,901 | 0,901 | 0,901 | 0,531 | 1,000 |

## 4. Distribuições, Proporção de Zeros, Saturação em 1 e Empates

| Detector | Bruto P50 | Bruto P95 | Norm. Média | Norm. DP | Zeros (%) | Sat. =1 (%) | Empates (%) | ρ vs Cons. (65.686) | ρ vs Cons. (62.857) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IF | 0,3504 | 0,5587 | 0,1813 | 0,2242 | 1,10% | 1,06% | 96,69% | 0,8858 | 0,8858 |
| LOF | 0,0409 | 0,2735 | 0,1334 | 0,1781 | 15,72% | 1,20% | 97,23% | 0,4763 | 0,4609 |
| ECOD | 0,6936 | 1,4331 | 0,2630 | 0,2158 | 1,16% | 1,83% | 97,46% | 0,8829 | 0,8812 |
| COPOD | 0,5492 | 1,2512 | 0,2327 | 0,2093 | 1,21% | 1,86% | 97,24% | 0,8755 | 0,8722 |
| HBOS | 0,3669 | 0,9170 | 0,1931 | 0,1923 | 7,92% | 0,93% | 97,33% | 0,8738 | 0,8756 |
| PCA | 0,0098 | 0,1098 | 0,0789 | 0,1472 | 1,08% | 1,08% | 97,13% | 0,4396 | 0,4135 |
