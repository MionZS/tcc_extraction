# Resultados da Ablação (`ABLATION_M0_M5` e Subdetectores)

## 1. Funil Sequencial Estrito ($M_1 \supseteq M_2 \supseteq M_3 \supseteq M_4 \supseteq M_5$)
| Estágio | UCs / Janelas | Subconjunto Estrito |
|:---|---:|:---:|
| M0: Janelas Totais Avaliadas (UC x Cutoff) | 65,686 | N/A |
| M1: UCs Unicas na Coorte OOT | 65,686 | Sim |
| M2: M1 & Cobertura >= 70% | 62,857 | Sim |
| M3: M2 & Contextualizacao Topologica | 58,857 | Sim |
| M4: M3 & Anomalia Estatistica/Eletrica | 4,390 | Sim |
| M5: M4 & Candidato Operacional Vistoria | 4,368 | Sim |

- Comparação não aninhada (M5 sem filtro de cobertura M2): comuns=4368, exclusivos M4=22, exclusivos M5 irrestrito=1936.

## 2. Ablação Interna do Ensemble Elétrico (LOF e PCA)
| Variante | Spearman vs Full-6 | Jaccard Top-5% vs Full-6 | P95 | P99 |
|:---|---:|---:|---:|---:|
| `full_6_detectors` | 1.0 | 1.0 | 0.6353 | 0.873 |
| `no_lof_5_detectors` | 0.9898 | 0.8326 | 0.7054 | 0.9828 |
| `no_pca_5_detectors` | 0.9969 | 0.8669 | 0.6719 | 0.9463 |
| `core_4_detectors` | 0.9865 | 0.7637 | 0.7653 | 1.0 |
