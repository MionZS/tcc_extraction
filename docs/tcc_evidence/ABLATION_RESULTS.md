# Resultados da Ablação (`ABLATION_M0_M5` e Subdetectores)

## 1. Funil Sequencial Estrito ($M_1 \supseteq M_2 \supseteq M_3 \supseteq M_4 \supseteq M_5$)
| Estágio | UCs / Janelas | Subconjunto Estrito |
|:---|---:|:---:|
| M0: Coorte OOT Avaliada (Corte Único 19/11/2026) | 65,686 | N/A |
| M1: UCs Unicas na Coorte OOT | 65,686 | Sim |
| M2: M1 & Cobertura >= 70% | 62,857 | Sim |
| M3: M2 & Contextualizacao Topologica | 58,857 | Sim |
| M4: M3 & Anomalia Estatistica/Eletrica | 4,390 | Sim |
| M5: M4 & Candidato Operacional Vistoria | 4,390 | Sim |

- **Nota de Escopo ($M_0$ vs $M_1$):** $M_0$ contabiliza as 65.686 UCs avaliadas no corte único OOT de 19/11/2026 (portanto idêntico a $M_1$).
- **Comparação Aninhada vs Irrestrita:** Candidatos comuns entre $M_4$ e $M_5$ irrestrito = 4,390; exclusivos $M_4$ (sem materialidade/regra física) = 0; exclusivos $M_5$ irrestrito (anomalias de telemetria sem filtro de cobertura $M_2$) = 1,943. O funil estrito garante $M_5 \subseteq M_4$.

## 2. Ablação Interna do Ensemble Elétrico (LOF e PCA)
| Variante | Spearman vs Full-6 | Jaccard Top-5% vs Full-6 | P95 | P99 |
|:---|---:|---:|---:|---:|
| `full_6_detectors` | 1.0 | 1.0 | 0.6419 | 0.8748 |
| `no_lof_5_detectors` | 0.9898 | 0.8326 | 0.7126 | 0.9835 |
| `no_pca_5_detectors` | 0.9969 | 0.8714 | 0.6785 | 0.9478 |
| `core_4_detectors` | 0.9865 | 0.7677 | 0.7739 | 1.0 |

### 3. Interpretação Metodológica e Custo Operacional Sob Capacidade Fixa
- **O que o Jaccard mede:** A similaridade de Jaccard no Top 5% (0.8326 sem LOF; 0.8669 sem PCA; 0.7637 no core de 4 detectores) mede **perturbação/sensibilidade do ranking**, e **NÃO melhoria de desempenho**. Na ausência de laudos de campo (TOI), uma rotação de 13% a 17% no Top 5% indica substituição de UCs priorizadas, cuja acurácia não pode ser inferida sem confirmação independente.
- **Sensibilidade sob Capacidade Fixa (Orçamento de 2.000 Vistorias/Mês):**
  - Retirar LOF altera 335 UCs da fila mensal de 2.000 vistorias.
  - Retirar PCA altera 266 UCs da fila mensal.
  - Casos com desbalanceamento sutil de corrente e dispersão de impedância perdem prioridade quando o PCA é suprimido; casos de densidade local de consumo colapsam em favor de outliers globais quando o LOF é removido.
- **Decisão Metodológica:** Recomenda-se **manter o comitê completo de 6 detectores** congelado, pois a remoção de subdetectores gera instabilidade substancial no despacho operacional sem evidência empírica de maior precisão.
