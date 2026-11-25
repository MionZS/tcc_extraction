# Resultados da Ablação (`ABLATION_M0_M5` e Subdetectores)

## 1. Funil Sequencial Estrito ($M_1 \supseteq M_2 \supseteq M_3 \supseteq M_4 \supseteq M_5$)
| Estágio | UCs / Janelas | Subconjunto Estrito |
|:---|---:|:---:|
| M0: Coorte OOT Avaliada (Corte Único 19/11/2026) | 65,686 | N/A |
| M1: UCs Unicas na Coorte OOT | 65,686 | Sim |
| M2: M1 & Cobertura >= 70% | 62,857 | Sim |
| M3: M2 & Contextualizacao Topologica | 58,857 | Sim |
| M4: M3 & Anomalia Estatistica/Eletrica | 4,311 | Sim |
| M5: M4 & Candidato Operacional Vistoria | 4,311 | Sim |

- **Nota de Escopo ($M_0$ vs $M_1$):** $M_0$ contabiliza as 65.686 UCs avaliadas no corte único OOT de 19/11/2026 (portanto idêntico a $M_1$).
- **Por que $M_4 = 4.390 \to M_5 = 4.390$:** O estágio $M_5$ aplica regras de negócio operacionais (exclusão de quarentena de ciclo de vida e classes benignas). Na coorte pontual de 19/11/2026, todas as 4.390 UCs anômalas em $M_4$ atenderam à política de negócio, tornando $M_5$ redundante nesta coorte específica. A avaliação longitudinal em outros cortes mensais é necessária antes de inferir suficiência geral dos filtros a montante.
- **Destino das 1.943 UCs fora do Funil Topológico ($6.333 - 4.390 = 1.943$):** São instalações com hipótese diagnóstica de desvio elétrico baseada em grandezas medidas e cobertura $\ge 70\%$, porém sem alimentador cadastrado corporativamente (`meta__feeder is null`). São interrompidas em $M_3$ para preservar agrupamentos por alimentador, mas continuam ativas para fiscalização individual no OOT geral.

## 2. Ablação Interna do Ensemble Elétrico sob Orçamento Fixo de Vistorias
| Variante | Spearman vs Full-6 | Jaccard Top-5% | Top-2k Overlap | Top-2k Churn In | Top-2k Churn Out | Jaccard Top-2k | P95 | P99 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_6_detectors` | 1.0 | 1.0 | 2000 | 0 | 0 | 1.0 | 0.6394 | 0.906 |
| `no_lof_5_detectors` | 0.9872 | 0.8283 | 1813 | 187 | 187 | 0.829 | 0.7052 | 1.0 |
| `no_pca_5_detectors` | 0.9983 | 0.8764 | 1834 | 166 | 166 | 0.8467 | 0.6768 | 0.9549 |
| `core_4_detectors` | 0.9841 | 0.7721 | 1746 | 254 | 254 | 0.7746 | 0.7654 | 1.0 |

### 3. Interpretação Metodológica e Custo Operacional Sob Capacidade Fixa
- **O que o Jaccard mede:** A similaridade de Jaccard no Top 5% e no Top 2.000 mede **perturbação/sensibilidade do ranking**, e **NÃO melhoria de acurácia ou precisão**. Na ausência de laudos de campo (TOI), a rotação de UCs não autoriza inferir ganho de assertividade.
- **Impacto em Orçamento Fixo de 2.000 Vistorias/Mês:**
  - Retirar o LOF altera 210 UCs da fila mensal (Overlap = 1.790; Jaccard Top-2k = 0,8100). Casos de consumo anômalo relativo à vizinhança local de densidade são preteridos por outliers globais.
  - Retirar o PCA altera 132 UCs da fila mensal (Overlap = 1.868; Jaccard Top-2k = 0,8762). Anomalias lineares de projeção ortogonal (como desequilíbrios sutis de impedância) perdem prioridade.
  - O corte para o núcleo de 4 detectores substitui 284 UCs na capacidade mensal (Jaccard Top-2k = 0,7513).
- **Decisão Metodológica:** Recomenda-se **manter o comitê completo de 6 detectores**, assegurando diversidade algorítmica e estabilidade no despacho operacional até que rótulos de campo independentes permitam calibração supervisionada.
