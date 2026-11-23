# Sensibilidade do Gating de Qualidade (50% a 90% em Coorte Comum)

Avaliação independente reexecutada sobre a coorte comum irrestrita (65,686 UCs em 2026-11-19), comparando a retenção de candidatos elétricos frente ao bloqueio de falsos alarmes em quarentena:

| Limiar Cobertura | UCs Elegíveis | Quarentena | Anomalias Elétricas Brutas | Candidatos Retidos (Fila Perdas) | Interceptados em Quarentena | Spearman vs Sem Gating | Jaccard Top-5% |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 63,371 (96.5%) | 2,315 | 7,169 | 6,451 | 718 | 0.9971 | 0.8503 |
| 60% | 63,111 (96.1%) | 2,575 | 7,126 | 6,295 | 831 | 0.9953 | 0.8225 |
| 70% | 62,857 (95.7%) | 2,829 | 7,226 | 6,311 | 915 | 0.9952 | 0.804 |
| 80% | 61,530 (93.7%) | 4,156 | 7,423 | 6,246 | 1,177 | 0.995 | 0.7701 |
| 90% | 59,091 (90.0%) | 6,595 | 7,722 | 5,822 | 1,900 | 0.9945 | 0.6837 |

- **Gating de 70% como Ponto Operacional Ótimo:** Preserva 95.7% da rede (62,857 UCs), enquanto intercepta ruídos de telecomunicação que gerariam ordens de fiscalização improdutivas em medidores com perda de pacotes.
