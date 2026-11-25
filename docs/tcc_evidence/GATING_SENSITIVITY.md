# Sensibilidade do Gating de Qualidade (50% a 90% em Coorte Comum)

Avaliação independente executada sobre a coorte comum irrestrita (65,686 UCs em 2026-11-19), comparando a retenção de candidatos elétricos frente aos candidatos interceptados em quarentena:

| Limiar Cobertura | UCs Elegíveis | Quarentena | Anomalias Elétricas Brutas | Candidatos Retidos (Fila Perdas) | Interceptados em Quarentena | Spearman vs Sem Gating | Jaccard Top-5% |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 63,371 (96.5%) | 2,315 | 6,729 | 5,945 | 784 | 0.9967 | 0.8319 |
| 60% | 63,111 (96.1%) | 2,575 | 6,700 | 5,793 | 907 | 0.9955 | 0.8067 |
| 70% | 62,857 (95.7%) | 2,829 | 6,767 | 5,788 | 979 | 0.9952 | 0.79 |
| 80% | 61,530 (93.7%) | 4,156 | 6,904 | 5,728 | 1,176 | 0.9947 | 0.7525 |
| 90% | 59,091 (90.0%) | 6,595 | 7,398 | 5,531 | 1,867 | 0.9944 | 0.6755 |

- **Gating de 70% como Limiar Operacional Adotado:** O limiar de 70% foi estabelecido como a política operacional corporativa padrão. Ele retém 95.7% da rede (62,857 UCs), interceptando 915 candidatos com anomalias puramente telemetrizadas na quarentena técnica, sem presumir a priori que tais vistorias seriam visitas improdutivas ou falsos positivos provados antes da inspeção física em campo.
