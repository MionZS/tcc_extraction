# Estabilidade de Normalização e Remoção de Extremos (Top 1%)

| Modo de Normalização | Avaliados | Remanescentes (-1%) | Deslocamento Médio | Deslocamento Máximo |
|:---|---:|---:|---:|---:|
| `batch_minmax` | 65,686 | 65,029 | 0.033952 | **0.1865** |
| `frozen_quantile` | 65,686 | 65,029 | 0.0 | **0.0** |
| `frozen_ecdf` | 65,686 | 65,686 | 0.0 | **0.0** |

Justificativa experimental: `frozen_quantile` (P01–P99 capturados exclusivamente no treino) e `frozen_ecdf` eliminam integralmente o deslocamento de lote (`0.0`).
