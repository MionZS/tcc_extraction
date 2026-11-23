# Contratos de Scores e Auditoria dos Detectores ML

## 1. Decomposição Multi-Visão
- **`S_comm`**: `0.5 * IsoForest + 0.5 * ECOD` sobre features de cobertura, latência e alarmes.
- **`S_elec`**: Comitê de 6 detectores PyOD sobre features puramente elétricas (`select_pure_electrical_features`), ponderado por:
  - `iso_forest` (0.22), `lof` (0.20), `ecod` (0.18), `copod` (0.18), `hbos` (0.10), `pca` (0.12) + boost condicional PRODIST (+0.15 quando violação crítica de tensão/desbalanço/reversão).
- **`S_tamper`**: `0.70 * clip(alarms / 3, 0, 1) + 0.30 * clip(-slope / 5, 0, 1)` + elevação condicional por troca recente (`+0.15`).
- **Normalização Congelada:** Quantis P01–P99 capturados em `.fit()` sobre a coorte de treino limpa.
- **Limiar Congelado (`OOT_MULTIVIEW`):** `0.5226` (Percentil 95 da distribuição de treino).

## 2. Auditoria Individual dos 6 Subdetectores (Coorte OOT)
| Detector | Raw Min | Raw P50 | Raw P95 | Raw P99 | Raw Max | Norm Std | Saturação (=1.0) | Spearman vs Consenso |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `iso` | `0.3231` | `0.3525` | `0.569` | `0.6529` | `0.7753` | `0.2307` | `1.108%` | `0.8951` |
| `lof` | `0.9372` | `1.0437` | `1.3539` | `2.3481` | `7416227528.3357` | `0.1775` | `1.809%` | `0.4571` |
| `ecod` | `0.3582` | `0.7046` | `1.4533` | `2.0318` | `3.7249` | `0.2195` | `1.888%` | `0.8962` |
| `copod` | `0.2941` | `0.557` | `1.2769` | `1.8925` | `3.7114` | `0.2115` | `1.891%` | `0.8845` |
| `hbos` | `0.1992` | `0.3746` | `0.9744` | `1.495` | `3.3329` | `0.2015` | `1.146%` | `0.8773` |
| `pca` | `0.0` | `0.0074` | `0.0498` | `0.1064` | `5.1138` | `0.1723` | `1.002%` | `0.3532` |
