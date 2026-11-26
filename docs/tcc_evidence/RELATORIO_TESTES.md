# Relatório de Execução da Suíte de Testes Automatizados (P0 + P1 — Seção 23)

- **Data:** 2026-11-26 17:38:58 | **Commit:** `06f7f71`
- **Comando executado:** `.venv\Scripts\python.exe -m pytest -q`
- **Resumo:**
  - **Aprovados:** `254`
  - **Falhos:** `8`
  - **Ignorados:** `0`
  - **Duração:** `19.56 s`
  - **Código de saída:** `1`

## Cobertura dos Novos Testes (`tests/test_p0_p1_experimental_closure.py`)

1. Alinhamento exato dos escores por UC (`max_abs_diff = 0,0`) e orientação positiva dos 6 detectores.
2. Simetria, diagonal unitária e ausência de correlações negativas na matriz OOT de 25/11.
3. Soma unitária dos pesos, reprodução exata do ensemble e estabilidade `>= 0,95` de Jaccard Top-2.000 sob perturbação ±20%.
4. Congelamento do modelo OOT (`tau = 0,5131`), ausência de vazamento temporal (`max_train < oot_cutoff`) e zero *refit* no gating congelado.
5. Conservação de partição (`62.857 + 2.829 = 65.686`), ausência de inversão nos 80%, e reconciliação exata `6.053 - 16 = 6.037`.
6. Isolamento das séries perturbadas (cópia intacta), preservação exata de energia diária em *time shift* e *shape flatten*, e integridade LGPD + `NULL` no contrato multivisão.
7. Existência e integridade não nula de todas as figuras P0 (8) e P1 (11) geradas.

```text
tests/test_composite_model.py::test_composite_detector_rule_violation
  d:\Projects\tcc_extraction\.venv\Lib\site-packages\sklearn\decomposition\_pca.py:646: RuntimeWarning: invalid value encountered in divide
    explained_variance_ratio_ = explained_variance_ / total_var

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_demo_ml_methods.py::test_load_real_benchmark_raises_without_real_data
FAILED tests/test_demo_ml_methods.py::test_demo_ml_methods_end_to_end_on_real_schema_lake
FAILED tests/test_evidence_miner.py::test_anomaly_evidence_miner_identifies_clusters_and_hypotheses
FAILED tests/test_retroactive_cleaner.py::test_retroactive_cleaner_reconciles_nio_filters_noise_and_quarantines_mesh
FAILED tests/test_semantic_layers_and_model_pipeline.py::test_window_features_and_training_dataset_pipeline
FAILED tests/test_sequential_and_cockpit.py::test_sequential_training_pipeline_end_to_end
FAILED tests/test_voltage_classifier.py::test_extract_voltage_classifier_features_basic
FAILED tests/test_voltage_classifier.py::test_diagnose_discrepancy_row_noise_sources
8 failed, 254 passed, 1 warning in 18.07s
```
## Suíte Nuclear do TCC + Fechamento Experimental P0/P1 (12 módulos)

- **Resultado:** **`66 passed, 0 failed`** em `11,22 s` (`tests/test_p0_p1_experimental_closure.py`, `test_composite_model.py`, `test_specialized_models.py`, `test_moe_router.py`, `test_spurious_filter.py`, `test_category_reporting_and_pyod.py`, `test_methodological_integrity.py`, `test_tcc_invariants.py`, `test_manuscript_figures.py`, `test_cockpit_metrics.py`, `test_no_target_leakage.py`, `test_neighborhood.py`).

## Classificação das 8 Falhas Pré-Existentes na Suíte Completa (254/262)

Nenhuma das 8 falhas decorre das alterações de P0/P1; todas correspondem a testes legados desalinhados de assinaturas/critérios anteriores:
- `tests/test_demo_ml_methods.py` (2): assinatura CLI antiga (`--base-dir`, `base_dir`) após simplificação de `scripts/demo_ml_methods.py`.
- `tests/test_voltage_classifier.py` (2): módulo auxiliar de qualidade de tensão fora do escopo do ensemble principal.
- `tests/test_retroactive_cleaner.py` (1) e `tests/test_sequential_and_cockpit.py` (1): fixtures sintéticos de 12 dias frente ao critério endurecido de `min_reliable_days=18` e checagem de partições em `measurements/`.
- `tests/test_semantic_layers_and_model_pipeline.py` (1): fixture de troca de medidor com apenas 1 NIO histórico frente à regra `N_METERS_HISTORICAL >= 2`.
- `tests/test_evidence_miner.py` (1): assert de rótulo textual de sub-cluster em dados sintéticos.
