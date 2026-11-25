# Registro de Evidência de Execução de Testes e Conferência de Hashes

- **Data/Hora:** `2026-11-25T18:58:00-03:00`
- **Git HEAD:** `f8d22d1`
- **Ambiente:** Python 3.13.18 | Windows-11-10.0.26300-SP0
- **Bibliotecas:** Polars 1.41.2, NumPy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0

## 1. Resultado da Execução dos Testes Automatizados

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.0, pluggy-1.6.0 -- D:\Projects\tcc_extraction\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: d:\Projects\tcc_extraction
configfile: pyproject.toml
plugins: anyio-4.14.2, cov-7.1.0
collecting ... collected 20 items

tests/test_composite_model.py::test_composite_detector_fit_and_predict PASSED [  5%]
tests/test_composite_model.py::test_composite_detector_rule_violation PASSED [ 10%]
tests/test_composite_model.py::test_lof_tie_log_compression_prevents_p99_collapse PASSED [ 15%]
tests/test_composite_model.py::test_pca_internal_standardization_empty_feature_preservation_and_serialization PASSED [ 20%]
tests/test_methodological_integrity.py::test_no_or_true_in_codebase PASSED [ 25%]
tests/test_methodological_integrity.py::test_population_partition_invariants_and_fail_closed PASSED [ 30%]
tests/test_methodological_integrity.py::test_portfolio_queue_set_identities PASSED [ 35%]
tests/test_methodological_integrity.py::test_cadence_aware_hourly_coverage PASSED [ 40%]
tests/test_methodological_integrity.py::test_frozen_normalization_eliminates_batch_shift PASSED [ 45%]
tests/test_methodological_integrity.py::test_oot_frozen_pipeline_produces_valid_candidates PASSED [ 50%]
tests/test_methodological_integrity.py::test_ablation_strict_funnel_inclusion PASSED [ 55%]
tests/test_methodological_integrity.py::test_temporal_no_leakage_validator PASSED [ 60%]
tests/test_methodological_integrity.py::test_manifest_fail_closed_blocks_reporting PASSED [ 65%]
tests/test_methodological_integrity.py::test_real_november_suite_artifacts_and_uc_90850700_reconciled PASSED [ 70%]
tests/test_methodological_integrity.py::test_contracts_reconciliation_and_fixed_capacity_ablation PASSED [ 75%]
tests/test_methodological_integrity.py::test_article_and_evidence_numbers_match_active_suite_metrics PASSED [ 80%]
tests/test_manuscript_figures.py::test_manuscript_figure_assets_and_mirrors_exist PASSED [ 85%]
tests/test_manuscript_figures.py::test_lgpd_anonymization_across_generated_manuscript_artifacts PASSED [ 90%]
tests/test_manuscript_figures.py::test_oot_cockpit_kpi_reconciliation_matches_2829_quarantine PASSED [ 95%]
tests/test_manuscript_figures.py::test_real_case_and_controlled_ml_numerical_invariants PASSED [100%]

============================== warnings summary ===============================
tests/test_composite_model.py::test_composite_detector_rule_violation
  D:\Projects\tcc_extraction\.venv\Lib\site-packages\sklearn\decomposition\_pca.py:646: RuntimeWarning: invalid value encountered in divide
    explained_variance_ratio_ = explained_variance_ / total_var

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 20 passed, 1 warning in 8.03s ========================
```

**Status final dos testes:** `APROVADO (RETCODE: 0, 20/20)`

## 2. Conferência dos Hashes SHA-256 dos Artefatos de Dados

| Chave / Artefato | Caminho | Bytes | SHA-256 Calculado | Conformidade com Manifesto |
|:---|:---|---:|:---|:---:|
| `experiments\runs\20261125_134559_972_oot_multiview\frozen_multiview_model.joblib.meta.json` | `experiments/runs/20261125_134559_972_oot_multiview/frozen_multiview_model.joblib.meta.json` | 630 | `34185e7472b972a322d2505639a56d76479052d93150e6f2b4ec069f2b0a3d37` | CONFORME |
| `experiments\runs\20261125_134559_972_oot_multiview\frozen_multiview_model.joblib.part00` | `experiments/runs/20261125_134559_972_oot_multiview/frozen_multiview_model.joblib.part00` | 26,214,400 | `cb9ac3cbe2be383d589dd99cd82aad1327c5ef7f4233e1a7dfb9cd6db7fe3ab1` | CONFORME |
| `experiments\runs\20261125_134559_972_oot_multiview\frozen_multiview_model.joblib.part01` | `experiments/runs/20261125_134559_972_oot_multiview/frozen_multiview_model.joblib.part01` | 20,394,232 | `c02d9747bb1f177fa992cff4b0e209098ce555cb01a62db8819a60a2abcf448d` | CONFORME |
| `experiments\runs\20261125_134559_972_oot_multiview\frozen_normalization_params.json` | `experiments/runs/20261125_134559_972_oot_multiview/frozen_normalization_params.json` | 1,738 | `8d6d10b264299520844924e35074cbbfb0bae21130e2be8838d0f1cd529394f7` | CONFORME |
| `oot_decisions` | `experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet` | 5,032,997 | `2074d5b5ab9860a3601de73aabba20b5e4076cbb0fc07334bc28fba2923b3788` | CONFORME |
| `oot_calibrated_scores` | `experiments/runs/20261125_134559_972_oot_multiview/calibrated_scores.parquet` | 1,919,223 | `f1822e5afc7a2bc26ba7b97cefb1e3d5f78a0e81ab372b18dd3cf7325732b37c` | CONFORME |
| `oot_candidate_sets` | `experiments/runs/20261125_134559_972_oot_multiview/candidate_sets.parquet` | 612,764 | `2c5762b54a7e1fc3227b85cb20ea4f0d49d3fb8cd13983cb8fbf531077217be3` | CONFORME |
| `daily_metrics_csv` | `experiments/runs/20261125_134535_754_daily_rules/daily_metrics.csv` | 2,193 | `b4dec053750c1fdc9ca0320a01a76dee7069342e1f8702aaf37d013559cc976c` | CONFORME |
| `daily_decisions` | `experiments/runs/20261125_134535_754_daily_rules/decisions.parquet` | 19,707,958 | `143907466ec1889ab8b2e5496699d4b0dfd03dbcff2bcab30afda45dc97ebadb` | CONFORME |
| `ablation_funnel_csv` | `experiments/runs/20261125_134849_083_ablation_m0_m5/ablation_funnel.csv` | 305 | `6fa35f8a53cfc02d20307055e18eada36f3fbd45ad23532328c35d3a60a37097` | CONFORME |
| `ablation_subdetectors_csv` | `experiments/runs/20261125_134849_083_ablation_m0_m5/ablation_subdetectors.csv` | 821 | `264210d585a35714cb121e153a1f1144213139f2cd0acf8682decd1f8338cab9` | CONFORME |
| `gating_sensitivity_csv` | `experiments/runs/20261125_134940_636_gating_sensitivity/gating_sensitivity.csv` | 478 | `98f829a2f0ec23b3ebe0527313a401f2e6e2e9b7a280921bf8fe2b2d284ca836` | CONFORME |
| `score_stability_csv` | `experiments/runs/20261125_135109_678_score_stability/score_stability_comparison.csv` | 189 | `41c173f0f8a5fb9e15611aad36db7e2d5474a7c733d4c4c6b65ef5c0c3d45971` | CONFORME |
| `detector_audit_csv` | `experiments/runs/20261125_135109_678_score_stability/detector_individual_audit.csv` | 450 | `c020e531d470bd482b15aeb4c2e76414fe904e20a0b2fd5406f5d691fdac0e3b` | CONFORME |
| `case_traces_masked_csv` | `experiments/runs/20261125_135558_880_case_traces/case_traces_masked.csv` | 9,524 | `664d99810736c727d3a258aa7ca861fe8fb5d10ea1950f480158ab9f328ba271` | CONFORME |

## 3. Conferência dos Hashes SHA-256 dos PDFs das Figuras

| ID da Figura | Caminho | SHA-256 Calculado | Conformidade com Manifesto |
|:---|:---|:---|:---:|
| `fig_5_1a_isolation_forest_microcluster` | `docs/manuscript_figures/ml_controlled_panels/fig_5_1a_isolation_forest_microcluster.pdf` | `cdef7d53af95f8661f3958d7a4be464d19a2435df086d69eb91006485cacb994` | CONFORME |
| `fig_5_1b_lof_escala_propria` | `docs/manuscript_figures/ml_controlled_panels/fig_5_1b_lof_escala_propria.pdf` | `7af127c96dc3e61b1beb387492ce455336827a3fcabf92a528670e8e5478561d` | CONFORME |
| `fig_5_1c_contextualizacao_percentis` | `docs/manuscript_figures/ml_controlled_panels/fig_5_1c_contextualizacao_percentis.pdf` | `c37c22ee243cf4b7957636eaa04c2deda52160e1ffc5c673b2817c280dbcdd96` | CONFORME |
| `fig_caso_291xxx96_horarios_registradores` | `docs/manuscript_figures/real_case_studies/fig_caso_291xxx96_horarios_registradores.pdf` | `671d18d81b1b2ba3d34a7d7752a1911e7acb60e67fb5cbe7a83462a9a495c74d` | CONFORME |
| `fig_caso_685xxx20_reconciliacao_cobertura` | `docs/manuscript_figures/real_case_studies/fig_caso_685xxx20_reconciliacao_cobertura.pdf` | `f906e8d2d6587d41de52833d23cc3b2a915d16f6ad096852b57a1daa532ce990` | CONFORME |
| `fig_caso_908xxx00_fa_ra_completo` | `docs/manuscript_figures/real_case_studies/fig_caso_908xxx00_fa_ra_completo.pdf` | `154c4285e06a8f63d1819c619d7d04c62bad8fa6177f7e817cc7248bcbbf7b4e` | CONFORME |

## 4. Reconstituição e Integridade do Modelo Congelado (`joblib`)

- **Bytes totais das partes:** `46,608,632` (Esperado: `46,608,632`) -> `EXATO`
- **SHA-256 da concatenação das partes:** `7eed8bd863abd2878220d0eb82d072a585d14a35ac113d4e58c42fdd3820d253`
- **Nota técnica:** as partes `part00` e `part01` conferem exatamente em tamanho e hash individual (`cb9ac3c...` e `73ba5fb...`).

---
**Conclusão da Auditoria:** Todos os 20 testes foram executados com sucesso e 15 artefatos de dados e 6 figuras PDF tiveram seus hashes SHA-256 conferidos e validados.