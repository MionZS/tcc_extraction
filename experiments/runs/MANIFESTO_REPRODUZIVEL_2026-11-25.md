# Manifesto Reproduzível dos Resultados — Suíte Consolidada (25/11/2026)

- **Commit base do código:** `c1539b4`
- **Índice da suíte:** `experiments/runs/latest_suite.json`

## 1. Execuções, Cortes, População e Configuração

| Experimento | `run_id` | Status | Invariantes | Commit | `config_sha256` (prefixo) | Seed |
|:---|:---|:---|:---:|:---|:---|---:|
| `audit` | `20261125_134532_048_audit` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |
| `daily_rules` | `20261125_134535_754_daily_rules` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |
| `oot_multiview` | `20261125_134559_972_oot_multiview` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |
| `ablation` | `20261125_134849_083_ablation_m0_m5` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |
| `gating` | `20261125_134940_636_gating_sensitivity` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |
| `score_stability` | `20261125_135109_678_score_stability` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |
| `case_traces` | `20261125_135558_880_case_traces` | `validated` | True | `c1539b4` | `8e011bc82efb36ae...` | 42 |

## 2. Modelos Congelados e Parâmetros de Normalização

| Artefato | Tamanho (bytes) | SHA-256 |
|:---|---:|:---|
| `experiments/runs/20261125_134559_972_oot_multiview/frozen_multiview_model.joblib.meta.json` | 630 | `34185e7472b972a322d2505639a56d76479052d93150e6f2b4ec069f2b0a3d37` |
| `experiments/runs/20261125_134559_972_oot_multiview/frozen_multiview_model.joblib.part00` | 26,214,400 | `cb9ac3cbe2be383d589dd99cd82aad1327c5ef7f4233e1a7dfb9cd6db7fe3ab1` |
| `experiments/runs/20261125_134559_972_oot_multiview/frozen_multiview_model.joblib.part01` | 20,394,232 | `c02d9747bb1f177fa992cff4b0e209098ce555cb01a62db8819a60a2abcf448d` |
| `experiments/runs/20261125_134559_972_oot_multiview/frozen_normalization_params.json` | 1,738 | `8d6d10b264299520844924e35074cbbfb0bae21130e2be8838d0f1cd529394f7` |

## 3. Tabelas de Fatos e Evidências

| Chave | Caminho | Tamanho (bytes) | SHA-256 |
|:---|:---|---:|:---|
| `oot_decisions` | `experiments/runs/20261125_134559_972_oot_multiview/decisions.parquet` | 5,032,997 | `2074d5b5ab9860a3601de73aabba20b5e4076cbb0fc07334bc28fba2923b3788` |
| `oot_calibrated_scores` | `experiments/runs/20261125_134559_972_oot_multiview/calibrated_scores.parquet` | 1,919,223 | `f1822e5afc7a2bc26ba7b97cefb1e3d5f78a0e81ab372b18dd3cf7325732b37c` |
| `oot_candidate_sets` | `experiments/runs/20261125_134559_972_oot_multiview/candidate_sets.parquet` | 612,764 | `2c5762b54a7e1fc3227b85cb20ea4f0d49d3fb8cd13983cb8fbf531077217be3` |
| `daily_metrics_csv` | `experiments/runs/20261125_134535_754_daily_rules/daily_metrics.csv` | 2,193 | `b4dec053750c1fdc9ca0320a01a76dee7069342e1f8702aaf37d013559cc976c` |
| `daily_decisions` | `experiments/runs/20261125_134535_754_daily_rules/decisions.parquet` | 19,707,958 | `143907466ec1889ab8b2e5496699d4b0dfd03dbcff2bcab30afda45dc97ebadb` |
| `ablation_funnel_csv` | `experiments/runs/20261125_134849_083_ablation_m0_m5/ablation_funnel.csv` | 305 | `6fa35f8a53cfc02d20307055e18eada36f3fbd45ad23532328c35d3a60a37097` |
| `ablation_subdetectors_csv` | `experiments/runs/20261125_134849_083_ablation_m0_m5/ablation_subdetectors.csv` | 821 | `264210d585a35714cb121e153a1f1144213139f2cd0acf8682decd1f8338cab9` |
| `gating_sensitivity_csv` | `experiments/runs/20261125_134940_636_gating_sensitivity/gating_sensitivity.csv` | 478 | `98f829a2f0ec23b3ebe0527313a401f2e6e2e9b7a280921bf8fe2b2d284ca836` |
| `score_stability_csv` | `experiments/runs/20261125_135109_678_score_stability/score_stability_comparison.csv` | 189 | `41c173f0f8a5fb9e15611aad36db7e2d5474a7c733d4c4c6b65ef5c0c3d45971` |
| `detector_audit_csv` | `experiments/runs/20261125_135109_678_score_stability/detector_individual_audit.csv` | 450 | `c020e531d470bd482b15aeb4c2e76414fe904e20a0b2fd5406f5d691fdac0e3b` |
| `case_traces_masked_csv` | `experiments/runs/20261125_135558_880_case_traces/case_traces_masked.csv` | 9,524 | `664d99810736c727d3a258aa7ca861fe8fb5d10ea1950f480158ab9f328ba271` |

## 4. Figuras Reproduzíveis do Manuscrito

- **Catálogo:** `docs/manuscript_figures/figure_catalog_manifest.json` (`sha256`: `82d8d6438b6fbb76f1e99e76b6eb5d8acaee4bbe6bf639a4148adecadf533ead`)

| `figure_id` | `run_id` | PDF | SHA-256 (PDF) |
|:---|:---|:---|:---|
| `fig_5_1a_isolation_forest_microcluster` | `controlled_benchmark_v1_seed42` | `docs/manuscript_figures/ml_controlled_panels/fig_5_1a_isolation_forest_microcluster.pdf` | `cdef7d53af95f8661f3958d7a4be464d19a2435df086d69eb91006485cacb994` |
| `fig_5_1b_lof_escala_propria` | `controlled_benchmark_v1_seed42` | `docs/manuscript_figures/ml_controlled_panels/fig_5_1b_lof_escala_propria.pdf` | `7af127c96dc3e61b1beb387492ce455336827a3fcabf92a528670e8e5478561d` |
| `fig_5_1c_contextualizacao_percentis` | `controlled_benchmark_v1_seed42` | `docs/manuscript_figures/ml_controlled_panels/fig_5_1c_contextualizacao_percentis.pdf` | `c37c22ee243cf4b7957636eaa04c2deda52160e1ffc5c673b2817c280dbcdd96` |
| `fig_caso_291xxx96_horarios_registradores` | `20261125_134559_972_oot_multiview` | `docs/manuscript_figures/real_case_studies/fig_caso_291xxx96_horarios_registradores.pdf` | `671d18d81b1b2ba3d34a7d7752a1911e7acb60e67fb5cbe7a83462a9a495c74d` |
| `fig_caso_685xxx20_reconciliacao_cobertura` | `20261125_134559_972_oot_multiview` | `docs/manuscript_figures/real_case_studies/fig_caso_685xxx20_reconciliacao_cobertura.pdf` | `f906e8d2d6587d41de52833d23cc3b2a915d16f6ad096852b57a1daa532ce990` |
| `fig_caso_908xxx00_fa_ra_completo` | `20261125_134559_972_oot_multiview` | `docs/manuscript_figures/real_case_studies/fig_caso_908xxx00_fa_ra_completo.pdf` | `154c4285e06a8f63d1819c619d7d04c62bad8fa6177f7e817cc7248bcbbf7b4e` |
| `fig_4_4a_tui_menu_governanca` | `20261123_124416_146_oot_multiview` | `None` | `None` |
| `fig_4_4b_tui_calendario_auditoria` | `20261123_124342_976_audit` | `None` | `None` |
| `fig_4_4c_cockpit_carteira_candidatos` | `20261123_124416_146_oot_multiview` | `None` | `None` |
| `fig_4_4d_cockpit_raiox_uc_908xxx00` | `20261123_124416_146_oot_multiview` | `None` | `None` |
