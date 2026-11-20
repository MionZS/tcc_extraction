# Matriz de Rastreabilidade de Evidências do TCC

| Objetivo Científico / Crítica | Funcionalidade / Módulo | Experimento Executado | Artefato Gerado | Conclusão Observada |
|:---|:---|:---|:---|:---|
| Integridade e População Real | `run_data_lake_audit` | Auditoria UC×NIO, cobertura e partições | `data_quality/data_lake_audit.json` | População real quantificada com denominadores explícitos |
| Reconciliação de KPIs do Cockpit | `src/eda/cockpit_metrics.py` | Auditoria UC vs Janela + testes `pytest` | `cockpit/cockpit_kpi_reconciliation.csv` | Divergência 262.849 vs 66.099 e 704 vs 201 eliminada |
| Pipeline como Resultado de Engenharia | `run_pipeline_engineering_benchmark` | Benchmark controlado Lazy `sink_parquet` vs Eager | `pipeline/pipeline_engineering_summary.json` | Throughput e volumes reais medidos sem suposições |
| EDA Completa da População | `src/tcc_evidence/eda_suite.py` | 12 gráficos reais + tabelas CSV + interpretações | `eda/EDA_REPORT.md` | Assimetria, contraste forma/magnitude e falhas documentados |
| Complementaridade ML e Estabilidade | `run_ml_methods_and_fusion_audit` | 6 detectores PyOD + estabilidade Min-Max vs Congelado | `ml_methods/detector_comparison_metrics.csv` | Correlação de postos e impacto da referência congelada |
| Sensibilidade do Gating (50%-90%) | `run_gating_sensitivity_study` | Variação de limiar e contaminação no treino | `gating/gating_sensitivity.csv` | Trade-off entre cobertura elegível e retenção de alvos |
| Comportamento e CPD/CUSUM | `run_behavior_and_cpd_study` | Clusters, matriz UC×Dia e quebras CUSUM | `behavior/uc_day_behavioral_calendar.png` | Separação visual entre regime estável e degraus |
| Ablação M0 → M5 | `run_ablation_study` | Funil em coorte fixa + taxa de gaps | `ablation/ablation_funnel_metrics.csv` | Redução de falsos positivos ligados a falhas de comunicação |
| Estudos de Caso e Raio-X | `run_case_studies` | Dossiês individuais anonimizados | `case_studies/CASE_STUDIES_XRAY.md` | Rastreabilidade completa de 8 arquétipos reais |
| Capacidade e Economia | `run_operations_and_economics` | Coortes entre cutoffs + simulação paramétrica | `economics/economic_sensitivity_scenarios.csv` | Cenários de sensibilidade claramente separados de reais |
