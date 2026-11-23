# Changelog Metodológico

- **[Correção 1]** Congelamento de normalização P01-P99, ECDF e limiares em `CompositeAnomalyDetector.fit()` e `MultiViewAnomalyPipeline.fit()`, eliminando vazamento OOT e deslocamento de lote.
- **[Correção 2]** Separação explícita de `DAILY_RULES`, `OOT_MULTIVIEW` e `ABLATION_M0_M5` em runs versionadas sob `experiments/runs/<run_id>/`.
- **[Correção 3]** Cobertura horária ajustada à `CADENCE_MINUTES` em `src/eda/uc_timeseries.py`.
- **[Correção 4]** Remoção de expressões `or True` e implantação de validadores fail-closed de partição, balanço de conjuntos em fila e anti-vazamento temporal.
- **[Correção 5]** Harmonização da taxonomia diagnóstica (`ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`, proibição de confirmação causal via telemetria) e restauração do schema completo em `classified_ucs_latest.parquet`.
- **[Correção 6]** Correção do funil `M0 -> M5` garantindo inclusão estrita $M_5 \subseteq M_4 \subseteq M_3 \subseteq M_2 \subseteq M_1$.
- **[Correção 7]** Correção da regra de ciclo de vida e roteamento de UC Nova: UCs com histórico prévio de telemetria no mesmo medidor não são mais sequestradas por atualizações cadastrais administrativas; casos com evidência física de reversão de fluxo (ex.: `908***00`) são roteados para `POSSIVEL_GD_NAO_CADASTRADA` e priorizados para vistoria. Desacoplamento formal dos destinos operacionais (`VISTORIA_CAMPO_PERDAS` vs `MANUTENCAO_TECNICA_TELECOM`).
