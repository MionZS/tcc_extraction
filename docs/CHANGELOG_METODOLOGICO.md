# Changelog Metodológico

- **[Correção 1]** Congelamento de normalização P01–P99, ECDF e limiares em `CompositeAnomalyDetector.fit()` e `MultiViewAnomalyPipeline.fit()`, eliminando vazamento OOT e deslocamento de lote.
- **[Correção 2]** Separação explícita de `DAILY_RULES`, `OOT_MULTIVIEW` e `ABLATION_M0_M5` em runs versionadas sob `experiments/runs/<run_id>/`.
- **[Correção 3]** Cobertura horária ajustada à `CADENCE_MINUTES` em `src/eda/uc_timeseries.py`.
- **[Correção 4]** Remoção de expressões `or True` e implantação de validadores fail-closed de partição, balanço de conjuntos em fila e anti-vazamento temporal.
- **[Correção 5]** Harmonização da taxonomia diagnóstica (`ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`, proibição de confirmação causal via telemetria) e restauração do schema completo em `classified_ucs_latest.parquet`.
- **[Correção 6]** Correção do funil `M0 -> M5` garantindo inclusão estrita $M_5 \subseteq M_4 \subseteq M_3 \subseteq M_2 \subseteq M_1$.
- **[Proposta Pendente — Regra de Ciclo de Vida]** Em `src/models/category_reporting.py`, casos como `908***00` onde `DATA_INSTALACAO` é atualizada administrativamente no CIS (`2026-10-31`) mantendo o mesmo `NIO` com histórico de telemetria > 30 dias passam a cair em `INICIACAO_PERFIL_UC_NOVA` mesmo com `s_elec = 0.9601` e `RA_INTERVAL > 0`. Recomenda-se exigir ausência de histórico de telemetria prévia > 30 dias antes de aplicar `INICIACAO_PERFIL_UC_NOVA` (ou distinguir reprogramação bidirecional/troca lógica).
