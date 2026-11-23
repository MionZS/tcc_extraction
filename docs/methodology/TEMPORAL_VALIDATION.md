# Protocolo de Validação Temporal e Anti-Vazamento

1. **Separação Estrita Treino vs OOT:** Treino restrito a `2026-08-31, 2026-09-30, 2026-10-31` (`<= 2026-10-31`). Inferência em `2026-11-19` realizada exclusivamente via `.score()` usando `SimpleImputer`, `RobustScaler`, subdetectores, limites P01–P99 e threshold congelados.
2. **Construção de Janelas:** Todo corte `T` utiliza estritamente `REPORT_DAY in (T - 30d, T]` e `DATA_INSTALACAO <= T`.
3. **Invariante Fail-Closed:** `validate_temporal_no_leakage` bloqueia publicação caso exista qualquer registro futuro ao corte ou data de execução.
