# Pendências Externas para Validação de Campo

1. **Integração de Ordens de Inspeção (TOI / Fiscalização Comercial):** Necessária para computar Precision@K, Recall@K, Yield@K e energia recuperada com rótulos independentes.
2. **Atributo `BAIRRO` no Cadastro CIS:** Permanece nulo na extração legada; geolocalização via `LATITUDE`/`LONGITUDE` e `ALIMENTADOR` permanece ativa.
3. **Atualização Contínua de Dezembro/2026:** Após fechamento mensal MDM, executar `scripts/run_tcc_experiments.py all --config configs/experiments/december_2026.yaml` e comparar contra novembro via `compare-runs`.
