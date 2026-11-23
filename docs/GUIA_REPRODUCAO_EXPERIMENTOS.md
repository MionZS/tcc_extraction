# Guia de Reprodução de Experimentos (Novembro e Próximos Meses)

## Executar Suíte Completa em Config Versionada
```bash
uv run python scripts/run_tcc_experiments.py all --config configs/experiments/november_2026.yaml
```

## Subcomandos Individuais
- Auditoria de fontes: `uv run python scripts/run_tcc_experiments.py audit --config configs/experiments/november_2026.yaml`
- Série diária (`DAILY_RULES`): `uv run python scripts/run_tcc_experiments.py daily-rules --config configs/experiments/november_2026.yaml`
- Modelo congelado OOT (`OOT_MULTIVIEW`): `uv run python scripts/run_tcc_experiments.py oot-multiview --config configs/experiments/november_2026.yaml`
- Ablação M0-M5 e Subdetectores: `uv run python scripts/run_tcc_experiments.py ablation --config configs/experiments/november_2026.yaml`
- Comparar duas execuções: `uv run python scripts/run_tcc_experiments.py compare-runs --baseline <RUN_ID_A> --candidate <RUN_ID_B>`
