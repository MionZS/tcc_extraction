# Guia de Reprodução Completa das Evidências

## 1. Ambiente e Versões Registradas
- **Commit Git / Tag:** `2f668f6`
- **Python:** `3.14.7`
- **Polars:** `1.562.8`
- **Scikit-Learn:** `1.9.1`

## 2. Comandos Exatos de Reprodução (100% Local)

```cmd
:: 1. Executar suíte de testes automatizados (incluindo invariantes e reconciliação do Cockpit)
uv run pytest tests -q

:: 2. Gerar todas as evidências do TCC em modo amostrado rápido (seed fixa = 42)
uv run python scripts/generate_tcc_evidence.py --mode sample --sample-size 3000 --seed 42

:: 3. Gerar evidências completas sobre 100% da população consolidada
uv run python scripts/generate_tcc_evidence.py --mode full --seed 42

:: 4. Retomar execução interrompida via checkpoints
uv run python scripts/generate_tcc_evidence.py --mode full --resume

:: 5. Iniciar Cockpit Web Reconciliado
uv run python scripts/run_web_cockpit.py --port 8050
```
