# Changelog Técnico da Consolidação Experimental

- **`src/eda/cockpit_metrics.py` (Novo):** Motor único de cálculo e reconciliação de KPIs (`unique_ucs` vs `all_windows`) e filtros do Cockpit.
- **`scripts/run_web_cockpit.py` (Atualizado):** Reconciliação dos cards KPI, painel analítico, filtros e exportação CSV; substituição de nomenclatura ("Anomalias Confirmadas" → "UCs c/ Anomalia Detectada"; "Diagnóstico Causal" → "Hipótese Diagnóstica").
- **`tests/test_cockpit_metrics.py` (Novo):** Testes automatizados garantindo consistência matemática entre KPIs, filtros e ordenação de candidatos.
- **`tests/test_tcc_invariants.py` (Novo):** Testes de invariantes (ausência de leakage temporal, blindagem de subespaço, isolamento por gating e reprodutibilidade por seed).
- **`src/tcc_evidence/*` & `scripts/generate_tcc_evidence.py` (Novo):** Orquestrador local idempotente com checkpoints, manifesto de proveniência, auditoria do Data Lake, EDA de 12 figuras, experimentos de ML (6 detectores + estabilidade), estudo de gating, comportamento/CPD, ablação M0→M5, estudos de caso mascarados e simulador econômico/operacional.
