# Definição Formal e Separação dos Três Experimentos

Os três produtos experimentais do projeto possuem objetivos, grãos, populações e interpretações distintas e **não devem ter suas contagens misturadas**.

## 1. `OOT_MULTIVIEW` (Experimento Principal de Avaliação ML)
- **Papel no Artigo:** Avaliação central *out-of-time* do pipeline especializado multi-visão (`S_comm`, `S_elec`, `S_tamper`) com normalização e limiares congelados no treino.
- **Treino Congelado:** Cortes limpos `2026-08-31, 2026-09-30, 2026-10-31` (`189,341` amostras limpas).
- **Inferência OOT:** `2026-11-19` (`65,686` UCs; `62,857` elegíveis; `2,829` em quarentena).
- **Saída:** `6,333` anomalias elétricas; `6,304` candidatos prioritários à vistoria.

## 2. `DAILY_RULES` (Série Diária de Monitoramento e Regras)
- **Papel no Artigo:** Monitoramento diário da carteira, dinâmica de filas (`I_t`, `O_t`, `P_t`), evolução de cobertura e comparação frente à política não calibrada anterior.
- **Período:** `2026-11-01` a `2026-11-19` (`19` cortes diários; `1,248,501` janelas `UC × cutoff`).
- **Nota Metodológica:** Utiliza regras físicas calibradas e limiares explícitos (não chamar de saída ML). Média de `5,387.7` candidatos/dia (frente a `28,368` na regra antiga sem materialidade).

## 3. `ABLATION_M0_M5` (Contribuição Incremental e Ablação)
- **Papel no Artigo:** Isolar o impacto sequencial de saneamento, gating de cobertura e políticas operacionais (`M1 >= M2 >= M3 >= M4 >= M5`) além da ablação dos 6 subdetectores (com/sem LOF e PCA).
- **Inclusão Estrita Verificada:** `True`.
