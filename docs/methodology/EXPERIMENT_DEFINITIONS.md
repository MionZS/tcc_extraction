# Definição Formal e Separação dos Três Experimentos

Os três produtos experimentais do projeto possuem objetivos, grãos, populações e interpretações distintas e **não devem ter suas contagens misturadas**.

## 1. `OOT_MULTIVIEW` (Experimento Principal de Avaliação ML)
- **Papel no Artigo:** Avaliação central *out-of-time* do pipeline especializado multi-visão (`S_comm`, `S_elec`, `S_tamper`) com normalização e limiares congelados no treino.
- **Treino Congelado:** Cortes limpos `2026-08-31, 2026-09-30, 2026-10-31` (`189,341` amostras limpas).
- **Inferência OOT:** `2026-11-19` (`65,686` UCs; `62,857` elegíveis; `2,829` em quarentena).
- **Saída:** `6,333` anomalias elétricas; `6,333` candidatos prioritários à vistoria.

## 2. `DAILY_RULES` (Série Diária de Monitoramento e Regras)
- **Papel no Artigo:** Monitoramento diário da carteira, dinâmica de filas (`I_t`, `O_t`, `P_t`), evolução de cobertura e comparação frente à política não calibrada anterior.
- **Período:** `2026-11-01` a `2026-11-19` (`19` cortes diários; `1,248,501` janelas `UC × cutoff`).
- **Nota Metodológica:** Utiliza regras físicas calibradas e limiares explícitos (não chamar de saída ML). Média de `5,387.7` candidatos/dia (frente a `28,368` na regra antiga sem materialidade).

## 3. `ABLATION_M0_M5` (Contribuição Incremental e Ablação)
- **Papel no Artigo:** Isolar o impacto sequencial de saneamento, gating de cobertura e políticas operacionais (`M1 >= M2 >= M3 >= M4 >= M5`) além da ablação dos 6 subdetectores (com/sem LOF e PCA).
- **Inclusão Estrita Verificada:** `True`.

## 4. Reconciliação Contratual Entre Métodos
| Relação Analítica | Quantitativo | Causa Metodológica e Destino Operacional |
|:---|---:|:---|
| `OOT_MULTIVIEW` (6.333) vs Gating 70% (6.311) | +22 UCs | OOT aplica modelo congelado em treino + regras operacionais de negócio (descarta 394 casos não priorizados pela política e mantém 416 casos com sinais elétricos na cauda marginal com $s \in [0,48, 0,5225]$). Gating 70% é sensibilidade exploratória com corte estatístico cego ($s \ge 0,5226$). |
| Limiar 60% (`DAILY_RULES`) vs 70% (`OOT_MULTIVIEW` / $M_2$) | N/A | 70% é a política operacional mandatória para despacho de campo e funil $M_2$. 60% era tolerância usada em monitoramento diário rápido para oscilações transitórias de fim de semana. |
| Candidatas OOT fora do funil topológico | 1.943 UCs | Possuem cobertura $\ge 70\%$, score anômalo e hipótese de desvio elétrico, mas `meta__feeder` é nulo em `electrical_hierarchy.parquet`. São interrompidas em $M_3$ (análise topológica), mas permanecem ativas para vistoria individual em OOT. |
| Estágio $M_4$ (4.390) $\to$ $M_5$ (4.390) | 0 exclusões | Redundante na coorte avaliada de 19/11/2026. A suficiência dos filtros a montante deve ser avaliada longitudinalmente em outros cortes mensais antes de conclusões definitivas. |
