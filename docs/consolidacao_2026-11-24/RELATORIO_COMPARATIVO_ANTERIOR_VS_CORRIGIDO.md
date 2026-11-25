# Relatório Comparativo: Execução Anterior (23/11) × Execução Corrigida (25/11)

**Commit base:** `c1539b4`  
**Execução Anterior:** `20261123_124416_146_oot_multiview` / `20261123_124750_081_ablation_m0_m5`  
**Execução Corrigida:** `20261125_134559_972_oot_multiview` / `20261125_134849_083_ablation_m0_m5` / `20261125_134940_636_gating_sensitivity` / `20261125_135109_678_score_stability`

---

## 1. Comparação OOT, LOF, PCA, Ablação e Estabilidade

| Métrica | Anterior (23/11) | Corrigido (25/11) |
|---|---:|---:|
| População OOT avaliada (`19/11/2026`) | `65.686` | `65.686` |
| UCs elegíveis ($G_i=1$, cobertura $\ge 70\%$) | `62.857` | `62.857` |
| UCs em quarentena técnica (`< 70%`) | `2.829` | `2.829` |
| Candidatos OOT à vistoria | `6.333` | `6.037` |
| Baseline Isolation Forest (`0,643`) | `3.122` | `3.122` |
| Score médio populacional / $P_{95}$ | `0,2944` / `0,9115` | `0,2865` / `0,9438` |
| Limiar congelado ($\tau_{\text{train}}$) | `0,5226` | `0,5131` |
| Correlação Spearman LOF × consenso (`s_elec`, $n=62.857$) | `+0,4409` | `+0,4609` |
| $P_{95}$ LOF normalizado / % zeros / valores distintos | `0,4469` / `0,55%` / `6.005` | `0,4939` / `15,72%` / `6.407` |
| Correlação Spearman PCA × consenso (`s_elec`, $n=62.857$) | `+0,3369` | `+0,4135` |
| $P_{95}$ PCA normalizado / valores distintos | `0,4504` / `6.119` | `0,3273` / `4.907` |
| Funil topológico ($M_4 = M_5$) | `4.390` | `4.311` |
| Fora do funil topológico (`meta__feeder is null`) | `1.943` | `1.726` |
| Substituições Top-2.000 sem LOF (`Jaccard`) | `210` (`0,8100`) | `187` (`0,8290`) |
| Substituições Top-2.000 sem PCA (`Jaccard`) | `132` (`0,8762`) | `166` (`0,8467`) |
| Substituições Top-2.000 Core-4 (`Jaccard`) | `284` (`0,7513`) | `254` (`0,7746`) |
| Deslocamento máximo (`batch_minmax` vs `frozen_quantile`/`ecdf`) | `0,1865` vs `0,0` | `0,2291` vs `0,0` |

---

## 2. Reconciliação das Contagens entre Experimentos

| Experimento | Contagem | Critério e Escopo |
|---|---:|---|
| `OOT_MULTIVIEW` (`19/11/2026`) | `6.037` | Modelo congelado (ago–out) + política OOT + regras diagnósticas |
| `DAILY_RULES` (`19/11/2026`) | `4.978` | Regras físicas calibradas no corte final (média `5.387,7`/dia em `19` cortes) |
| `BASELINE_IF` (`19/11/2026`) | `3.122` | *Isolation Forest* isolado com limiar congelado `0,643` |
| `ABLATION_M0_M5` ($M_4 = M_5$) | `4.311` | Subconjunto estrito com alimentador cadastrado ($M_3 = 58.857$) |
| `GATING_SENSITIVITY` (`70%`) | `5.788` | Retidos no ensaio de sensibilidade (interceptados = `979`) |

---

## 3. Nota de Escopo

Todas as métricas acima decorrem exclusivamente de execução retrospectiva computacional sobre dados AMI reais. Não houve inspeções de campo, retorno de OS/TOI ou confirmação independente de irregularidades neste TCC.
