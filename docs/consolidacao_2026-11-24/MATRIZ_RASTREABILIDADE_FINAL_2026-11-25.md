# Matriz de Rastreabilidade Quantitativa Final — TCC AMI (25/11/2026)

**Referência de código:** commit `c1539b4` / suíte `experiments/runs/latest_suite.json` (execução de `25/11/2026`).  
**Escopo científico:** avaliação estritamente computacional retrospectiva sobre telemetria AMI real; sem vistoria de campo, OS/TOI ou confirmação independente de fraude.

---

## 1. Métricas Principais da Suíte Vigente (`25/11/2026`)

| Afirmação / Número no Manuscrito | Local no LaTeX (`eekBR`) | Execução (`run_id`) | Corte / Período | População / Escopo | Artefato de Origem |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `189.341` amostras elegíveis de ajuste | `cap13.tex` | `20261125_134559_972_oot_multiview` | 31/08, 30/09, 31/10/2026 | Treino OOT | `metrics.json` (`train_clean_rows`) |
| `65.686` UCs avaliadas; `62.857` elegíveis (`95,69%`); `2.829` quarentena (`4,31%`) | `00-dados.tex`, `cap13.tex`, `cap15.tex` | `20261125_134559_972_oot_multiview` | `2026-11-19` | Coorte OOT | `metrics.json` / `decisions.parquet` |
| `6.037` candidatas OOT à vistoria; média `0,2865`; $P_{95}=0{,}9438$; $\tau_{train}=0{,}5131$ | `00-dados.tex`, `cap08.tex`, `cap12.tex`, `cap13.tex`, `cap14.tex`, `cap15.tex` | `20261125_134559_972_oot_multiview` | `2026-11-19` | Coorte OOT | `metrics.json` |
| `3.122` selecionadas pelo baseline IF (`0,643`) | `cap13.tex` | `20261125_134559_972_oot_multiview` | `2026-11-19` | Coorte OOT | `metrics.json` (`oot_baseline_anomalies`) |
| Distribuição de classes (`44.674` / `12.084` / `4.087` / `1.829` / `1.451` / `910` / `515` / `71` / `65`) | `cap13.tex` (Tabela `\ref{classes}`) | `20261125_134559_972_oot_multiview` | `2026-11-19` | `65.686` UCs | `metrics.json` (`category_distribution`) |
| `1.248.501` janelas; média `5.387,7`/dia; mín `4.737` (15/11); máx `6.728` (02/11); `4.978` em 19/11 | `00-dados.tex`, `cap12.tex`, `cap13.tex` | `20261125_134535_754_daily_rules` | `01/11–19/11/2026` (19 cortes) | Carteira diária calibrada | `metrics.json` / `daily_metrics.csv` |
| Gating `50%..90%`; a `70%`: `5.788` retidos, `979` interceptados, Spearman `0,9952`, Jaccard Top-5% `0,7900` | `cap13.tex` (Tabela `\ref{gating}`, `figuras/funil.pdf`) | `20261125_134940_636_gating_sensitivity` | `2026-11-19` | `65.686` UCs | `metrics.json` / `gating_sensitivity.csv` |
| Funil $M_0=M_1=65.686$, $M_2=62.857$, $M_3=58.857$, $M_4=M_5=4.311$; `1.726` fora do funil (`feeder null`) | `00-dados.tex`, `cap13.tex`, `cap14.tex`, `cap15.tex`, `figuras/funil.pdf` | `20261125_134849_083_ablation_m0_m5` | `2026-11-19` | Coorte OOT + topologia | `metrics.json` / `ablation_funnel.csv` |
| Ablação Top-2.000: sem LOF `187` trocas (`0,8290`); sem PCA `166` (`0,8467`); Core-4 `254` (`0,7746`) | `00-dados.tex`, `cap13.tex`, `cap15.tex` (Tabela `\ref{ablacao}`) | `20261125_134849_083_ablation_m0_m5` | `2026-11-19` | Top-2.000 fixo | `metrics.json` / `ablation_subdetectors.csv` |
| Estabilidade: `batch_minmax` max `0,2291` (média `0,034047`); `frozen_quantile`/`ecdf` `0,0`; auditoria Spearman (`0,8858` IF, `0,4763` LOF, `0,8829` ECOD, `0,8755` COPOD, `0,8738` HBOS, `0,4396` PCA) | `cap13.tex`, `cap15.tex` | `20261125_135109_678_score_stability` | `2026-11-19` | `65.686` vs `65.029` (-1%) | `metrics.json` / `detector_individual_audit.csv` |
| Caso `908***00` (`NIO-435***63`): FA `18,05 -> 5,72 kWh/d`, 12d RA (`14–31,27 kWh/d`), razão `0,5069`, escore `1,0000` | `cap11.tex`, `cap13.tex`, `fig_caso_908xxx00_fa_ra_completo.pdf` | `20261125_135558_880_case_traces` / `oot_multiview` | `02/08–19/11/2026` | UC individual | `case_traces_masked.csv` / `decisions.parquet` |
| Caso `291***96` (`NIO-423***54`): `314` leituras `RA_TOTAL`, cadência `10 min` até `05/11`, silêncio `06–19/11` (`3,33%`, `None`) | `cap11.tex`, `cap13.tex`, `fig_caso_291xxx96_horarios_registradores.pdf` | `20261125_135558_880_case_traces` / `oot_multiview` | `01/08–19/11/2026` | UC individual | `fig_caso_291xxx96_horarios_registradores.meta.json` |
| Caso `685***20` (`NIO-437***86`): `168h` (`6/12=50,00%` grade 5 min vs `6/6=100,00%` 10 min); `30d` (`4.298/4.320=99,49%`); escore OOT `0,9850` | `cap11.tex`, `cap13.tex` (Tabela `\ref{tab:reconciliacao-uc-685xxx20}`), `fig_caso_685xxx20_reconciliacao_cobertura.pdf` | `20261125_134559_972_oot_multiview` | `06/10–04/11` e `19/11/2026` | UC individual | `fig_caso_685xxx20_reconciliacao_cobertura.meta.json` |
| Painéis controlados 5.1 ($N=1.000$, `seed=42`): IF microcluster `0,500 < 0,564`; LOF $k=35$ `2,39` e `3,15 > 1,99`; Contextualização Jaccard Top-5% `0,776` | `cap09.tex`, `fig_5_1a..c.pdf` | `controlled_benchmark_v1_seed42` | Sintético controlado | `1.000` perfis | `docs/manuscript_figures/ml_controlled_panels/*.meta.json` |

---

## 2. Resultados Históricos Preservados (Explicitamente Rotulados como Históricos)

| Referência Histórica | Local no LaTeX | Origem e Papel |
| :--- | :--- | :--- |
| Coorte histórica `66.099` UCs (`62.766` elegíveis, 16 atributos), correlação LOF `0,0202` e PCA `-0,2030` | `cap12.tex`, `cap13.tex` (`figuras/correlacao.pdf`), `cap14.tex` (`274` UCs no exercício de sensibilidade) | Diagnóstico histórico pré-saneamento que motivou as correções em `src/models/composite.py`. |
| Execução OOT pré-saneamento (`23/11/2026`): `6.333` candidatas, `4.390` funil, `210`/`132`/`284` trocas Top-2.000, $\tau=0{,}5226$, média `0,2944`, $P_{95}=0{,}9115$ | `cap08.tex`, `cap11.tex`, `cap13.tex`, `cap15.tex` | Referência explícita "anterior às correções (23/11)" para comparação antes × depois do commit `c1539b4`. |

---

## 3. Dependências Externas / Institucionais (Fora da Reprodutibilidade Pública)

- **Extração bruta Oracle/MDM/CIS:** exige credenciais e rede corporativa da distribuidora; a reprodutibilidade do repositório parte das camadas particionadas do Data Lake e dos artefatos congelados em `experiments/runs/20261125_*`.
- **Compilação do PDF a partir de `main.tex`:** tratada separadamente em ambiente com MiKTeX/TeX Live.
- **Metadados administrativos finais (`00-dados.tex` / `DADOS_PRETEXTUAIS.md`):** confirmação oficial de titulações/vínculos da banca, termo de aprovação oficial e exigência de ficha catalográfica.
