# Pacote de Figuras Reproduzíveis e Fragmentos LaTeX para `TCC_AMI_Consolidado.pdf`

**Execução de Referência (`latest_suite.json`):** `20261125_134559_972_oot_multiview`

## Mapa de Incorporação no Manuscrito LaTeX

| Destino no Manuscrito | Arquivo LaTeX Pronto | Figuras Vinculadas | O que Substitui |
|:---|:---|:---|:---|
| **Seção 4.4 (Nova Subseção Curta)** | `manuscript_figures/latex_snippets/sec_4_4_interfaces_reais.tex` | `fig_4_4a_tui_menu_governanca`, `fig_4_4b_tui_calendario_auditoria`, `fig_4_4c_cockpit_carteira_candidatos`, `fig_4_4d_cockpit_raiox_uc_908xxx00` | Insere 4 capturas reais da TUI (Rich SVG) e Cockpit Web (Playwright) sobre dados retrospectivos anonimizados |
| **Seção 5.1 (Página 21)** | `manuscript_figures/latex_snippets/sec_5_1_demonstracoes_ml.tex` | `fig_5_1a_isolation_forest_microcluster`, `fig_5_1b_lof_escala_propria`, `fig_5_1c_contextualizacao_percentis` | Substitui integralmente a **Nota Editorial 1** pelos 3 painéis controlados finalizados com resultados calculados |
| **Seção de Estudos de Caso** | `manuscript_figures/latex_snippets/sec_estudos_caso_reconciliados.tex` | `fig_caso_908xxx00_fa_ra_completo`, `fig_caso_291xxx96_horarios_registradores`, `fig_caso_685xxx20_reconciliacao_cobertura` | Substitui a **Nota Editorial 2** com séries reais completas FA/RA, 314 leituras de registradores e reconciliação da UC `685***20` |

## Nota para o Rich SVG
Para compilação em `pdflatex` padrão (que aceita `.pdf` e `.png`), criei automaticamente versões `.png` em alta resolução (`1260px @ 2x`) e para conversão vetorial direta para `.pdf` no LaTeX os arquivos `.svg` originais de alta fidelidade permanecem junto de cada figura.
