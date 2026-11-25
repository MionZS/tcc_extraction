# Integração científica e editorial — 24/11/2026

## Documentos de referência

- Estrutura e profundidade: `TCC_AMI_REVISAO_INTEGRADA_2026-11-23_eekBR_UFPR.pdf` (107 páginas) e seu projeto LaTeX.
- Fonte científica prioritária: `TCC_AMI_Consolidado_Revisado.pdf` (37 páginas), fonte incorporado ao PDF consolidado e `TCC_AMI_Consolidado_Revisao_Alteracoes.md`.
- Atualizações: `RELATORIO_COMPLETO_DAS_MUDANCAS.md` e `RELATORIO_MUDANCAS_NOVA_REVISAO.md`.
- Critérios de revisão: `Avaliação Crítica de TCC - Google Gemini (1).pdf` e `REVISAO_BANCA_E_PENDENCIAS.md`.

## Atendimento ao feedback

| Questão | Intervenção | Estado |
|---|---|---|
| Fundamentação fragmentada e PNT | Articulação do balanço energético, observabilidade, métodos de detecção e contexto elétrico; inclusão de referências específicas. | Revisado |
| Justificativa das técnicas | Explicação das escolhas e limites de K-Means, PCA, detectores e contextualização; sem atribuir superioridade não medida. | Revisado; comparação adicional não executada |
| Pipeline pouco demonstrado | Preservação da descrição de engenharia, contratos UC–NIO, Oracle, lotes, Polars, persistência, retomada, calendário e testes; benchmark local incorporado. | Revisado |
| Divergência entre ensembles | Separação da configuração histórica de quatro detectores, OOT de seis e ablação. | Revisado documentalmente |
| Fusão e heurísticas | Formalização do consenso elétrico, regras multivisão, gate e roteamento hierárquico; parâmetros e limites distinguidos. | Sensibilidade individual ainda pendente |
| Avaliação temporal | Separação do ajuste anterior e da inferência de novembro; quantis congelados; resultados de estabilidade. | Evidência documental; execução não repetida |
| Resultados divergentes | Série calibrada de 19 cortes e OOT separados; 5.387,7 candidatos/dia, 4.978 por regras e 6.333 por OOT em 19/11. | Reconciliado conforme fontes atualizadas |
| Gating | Separação do ensaio sintético enviesado, sensibilidade retrospectiva e política de qualidade. | Interpretado; sem recall de fraude |
| Funil e ablação | Distinção entre etapas de seleção e mudanças de composição do Top-K por detector. | Revisado |
| Estudos de caso | Narrativas atualizadas, sem confirmação causal; diferença de cobertura explicitada. | Séries completas e reconciliação ainda pendentes |
| Excesso de conteúdo prospectivo | Capítulos de operação e economia reunidos; 2.000 vistorias/mês como cenário hipotético. | Revisado |
| Objetivos, resultados e conclusão | Alinhamento à execução retrospectiva e às contribuições efetivamente demonstradas; OS/TOI e avaliação em campo reservados para trabalho futuro. | Revisado |
| Figuras e tabelas | Preservação das evidências relevantes; análise textual e referências cruzadas; imagens antigas não utilizadas excluídas do ZIP. | Novas capturas e três painéis não incluídos nesta rodada |
| Equações e formatação | Reconstrução de fórmulas incompletas do fonte consolidado, padronização de notação, numeração de tabelas e referências; retirada de contagem de citações na bibliografia. | Compilado e conferido |

## Pendências externas e limites

- Orientador informado e inserido no fonte. Confirmar titulações, vínculos institucionais, termo de aprovação e ficha catalográfica; ver DADOS_PRETEXTUAIS.md.
- Incorporar posteriormente as capturas reais das interfaces, completar painéis metodológicos e séries individuais e reconciliar a cobertura da UC `685***20`.
- Anexar `experiments/runs/latest_suite.json`, tabelas de fatos e artefatos congelados de treinamento para reprodução independente. Os resultados desta edição derivam dos relatórios fornecidos, não de nova execução dos modelos.
- Confirmar autorização institucional e anonimização dos dados antes de divulgação.
- Conferência bibliográfica: registros dos DOIs recentes foram consultados; não houve verificação integral do conteúdo de todos os artigos.
- A compilação e inspeção visual não equivalem à certificação institucional de conformidade ABNT.

## Compilação

Template eekBR fornecido pelo autor, commit `f6ec8bd4e9213ce2682ba593b8b8c25bb43cca96`, classe abntex2, estilo UFPR e bibliografia biblatex-abnt/Biber. PDF A4, 78 páginas. Verificação do log final: sem erros fatais, referências ou citações indefinidas, caracteres ausentes ou caixas que ultrapassem as margens (overfull). Permanece um aviso não impeditivo de opções globais herdadas do template (`subsection`, `portugues`).
