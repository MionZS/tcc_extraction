# Relatório completo das mudanças no TCC

**Autor:** Matheus Mion Chollet  
**Base:** TCC original de 107 páginas, 19 capítulos, e evidências em `docs/tcc_evidence/` e `docs/figures/`.  
**Produto:** manuscrito consolidado de 37 páginas, sete capítulos, 14 figuras, nove tabelas e 14 equações.  
**Data da consolidação:** 24 de novembro de 2026.

## 1. O que mudou na organização

| Capítulos originais | Destino consolidado | Mudança |
|---|---|---|
| 1–2 | 1. Introdução, problema e objetivos | Unificação da motivação, pergunta, hipótese, objetivo e delimitação. |
| 3–4 | 2. Fundamentação e delineamento metodológico | Explicação integrada de medição, detectores e avaliação temporal. |
| 5–6 | 3. Arquitetura de dados e engenharia do processamento | Vínculo UC–NIO, vigência, aquisição, normalização, cobertura e matriz analítica reunidos em sequência. |
| 8, 11, 13 e parte do 16 | 4. Arquitetura analítica multivisão e plataforma | Especialistas, consenso elétrico, gate, contexto por alimentador e interfaces apresentados em um só fluxo. |
| 7, 9, 10, 12 e 17 | 5. Demonstração controlada, EDA e resultados | Resultados sintéticos e reais organizados por produto experimental e unidade de comparação. |
| 14–15 e 16.2.4 | 6. Projeção operacional e governança | Reserva de capacidade, coortes O/A/N, backlog, Yield@k e avaliação econômica consolidados sem triplicar conceitos. |
| 18–19 | 7. Limitações e conclusões | Alcance dos ensaios e conclusões vinculados aos resultados verificados. |

A numeração de figuras, tabelas e equações foi refeita. Todas as nove tabelas aparecem na lista de tabelas. Os principais diagramas textuais foram substituídos por figuras vetoriais de arquitetura, roteamento e ciclo do caso. A versão consolidada mantém o título original em inglês.

## 2. Mudanças de voz e alcance

O texto foi reescrito em voz científica impessoal. Foram eliminadas formulações que descreviam inspeção externa do código ou de um pacote de terceiros, incluindo referências narrativas a arquivos internos de implementação. Os caminhos permanecem apenas no parecer separado, onde servem como instruções executáveis ao autor.

A delimitação do estudo aparece na introdução, no método, na abertura dos resultados e nas limitações/conclusões. Categoria diagnóstica passou a significar hipótese de investigação. “Candidato” corresponde à política que produziu cada lista. O texto não atribui confirmação de fraude, inspeção realizada ou retorno econômico observado às saídas computacionais. Município e concessionária foram omitidos do manuscrito consolidado.

**Decisão editorial:** foram mantidas duas notas explícitas de consolidação junto às figuras que ainda exigem substituição ou reconciliação. Elas devem sair antes do PDF destinado à banca.

## 3. Equações, notação e conceitos

- **MAD:** a mediana das distâncias absolutas passou a ser calculada em relação à mediana da própria série, com conjunto de dias válidos definido.
- **Vigência UC–NIO:** a condição de instalação e retirada passou a usar conjunção lógica e alternativa explícita para retirada em aberto. Troca no mesmo dia permanece ambígua sem horário.
- **Cobertura:** número de observações válidas dividido pela grade esperada para canal, janela e cadência. Leituras com energia zero continuam válidas.
- **Ensemble:** a inferência OOT usa seis detectores, com pesos IF/LOF/ECOD/COPOD/HBOS/PCA de 0,22/0,20/0,18/0,18/0,10/0,12 e bônus de regra de 0,15. A variante histórica de quatro detectores usa 0,30/0,25/0,25/0,20 e bônus de 0,20. As variantes não foram usadas para justificar a mesma contagem.
- **Gate:** foram separados o indicador de decisão elétrica `D_i`, a suficiência de cobertura `G_i` e sua conjunção para declaração elétrica confiável.
- **Dias atípicos:** numerador e denominador usam dias válidos na mesma janela. Sem dia válido, a taxa fica indisponível.
- **Backlog:** adotou-se um único mês `m` para estoque inicial, entradas únicas, conclusões efetivas, baixas administrativas e capacidade. Saída de uma lista diária não equivale automaticamente a caso concluído.
- **Capacidade:** `K_m` é a capacidade mensal, enquanto `k` é a profundidade efetivamente inspecionada em `Yield@k`.
- **Antecedência:** `Δt_i = t_base,i − t_modelo,i`. Valor positivo representa decisão mais cedo pelo modelo.
- **Reserva:** a parcela de até 40% foi apresentada uma vez como hipótese ilustrativa. Em 2.000 vagas mensais, o teto calculado é 800 vagas.
- **Colunas técnicas:** nomes `x__*` foram tratados como código monoespaçado. Caracteres estranhos e remissões genéricas foram retirados.

## 4. Engenharia de dados e EDA incorporadas

A análise histórica identificou **69.895 UCs**, **72.707 registros de instalação**, **72.661 NIOs distintos**, **1.288 UCs com múltiplos medidores** e **46 NIOs reutilizados**. Esses números fundamentam a reconstrução de vigência. O parque ligado, com **67.479 medidores em 65.762 UCs**, foi apresentado em tabela própria, separada da coorte OOT.

A EDA foi redigida no passado. Incorporaram-se distribuição de cobertura, UCs por alimentador, comparação de curvas, dispersão robusta, formas horárias e elegibilidade. No consolidado de **66.099 UCs**, **62.766** passaram pelo gate de 70% e **3.333** ficaram abaixo. A mediana de consumo diário foi **4,936 kWh/dia** e o percentil 99, **55,060 kWh/dia**. O par de UCs com cerca de 20 kWh/dia demonstrou MAD de **0,051** versus **16,564 kWh/dia**.

A camada diária tinha **102 partições** entre 2 de agosto e 12 de novembro, intervalo de **103 dias civis**. O dia **1º de outubro** não estava nessa camada. A distribuição histórica de sequências incluiu valores maiores que a janela nominal de 30 dias. O texto deixou de atribuir essas durações a estudos de caso específicos.

## 5. Resultados reconciliados

| Produto | Coorte e política | Resultado incorporado |
|---|---|---|
| OOT multivisão, 19/11 | 65.686 UCs, referência de treino congelada | 62.857 elegíveis, 2.829 em quarentena, 6.333 candidatos; baseline IF selecionou 3.122 sob outro critério. |
| DAILY_RULES, 1º–19/11 | 19 cortes e 1.248.501 janelas UC × dia | Média de 5.387,7 candidatos/dia; 4.978 em 19/11. |
| Sensibilidade em 70% | Coorte OOT comum com política do ensaio | 62.857 elegíveis, 6.311 retidos e 915 interceptados em quarentena. |
| Funil sequencial OOT | Contexto topológico exigido em M3 | M0–M5: 65.686 → 65.686 → 62.857 → 58.857 → 4.390 → 4.390. |
| Ablação do ensemble | Top-2.000 em orçamento fixo | Sem LOF, 210 UCs substituídas; sem PCA, 132; núcleo de quatro, 284. |

A nova figura da série diária foi calculada dos 19 cortes calibrados. Em 19/11, **4.342 permanências + 636 entradas = 4.978**. A lista do dia anterior tinha **4.342 permanências + 588 saídas = 4.930**. A figura antiga, na faixa aproximada de 30 mil, foi retirada dessa demonstração.

A diferença entre **6.333** do OOT e **6.311** do ensaio de cobertura corresponde a listas parcialmente diferentes. A interseção conhecida contém **5.917 UCs**, com **416 exclusivas do OOT** e **394 exclusivas da sensibilidade**. O funil com alimentador exclui **1.943** das candidatas presentes na lista geral por ausência de vínculo topológico.

A correlação entre detectores e a estabilidade dos escores foram integradas. Na análise histórica, IF–ECOD apresentou correlação aproximada de **0,95**, LOF–consenso **0,02** e PCA–consenso **−0,20**. No ensaio de remoção de 1% dos extremos, a normalização dependente do lote produziu deslocamento máximo de **0,1865**, enquanto quantis congelados produziram **0** nessa perturbação.

## 6. Estudos de caso e figuras

**UC 908***00.** A atualização cadastral de 31/10 foi interpretada com continuidade do mesmo NIO. A FA mudou de **18,05 para 5,72 kWh/dia**, com **12 dias** de RA entre **14 e 31 kWh/dia** e persistência de **20 dias**. O escore elétrico OOT foi **0,9601**. Inseriu-se figura quantitativa de patamares e faixa de RA. Falta compilar a série temporal final com FA, RA e lacunas.

**UC 291***96.** A cobertura aparente de 50% é compatível com seis leituras de dez minutos por hora confrontadas com uma grade de doze posições de cinco minutos. O agregado diário registra cadência indeterminada. A estabilidade do consumo e a integridade do acumulador foram preservadas na interpretação. Falta demonstrar a cadência com timestamps brutos e deltas no mesmo recorte.

**UC 685***20.** O painel de 168 horas apresenta cerca de **50% de cobertura**, enquanto a janela do corte de 04/11 registra **99,49%**. O manuscrito apresenta ambos e exige reconciliar canal, cadência e denominador ou substituir o exemplo. Uma pontuação alta isolada não demonstra passagem pelo gate.

**Demonstração sintética.** Foram recuperados painéis de K-Means, PCA, resíduo e comparação semanal. O texto foi ajustado aos valores existentes. Dois membros do microagrupamento tinham escores IF próximos de **0,995 e 0,998**, contrariando a explicação antiga de ponto cego. A escala do LOF exige ampliação e explicitação da transformação. A posição **999º global / 250º local** da UC sintética nº 384 exige percentis, tamanho de coorte e orientação do ranking antes de sustentar uma promoção contextual.

## 7. Formatação e referências

O PDF consolidado contém folha de rosto, resumos em português e inglês, listas de figuras e tabelas, sumário e 15 referências herdadas do original. Foram eliminados os textos “Figura figura”, “Tabela tabela”, “Figura sensibilidade do gating”, `NI0`, o caractere `น` e as anotações “Citado ... na página”. Foram conferidas as referências cruzadas no PDF gerado. O fonte LaTeX foi incluído no pacote editável e também incorporado como anexo ao PDF.

**Pendências bibliográficas:** verificar manualmente os dois DOIs de 2026, conferir edição do PRODIST e ampliar a revisão de AMI, cadência, viés de inspeção, validação temporal e decisão sob custo. O relatório de banca independente especifica os critérios de aceite.

## 8. Estado de entrega

Esta versão serve como **consolidação para fechamento editorial**. Antes da submissão à banca, é necessário substituir as duas notas de consolidação, finalizar ou retirar os gráficos pendentes, preencher o orientador, revisar títulos internos e anonimização de todas as imagens reaproveitadas, conferir a bibliografia e compilar o LaTeX no ambiente institucional. A compilação do fonte no ambiente LaTeX de submissão ainda não foi verificada. O PDF consolidado foi paginado e inspecionado visualmente.

O parecer `REVISAO_BANCA_E_PENDENCIAS.md` contém checklist priorizado, avaliação crítica e dez perguntas de arguição. A nota **7,5/10** e o potencial de **9,0/10** apresentados nele são **simulações de banca**, não avaliação oficial.

