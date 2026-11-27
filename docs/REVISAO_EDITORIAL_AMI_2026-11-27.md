# Revisão manual do TCC AMI

45 pontos de edição no PDF TCC_AMI_BANCA_2026-11-27_COMPILADO.pdf (101 páginas). A primeira página é a capa. A coluna “Impressa” usa o número que aparece na folha. A referência ao LaTeX corresponde à pasta docs/TCC_AMI_FINAL_BANCA_eekBR_FONTES. As mudanças posteriores nas figuras podem deslocar a paginação de uma próxima compilação.

## Por onde começar

A página da captura é a 20 do leitor, 19 impressa. O problema ali é a hierarquia: o parágrafo, a pergunta em bloco e os dois itens disputam atenção em negrito. Uma pergunta de pesquisa explícita é adequada ao TCC. O excesso aparece quando a introdução da pergunta, as auxiliares e o objetivo seguinte repetem o mesmo argumento. Esses sinais orientam a edição e não permitem atribuir autoria.

**Ordem para seis horas.** Primeira hora: resumo PT/EN, carteira e denominadores (PDF 3, 4, 83 e 94), identificação na p. 74 e remissão na p. 95. Nas duas horas seguintes, revise a p. 20 e as aberturas dos capítulos. Reserve duas horas para resultados e conclusões. Na última hora, leia em voz alta os parágrafos modificados e confira as páginas depois da compilação.

## Sugestão para a seção 2.1 — PDF 20, impressa 19

Substituição candidata para o trecho de “O problema científico” até a segunda pergunta auxiliar. Conserve o objetivo geral na seção 2.2 e use fonte normal no bloco.

> Uma queda no consumo registrado admite várias explicações. Na mesma UC, a série de energia pode indicar uma mudança de rotina, enquanto as lacunas de telemetria dificultam determinar quando a queda começou. A análise precisa localizar o desvio e identificar quais medições sustentam cada hipótese. Como a equipe só pode inspecionar parte das unidades sinalizadas, a seleção também precisa registrar por que cada caso entrou na fila.

> A questão de pesquisa é como localizar e caracterizar, nos dados AMI de baixa frequência, padrões compatíveis com perdas não técnicas. Para investigá-la, o trabalho combina o histórico de consumo, as grandezas elétricas, a disponibilidade de comunicação e o contexto da unidade e de seu alimentador.

## Candidatos em ordem de leitura

“Urgente” indica mudança de sentido, denominador, identificação ou referência. “Alta” indica passagem de grande visibilidade ou interpretação mais forte que a evidência. Cada âncora é um fragmento pesquisável.

| PDF | Impressa | Prioridade | Fonte | Procure por | Edição candidata |
|---:|---:|---|---|---|---|
| 3 | — | Alta | `00-dados.tex:75` | “O pipeline reconstruiu” | Uma frase abriga o vínculo UC–NIO, seis detectores e quase toda a arquitetura. Deixe uma frase para a reconstrução do histórico e outra para o cálculo dos escores. Os nomes dos módulos cabem no método. |
| 3 | — | Urgente | `00-dados.tex:75` | “mostrou robustez” | Atribui 88%–91,5% ao Top-2.000. Na p. 83 do PDF, o Top-2.000 reteve 89,8% na v1 e 88,0% na v2. Os 91,5% pertencem à carteira completa v2 (3.933/4.298). Escolha uma carteira ou identifique ambas. |
| 4 | — | Urgente | `00-dados.tex:80` | “remained robust” | Espelhe a correção do resumo. Sugestão: “Between 19 and 25 November, 88.0% of the v2 top 2,000 remained in that group. Across the full v2 candidate list, retention reached 91.5%.” |
| 17 | 16 | Alta | `cap01.tex:9` | “Um algoritmo útil” | O fecho prescreve o que qualquer algoritmo deveria fazer e repete o objetivo. Escreva o que o analista recebe neste estudo: UC, período e sinais que justificaram a seleção. |
| 18 | 17 | Alta | `cap01.tex:11` | “Essa formulação permanece relevante” | “Será preservada” soa como proposta anterior à execução. Diga que a ordenação global serviu de referência e qual dificuldade da telemetria motivou o segundo modelo. |
| 19 | 18 | Média | `cap01.tex:21` | “A pesquisa reúne três contribuições” | O parágrafo anuncia novamente o escopo já explicado acima. Preserve a fronteira entre avaliação retrospectiva e aplicação futura em uma frase. |
| 20 | 19 | Alta | `cap02.tex:7` | “O problema científico” | A oposição abstrata prepara três perguntas em negrito. Abra com o problema observável e retire o destaque das frases inteiras. Veja a sugestão abaixo. |
| 20 | 19 | Alta | `cap02.tex:15` | “Dessa pergunta derivam” | As duas perguntas auxiliares repetem a cadência da pergunta principal e antecedem os objetivos específicos. Incorpore suas duas restrições à prosa: hipóteses concorrentes e capacidade de inspeção. |
| 21 | 20 | Média | `cap02.tex:52` | “O oitavo” | Referência por número obriga o leitor a voltar oito itens. Nomeie o piloto e concentre a ressalva sobre OS/TOI em uma única frase junto da Tabela 1. |
| 23 | 22 | Alta | `cap03.tex:5` | “Há quatro decisões científicas” | A abertura promete um roteiro antes de apresentar o problema. Comece pelo primeiro dilema concreto: uma queda observada tem interpretação diferente quando faltam leituras. Conserve as citações nas afirmações que elas sustentam. |
| 24 | 23 | Alta | `cap03.tex:15` | “evidências independentes” | “Independentes” tem sentido estatístico. A própria p. 94 informa correlação de +0,9626 entre IF e ECOD. Se o sentido aqui é origem distinta, nomeie as fontes e evite sugerir independência entre escores. |
| 25 | 24 | Média | `cap03.tex:31` | “A contribuição da literatura” | “Portanto, principalmente estrutural” ocupa o lugar da decisão. Diga quais operações foram aproveitadas e qual escala temporal a AMI suporta. |
| 26 | 25 | Média | `cap03.tex:55` | “responde a uma pergunta diferente” | A explicação ganha precisão começando pela diferença temporal: um novo patamar pode persistir mesmo quando nenhuma leitura isolada é rara. Depois apresente o CPD. |
| 27 | 26 | Média | `cap03.tex:75` | “Nem todo fenômeno deve ser aprendido” | Máxima genérica na abertura de 3.9. Use uma regra física já documentada e explique por que ela tem tratamento determinístico. |
| 29 | 28 | Média | `cap04.tex:19` | “Ela responde à pergunta” | Depois do roteiro da abordagem original, a pergunta reexplica o baseline. Substitua pelo objeto da comparação: todas as UCs no mesmo espaço de atributos. |
| 30 | 29 | Alta | `cap04.tex:47` | “A principal mudança é epistemológica” | A abstração encobre uma decisão clara. “A versão refinada separa a qualidade da comunicação da raridade elétrica antes de ordenar as UCs.” Mantenha em seguida apenas os detalhes novos. |
| 32 | 31 | Alta | `cap05.tex:5` | “A hipótese simplificadora” | O PDF repete “Hipótese simplificadora” entre a frase e o destaque. Dê a premissa uma vez e siga diretamente para a troca de medidor e o intervalo de vigência. |
| 34 | 33 | Média | `cap05.tex:107` | “O propósito de” | O contraste sobre acelerar o processamento antecipa benefícios já enumerados. Comece pelo efeito da gravação: a partição diária concluída pode ser reutilizada após uma interrupção. |
| 37 | 36 | Média | `cap06.tex:5` | “A engenharia de dados é um componente científico” | Frase de legitimação no início de um capítulo técnico. Abra com as unidades efetivamente processadas e explique onde uma junção incorreta altera o histórico. |
| 39 | 38 | Média | `cap06.tex:79` | “É importante distinguir” | Mostre a distinção pelo teste: o arquivo pode existir e ainda conter uma partição incompleta. Preserve as verificações concretas. |
| 43 | 42 | Alta | `cap06.tex:135` | “A principal contribuição da plataforma” | O contraste com acelerar consultas repete a p. 34. Mantenha o que mudou de assunto: como rastrear uma evidência exibida até a medição original. |
| 44 | 43 | Alta | `cap06.tex:145` | “Ao contrário de” | Comparação com pipelines “genéricos” sem comparador definido. Comece pelos problemas encontrados no parque AMI. No item de corrente, mantenha fases ativas, 0,1 A, carga mínima de 1,0 A e P90. |
| 45 | 44 | Média | `cap06.tex:156` | “Engenharia de produção para infraestrutura corporativa” | O título amplo e os qualificativos cobrem duas decisões independentes. Separe limite de cinco sessões Oracle de verificação dos arquivos por SHA-256. |
| 46 | 45 | Média | `cap07.tex:2` | “A caracterização da população fundamenta” | Introdução genérica imediatamente antes de uma distinção decisiva. Abra pelos recortes cadastral, exploratório e OOT. Conserve o denominador junto a cada contagem. |
| 50 | 49 | Média | `cap08.tex:9` | “Essa separação é essencial” | Troque o julgamento por sua consequência: uma leitura ausente informa disponibilidade de comunicação e restringe a interpretação elétrica. |
| 51 | 50 | Alta | `cap08.tex:36` | “O objetivo não é somar arbitrariamente” | Contraste com uma alternativa que o leitor não propôs. Diga qual saída o componente realmente entrega e conecte essa saída ao fluxo mostrado na página. |
| 52 | 51 | Média | `cap08.tex:71` | “O sinal de interesse não é apenas” | A comparação verbal repete a equação imediatamente seguinte. Introduza diretamente a diferença entre a janela recente e a janela de referência. |
| 55 | 54 | Média | `cap08.tex:137` | “O agrupador comportamental” | Três frases delimitam o que o escore comportamental poderia significar. Preserve uma positiva: PCA e K-Means atribuem rótulos, cuja sequência descreve mudanças de rotina. Mantenha explícita a falta de calibração probabilística. |
| 58 | 57 | Alta | `cap08.tex:215` | “Prática convencional comum” | Na Tabela 7, a coluna parece atribuir práticas a toda a área. Renomeie para “Falha considerada” se são cenários de implementação. Descreva o zero observado no teste de perturbação como resultado daquele ensaio. |
| 62 | 61 | Média | `cap09.tex:13` | “A confirmação de uma irregularidade” | A interpretação da Figura 8 já funciona pelo desacordo IF–LOF. Concentre a ressalva sobre vistoria no fim da seção e evite repeti-la junto da chamada ao apêndice. |
| 63 | 62 | Alta | `cap12.tex:3` | “A aplicação constitui parte” | O capítulo começa defendendo a importância da interface. Comece com uma tarefa: selecionar o período e o alimentador, iniciar a extração e acompanhar os lotes. |
| 66 | 65 | Média | `cap12.tex:62` | “O primeiro é científico” | A dicotomia “científico/operacional” posterga a pergunta sobre o alimentador. Apresente a hipótese de maior homogeneidade e diga qual comparação a avaliaria. |
| 67 | 66 | Média | `cap12.tex:84` | “A distinção entre as interfaces” | A enumeração de verbos repete a descrição anterior. Dê uma ação por interface e preserve os tempos verbais que distinguem recurso disponível de recurso projetado. |
| 69 | 68 | Média | `cap12.tex:150` | “respondem a perguntas diferentes” | Identifique a escala logo no início do parágrafo: a tela individual descreve uma UC, a carteira agrega casos segundo a política do corte. |
| 74 | 73 | Urgente | `cap13.tex:75` | “A discreta diferença” | Tabela 11: a explicação intercala precisão numérica, classe diagnóstica e flag final, além de imprimir uma UC sem máscara. Mascarar como nos estudos de caso. Distinguir o predicado numérico (6.038) da fila operacional (6.037). Reservar causalidade exclusiva ao arredondamento e igualdade dos conjuntos para quando houver verificação por UC. |
| 76 | 75 | Alta | `cap13.tex:115` | “um achado de transparência essencial” | A frase avalia a qualidade da própria auditoria. Abra pelo resultado: na v1 de 25/11, o bônus foi acionado em 30.694 UCs. Em seguida explique qual condição o acionava. |
| 79 | 78 | Média | `cap13.tex:147` | “os resultados evidenciaram tanto os ganhos” | O anúncio não acrescenta nada aos itens. Comece pela falha dos blocos de 15 dias e diga por que a cobertura média dos dias presentes a ocultava. |
| 83 | 82 | Alta | `cap13.tex:214` | “Isso evidencia que o ranking captura” | Seis dias medem permanência entre dois cortes. “Estruturais e duradouros” extrapola o período observado. Termine na retenção medida e identifique o denominador. |
| 88 | 87 | Média | `cap13.tex:268` | “permanecendo corretamente retida” | “Corretamente” sugere confirmação independente. Descreva o resultado verificável: a UC permaneceu em quarentena pelo critério de comunicação. Preserve a distinção entre telemetria ausente e alarmes recebidos. |
| 92 | 91 | Média | `cap14.tex:55` | “Em um exercício de sensibilidade separado” | O travessão inicial contém os dois denominadores antes das hipóteses de custo. Abra com as 274 candidatas do consolidado exploratório. Depois apresente custo, realização e probabilidade assumida. Deixe o cenário de capacidade mensal em outro parágrafo. |
| 93 | 92 | Alta | `cap15.tex:7` | “duas lições de engenharia documentadas com transparência” | Autoavaliação antes de achados técnicos. Dê uma frase ao acionamento do bônus, outra à leitura dos alarmes e outra ao silêncio de 15 dias. Vincule cada correção ao teste correspondente. |
| 94 | 93 | Urgente | `cap15.tex:15` | “Complementaridade comprovada” | O título anuncia prova ampla e o item reúne coorte real e perturbações. Separe a exclusão de quarentenas na Base A do F1 de 68,25% para 76,00% na Base B. No item seguinte, corrija também os denominadores de retenção do resumo. |
| 95 | 94 | Urgente | `cap15.tex:20` | “Como continuidade natural” | A frase final concentra piloto, treinamento e avaliação econômica. O piloto está no capítulo impresso 12 (PDF pp. 89–92), embora o texto diga “Capítulo 14”. Use referência LaTeX por rótulo e descreva rendimento e valor econômico como resultados a estimar. |
| 99 | 98 | Baixa | `ap02.tex:2` | “segundo a documentação da demonstração” | O apêndice já identifica 1.000 observações e semente 42. Se os procedimentos foram conferidos no gerador, descreva a configuração diretamente. Caso contrário, identifique o artefato consultado. |
| 101 | 100 | Baixa | `ap02.tex:22` | “reportado no painel” | O parágrafo já informa sete promoções, seis atenuações e Jaccard 0,776. Elimine a locução e mantenha a interpretação dos empates. |

## O que preservar

• PDF 48, impressa 47, comentário da Figura 4. As duas UCs têm cobertura igual (99,44%) e medianas diárias próximas, mas perfis horários distintos. A comparação mostra o que a escala diária esconde. É um bom modelo para reescrever as conclusões.

• PDF 85–87, impressas 84–86, estudos de caso. Preservar a sequência entre observação física, histórico cadastral e limite de interpretação. Na UC 291***96, o registrador avançou após a retirada cadastrada. A identidade do ponto de injeção ainda exige apuração.

• PDF 98, impressa 97, Apêndice A. Manter identificadores, datas e custódia. Listas técnicas, símbolos, bibliografia e legendas factuais não precisam de variação estilística artificial.

## Conferência bibliográfica interna

No LaTeX dos capítulos, resumos e apêndices examinados, foram encontradas 26 chaves distintas em comandos de citação. Todas têm entrada no arquivo referencias.bib. Essa conferência verifica a ligação entre texto e bibliografia. O conteúdo dos artigos originais e os resultados computacionais não foram revalidados nesta revisão editorial.
