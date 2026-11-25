# Mudanças da nova revisão do TCC AMI

**Comparação:** TCC_AMI_Consolidado.pdf (37 páginas) → TCC_AMI_Consolidado_Revisado.pdf, com base no parecer “Avaliação Crítica de TCC - Google Gemini” e no arquivo TCC_AMI_Consolidado_Revisao_Alteracoes.md.  
**Escopo deste documento:** segunda rodada de revisão, posterior à consolidação dos 19 capítulos. O relatório anterior descrevia principalmente a primeira rodada e não atendia a esta comparação.

## Síntese da segunda rodada

A nova revisão manteve sete capítulos, 14 figuras, nove tabelas e 14 equações. A intervenção concentrou-se em aprofundar o fundamento de perdas não técnicas (PNT), responder à crítica sobre escolhas heurísticas, delimitar o que os ensaios de fato demonstraram, alinhar objetivo e conclusão, e ampliar referências. O arquivo de alterações lista substituições por página e as 14 expressões matemáticas em LaTeX para incorporação posterior ao fonte.

| Crítica recebida | Mudança registrada na nova revisão | Resultado e limite |
|---|---|---|
| Projeção operacional dominava a exposição empírica. | Resumo, introdução, objetivos e conclusão passaram a separar avaliação computacional de protocolo futuro. Capítulo 6 esclarece capacidade e simulação. | 6.333 candidatos em um corte são estoque instantâneo, não incidência mensal. |
| Insuficiente revisão específica de PNT. | Fundamentação acrescentou classificação das perdas, observabilidade, balanço de energia, medição a montante, topologia, GD, alternativas supervisionadas e viés de seleção. | A análise individual por curvas é diferenciada do balanço por circuito. O trabalho não mediu diretamente a energia perdida por alimentador. |
| Escolhas de modelos insuficientemente justificadas. | K-Means sobre formas normalizadas foi explicado pela comparação em horários fixos. DTW e autoencoders foram discutidos como alternativas. | Não houve comparação experimental com DTW ou autoencoders. |
| Seis detectores OOT confundidos com variante de quatro e com ablação. | A redação individualizou o comitê OOT de seis membros, a variante histórica de quatro e o núcleo de quatro do ensaio de ablação. | Pesos e bônus têm significado de configuração de projeto. Não há evidência de superioridade pericial de um conjunto. |
| Constantes no especialista de eventos e roteamento local apresentadas sem domínio. | Foram definidos contagem de alarmes, inclinação, marcador de troca, clip, tamanho da coorte e renormalização dos pesos disponíveis. | Pesos e cortes 150/1.000 são heurísticos. Falta avaliação de sensibilidade própria para legitimá-los. |
| Demonstração do gate poderia parecer validação de PNT. | A nova redação explica que Beta(8,2) e Beta(7,3) impõem artificialmente associação entre qualidade e anomalia nos 1.000 perfis. | A retenção de 2/8 e Top-50 de 1/8 a 70% mostra efeito do desenho sintético. Não mede desempenho para PNT reais. |
| Figura antiga da lista diária tinha salto para cerca de 30 mil. | A revisão conserva 19 cortes calibrados, distingue a regra legada sem materialidade e relaciona entradas e saídas à dinâmica da lista. | A causa individual de cada transição ainda não foi reconstruída. O valor calibrado final é 4.978 e a média é 5.387,7. |
| Interpretação superficial do benchmark de engenharia. | O texto quantifica redução de volume persistido de cerca de 85,9% e redução do tempo local de cerca de 17,1%. | Esse tempo mede leitura/escrita local do consolidado. Não representa duração de extração corporativa. |
| Simulação econômica sem referência de equilíbrio. | Acrescentou-se o ponto de equilíbrio para custo R$ 180, valor potencial R$ 1.800 e realização 65%. | p = 180/(0,65 × 1.800) ≈ 15,38%. Os parâmetros são hipóteses; não demonstram recuperação observada. |
| Fórmulas, termos e referências incompletos. | Equações foram recompostas no arquivo de alterações. Foram incluídas fontes de PNT e regulação. As notas editoriais visíveis foram retiradas do PDF revisado. | O PDF foi alterado diretamente. Incorporar as mesmas mudanças ao LaTeX mestre e recompilar ainda é uma pendência. |

## Mudanças por parte do texto

### Resumo e abstract

Os dois textos passaram a explicitar detecção, **localização** e caracterização retrospectiva de anomalias com ênfase em padrões compatíveis com PNT. Os resultados quantitativos continuam vinculados aos seus produtos: 65.686 UCs avaliadas, 62.857 elegíveis e 6.333 candidatas no OOT, 4.390 no funil com alimentador, 210 substituições ao remover LOF e média diária calibrada de 5.387,7. O fechamento passou a tratar confirmação e rendimento de inspeções como avaliação posterior.

### Introdução e objetivos

A definição de localização foi fixada como associação do desvio à UC, ao período e, quando o cadastro permitir, ao alimentador. A introdução passou a diferenciar alterações individuais, falhas de comunicação e eventos coletivos. O objetivo geral voltou a mencionar explicitamente PNT e localização. A apresentação dos capítulos foi encadeada em duas entregas examinadas computacionalmente, engenharia de dados e avaliação da seleção, mais um fluxo prospectivo de uso e verificação.

### Fundamentação

A nova seção de PNT distingue perda técnica, perda não técnica e furto. Explica por que um balanço energético de alimentador exige medição a montante, sincronização, estimativa das perdas técnicas, vínculo topológico e consideração de GD. Uma diferença de balanço não localiza automaticamente uma UC. O texto passa então à investigação por curvas individuais. A revisão introduziu Glauner et al. (2017), Villar-Rodriguez et al. (2017), página de perdas da ANEEL e REN 1.000/2021, segundo o registro das alterações.

A justificativa de K-Means foi ampliada. A distância a centroides mantém comparação por horário nas curvas normalizadas. DTW admite deslocamentos temporais e autoencoders representações mais flexíveis. Sua menção oferece contexto metodológico, sem atribuir-lhes resultados de comparação que não foram obtidos.

### Pipeline de dados e plataforma

A sincronização incremental agora é descrita pela verificação de partições, recuperação de lacunas e avanço ao último dia completo. Presença física de arquivo e integridade de canal, linha e equipamento são controles distintos. O benchmark local foi interpretado quantitativamente, preservando a separação entre materialização e consulta às fontes corporativas. A TUI foi caracterizada como interface para extrair, auditar, treinar, registrar e comparar versões, sem transformar a proposta de despacho em função já verificada.

### Arquitetura multivisão

A descrição do especialista de eventos define a contagem de alarmes a ≥ 0, inclinação b, marcador binário de troca m e função de saturação c no intervalo [0,1]. O texto explicita a natureza heurística dos coeficientes. A normalização por quantis P01/P99 congelados foi explicada como escala relativa do treino, sem equivalência a probabilidade de irregularidade e com tratamento necessário quando o denominador é nulo.

O comitê elétrico OOT utiliza IF, LOF, ECOD, COPOD, HBOS e PCA residual, pesos 0,22/0,20/0,18/0,18/0,10/0,12, bônus de 0,15 e limiar elétrico de referência 0,5226. A variante histórica tem IF/LOF/ECOD/PCA, pesos 0,30/0,25/0,25/0,20 e bônus de 0,20. A ablação que reduz o comitê OOT a quatro componentes é outro ensaio. O agrupador comportamental produz classes e referências de curva, sem score probabilístico único.

O roteamento por alimentador foi qualificado como combinação heurística de referências, condicionada ao tamanho da coorte. Para circuitos com pelo menos 1.000 registros, pesos nominais local/regional/global 0,70/0,20/0,10 tornam-se 0,875 local e 0,125 global quando falta a referência regional. A redação registra que a disponibilidade do nível local e a avaliação do consenso elétrico são problemas distintos.

### Demonstração e resultados

O trecho sobre as figuras sintéticas foi corrigido para reconhecer escores IF de cerca de 0,995 e 0,998 no microagrupamento, escala comprimida do LOF e a posição 999ª global e 250ª local da UC sintética nº 384. A posição ordinal não prova ganho contextual sem tamanho de coorte, percentis e orientação. As figuras correspondentes continuam precisando de correção própria.

A apresentação de OOT separou a seleção de 6.333 candidatos do baseline IF com 3.122. A análise de gating manteve 6.311 candidatas a 70% sob outra política. A diferença líquida de 22 contém 5.917 comuns, 416 exclusivas do OOT e 394 exclusivas da sensibilidade. O funil topológico reteve 4.390 e a falta de alimentador foi associada a 1.943 candidatas da lista geral. A nova redação reforça que correlação, Jaccard e churn descrevem composição, enquanto precisão e rendimento dependeriam de desfechos de campo.

A revisão registra explicitamente a necessidade de reconciliar os 50% de cobertura horária e os 99,49% da janela na UC 685***20. Não atribui causa à divergência sem observar o canal, a cadência e os registros intervalares.

### Projeção operacional, economia e conclusão

Para K = 2.000 inspeções mensais, os quocientes pontuais 6.333/2.000 ≈ 3,17 e 4.390/2.000 ≈ 2,20 foram identificados como comparação de **estoque no corte** com capacidade ilustrativa. Não representam meses para esgotar uma demanda recorrente. A demanda mensal requer deduplicação por episódio.

Na simulação econômica de 274 inspeções, custo de R$ 180, potencial de R$ 1.800 e realização de 65%, o saldo estimado é −R$ 1.233 para probabilidade hipotética de 15% e +R$ 30.825 para 25%. O ponto de equilíbrio calculado é 15,38%. A comparação não usa recuperação real de energia ou receita.

A conclusão agora responde à hipótese através de diferenças verificadas de listas. O filtro topológico e a retirada dos detectores mudaram quem é selecionado. O ensaio de normalização congelada demonstrou estabilidade diante da remoção de extremos no lote. O texto reserva eficácia pericial, valor econômico realizado e descoberta incremental à avaliação prospectiva com desfechos independentes.

## Equações e referências

O arquivo de alterações contém texto substitutivo por página e LaTeX das 14 equações, incluindo MAD, vigência, cobertura, evento, normalização, consenso, gate, dias atípicos, erro PCA, backlog, reserva, Yield/Lift, antecedência e valor econômico. Essas expressões devem ser transpostas ao fonte mestre correspondente ao PDF revisado e conferidas no PDF recompilado.

A bibliografia passou de 15 para 19 entradas, segundo as 15 referências da primeira consolidação e as quatro novas entradas listadas na revisão: duas fontes ANEEL, Glauner et al. e Villar-Rodriguez et al. O registro de revisão informa conferência de metadados de Han e Pavlidis (2026), sem leitura integral desses artigos. A presença de uma fonte na lista não equivale à validação de todas as afirmações associadas.

## Estado dos arquivos e pendências

| Arquivo novo | Função |
|---|---|
| `Avaliação Crítica de TCC - Google Gemini (1).pdf` | Parecer externo sobre a versão de 107 páginas, usado como pauta crítica. Sua recomendação de explicar o salto de cerca de 30 mil se refere a figura legada substituída. |
| `TCC_AMI_Consolidado_Revisado.pdf` | PDF canônico de 37 páginas depois da segunda rodada textual. Conserva 14 figuras, nove tabelas e 14 equações. |
| `TCC_AMI_Consolidado_Revisao_Alteracoes.md` | Registro detalhado das substituições por página, referências acrescentadas, fórmulas em LaTeX e seis pendências explicitadas. |
| `REVISAO_BANCA_E_PENDENCIAS(1).md` | Checklist e arguição da rodada anterior. Deve ser atualizado para distinguir o que a segunda revisão resolveu do que permanece aberto. |

Permanecem pendentes: nome oficial do orientador; fonte LaTeX sincronizado exatamente com o PDF revisado e recompilação institucional; correção dos três painéis metodológicos; séries finais dos estudos de caso; reconciliação da cobertura da UC 685***20; manifesto da suíte, artefatos dos modelos e tabelas de fatos compatíveis com as contagens. A revisão textual não executou novamente o pipeline e não acrescentou nova série temporal de caso. Sensibilidade de pesos, inspeções de campo e métricas reais de rendimento continuam pesquisa posterior, conforme o escopo retrospectivo declarado.

