---
title: "Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches"
author: "Matheus Mion Chollet"
date: "2026"
lang: pt-BR
---

# Introdução {#introduuxe7uxe3o}

A expansão das Infraestruturas de Medição Avançada (*Advanced Metering
Infrastructure* - AMI) altera de forma substancial a quantidade e a
natureza da informação disponível às distribuidoras de energia. Em vez
de depender apenas de leituras mensais de faturamento, passa a ser
possível observar curvas de energia em intervalos de minutos, grandezas
elétricas instantâneas, registradores, alarmes do equipamento e
informações relacionadas à disponibilidade da própria comunicação. Esse
aumento de resolução cria uma oportunidade relevante para reconhecer
mudanças de comportamento, localizar eventos anômalos e fornecer novas
evidências para investigação de perdas não técnicas.

Entretanto, maior disponibilidade de dados não implica automaticamente
maior capacidade de identificar irregularidades. A detecção de perdas
não técnicas é um problema particularmente difícil porque diferentes
fenômenos podem produzir assinaturas semelhantes. Uma queda de energia
registrada pode decorrer de redução legítima de carga, mudança de
rotina, substituição do medidor, indisponibilidade de telemetria,
geração local, defeito elétrico ou manipulação da medição. De forma
análoga, uma UC com comportamento estatisticamente raro não é
necessariamente uma UC irregular. A inspeção de campo permanece,
portanto, como mecanismo de confirmação operacional.

A literatura de detecção de perdas não técnicas tem explorado técnicas
supervisionadas e não supervisionadas, bem como métodos de ordenação de
consumidores para inspeção. Sau et al. tratam explicitamente a
identificação e o *ranking* de consumidores suspeitos como forma de
aumentar a utilidade das inspeções [@sau2023]. Trabalhos recentes também
aproximam a detecção de furto da decisão operacional, incorporando valor
econômico e restrições de inspeção [@han2026], enquanto abordagens
explicáveis buscam reduzir a distância entre o score produzido pelo
modelo e a interpretação realizada pelo especialista [@pavlidis2026].

O objetivo central deste trabalho é a **detecção, localização e
caracterização de padrões anômalos em redes de distribuição**, com
interesse especial em padrões compatíveis com perdas não técnicas e
furto de energia. A priorização de inspeções constitui aplicação
operacional da detecção e da caracterização das evidências. Um algoritmo
útil para a distribuidora deve ser capaz não apenas de atribuir um
score, mas também de mostrar por que uma determinada UC se destaca,
quais evidências sustentam essa posição e se o caso é compatível com a
capacidade de investigação disponível.

A formulação inicial do trabalho partiu de uma abordagem global de
detecção de anomalias: janelas históricas eram transformadas em features
estatísticas, e diferentes detectores não supervisionados eram
combinados para produzir um score de anomalia. Essa formulação permanece
relevante e será preservada como **baseline metodológico**. O
desenvolvimento do pipeline e a observação das particularidades dos
dados AMI, entretanto, mostraram que parte da variabilidade encontrada
pelo modelo não pertence ao mesmo domínio físico. Falhas de comunicação,
troca de medidor, sazonalidade, mudança de regime, diferenças entre
alimentadores e eventos regulatórios podem aparecer simultaneamente no
mesmo vetor de atributos, apesar de exigirem interpretações diferentes.

A arquitetura atual surge dessa observação. Em vez de substituir o
objetivo original, ela decompõe o problema em visões complementares:
qualidade e comunicação, mudança temporal, comportamento diário,
grandezas elétricas, alarmes físicos, regras determinísticas e contexto
topológico. A hipótese é que essa decomposição aumenta a
interpretabilidade e reduz a quantidade de candidatos produzidos por
causas espúrias, permitindo que o detector global seja comparado com
versões progressivamente enriquecidas da metodologia.

Outro ponto central é a natureza temporal da própria identidade do dado.
A telemetria é indexada pelo medidor físico, enquanto a unidade de
interesse comercial e operacional é a Unidade Consumidora (UC). Uma UC
pode receber diferentes medidores ao longo do tempo; um mesmo medidor
pode ser retirado, substituído ou reutilizado. Sem reconstruir essa
relação por vigência temporal, uma troca de equipamento pode produzir
uma falsa quebra de consumo e ser confundida com uma anomalia real. Esse
problema de engenharia de dados passa, portanto, a fazer parte da
metodologia de detecção e não apenas da etapa de ETL.

A escala do problema também exige uma solução de software compatível com
a realidade operacional. A aplicação desenvolvida não se restringe aos
modelos de Machine Learning. Ela inclui extração incremental e
idempotente, Data Lake particionado, processamento colunar com Polars,
escrita por *sinking*, auditoria de continuidade temporal, controle de
integridade, geração de datasets, treinamento, versionamento de modelos
e interfaces para exploração dos resultados. A TUI organiza os fluxos de
engenharia e governança, enquanto o Cockpit Web em desenvolvimento
aproxima o resultado analítico do processo de investigação por UC.

Essa evolução também modifica o paradigma operacional. Em vez de
produzir uma lista estática de suspeitos, o sistema passa a manter uma
fila dinâmica de evidências. Cada camada pode remover casos sem
confiabilidade suficiente, redirecionar ocorrências para telecomunicação
ou manutenção, reforçar suspeitas individuais ou reconhecer eventos
coletivos. O número de casos remanescentes pode então ser confrontado
com a capacidade mensal de inspeção. Dessa forma, o valor do modelo
passa a ser medido tanto pela qualidade da detecção quanto pela
capacidade de transformar a população analisada em um conjunto
acionável.

As principais contribuições do trabalho são:

1.  uma arquitetura de dados capaz de reconstruir a relação UC--NIO ao
    longo do tempo e preservar a auditabilidade da série histórica;

2.  um pipeline de extração e processamento em larga escala, com
    retomada inteligente, particionamento e persistência incremental;

3.  uma metodologia multivisão que separa comunicação, comportamento,
    grandezas elétricas, eventos físicos e contexto topológico;

4.  a combinação entre detecção de mudança de regime e detecção de
    raridade estatística;

5.  a introdução da análise de perfis diários e recorrência de dias
    atípicos como representação do comportamento intradiário;

6.  a preservação de regras determinísticas para fenômenos cuja
    interpretação é definida pela física ou por referência regulatória;

7.  um desenho experimental que compara a metodologia original com
    refinamentos sucessivos por meio de estudo de ablação;

8.  a integração dos resultados em uma aplicação de apoio à análise e
    priorização de inspeções, sem utilizar o modelo como fonte
    automática de verdade;

9.  um experimento demonstrativo reproduzível que expõe, sobre os mesmos
    dados controlados, a pergunta respondida e o efeito de cada técnica
    de aprendizado de máquina utilizada no pipeline.

A detecção de anomalias por aprendizado de máquina é examinada a partir
de uma pergunta ampliada: **quais anomalias são estatisticamente
relevantes, quais são explicáveis por qualidade ou contexto, quais
permanecem compatíveis com perdas não técnicas e como essas evidências
podem ser apresentadas de forma reproduzível ao processo de inspeção?**

# Problema de pesquisa e objetivos

## Problema de pesquisa

Dados AMI possibilitam observar alterações que não aparecem no
faturamento agregado, mas a mesma riqueza de dados também cria novas
fontes de ambiguidade. Uma anomalia pode existir no consumo, na
qualidade de comunicação, na relação entre fases, no comportamento
temporal ou no próprio cadastro do equipamento. Além disso, a população
de consumidores é heterogênea e está distribuída em alimentadores com
características distintas.

O problema científico não é, portanto, apenas separar valores normais de
valores raros. É necessário **localizar a origem da discrepância,
caracterizar sua natureza e distinguir evidência de anomalia de
evidência de causa**. O problema operacional acrescenta uma segunda
restrição: mesmo quando muitas UCs apresentam desvios, somente uma
parcela pode ser inspecionada.

A pergunta principal do trabalho é:

> **Como detectar, localizar e caracterizar padrões anômalos em dados
> AMI de baixa frequência, combinando evidências temporais, elétricas,
> comportamentais, de comunicação e contextuais, com ênfase em padrões
> compatíveis com perdas não técnicas?**

Dessa pergunta derivam duas questões auxiliares:

1.  **Como decompor as diferentes fontes de evidência sem confundir
    falhas de telemetria, eventos de rede, mudanças legítimas de
    comportamento e potenciais irregularidades?**

2.  **Como transformar as evidências produzidas pelo modelo em uma
    priorização rastreável e compatível com a capacidade real de
    inspeção?**

## Objetivo geral

Desenvolver, implementar e examinar experimentalmente uma metodologia para **detecção, localização e caracterização de padrões anômalos em redes de distribuição com dados de medição inteligente**, combinando métodos de aprendizado de máquina, análise temporal, grandezas elétricas e contexto cadastral/topológico, com especial atenção a padrões compatíveis com perdas não técnicas. A contribuição operacional consiste em transformar evidências analíticas em prioridades de investigação auditáveis; sua eficácia em inspeções é objeto de um protocolo de validação de campo distinto da avaliação computacional aqui apresentada.

## Objetivos específicos {#objetivos-especuxedficos}

Os objetivos específicos são:

1.  consolidar dados AMI, cadastrais, topológicos e de eventos em uma
    arquitetura temporalmente consistente;

2.  reconstruir a relação histórica entre UC e medidor físico, evitando
    que trocas de equipamento sejam interpretadas como anomalias de
    consumo;

3.  construir indicadores e features robustas capazes de representar
    magnitude, dispersão, tendência, qualidade da telemetria e
    comportamento intradiário;

4.  implementar e comparar a abordagem global originalmente proposta com
    uma arquitetura multivisão baseada em especialistas;

5.  combinar detecção de mudança de regime, perfis comportamentais e
    detecção multivariada de anomalias para representar alterações
    abruptas, graduais, recorrentes ou persistentes;

6.  utilizar regras físicas e regulatórias como mecanismos
    determinísticos de contextualização, evitando transferir ao ML
    decisões já definidas pelo domínio;

7.  produzir categorias de compatibilidade e evidências rastreáveis que
    permitam diferenciar candidatos a perdas não técnicas de eventos
    explicáveis por comunicação, manutenção ou contexto de rede;

8.  avaliar o impacto do método sobre o funil de candidatos, a
    capacidade de inspeção e o rendimento operacional, mantendo a
    inspeção de campo como confirmação final.

## Rastreabilidade dos objetivos às evidências

Para avaliar o atendimento dos objetivos sem confundir implementação e desempenho em campo, as evidências são organizadas por nível de observação. *Implementação* designa uma função identificável no código; *experimento controlado* indica resultados sobre dados gerados artificialmente; *estudo de caso AMI* designa telemetria mascarada efetivamente observada; *protocolo de campo* define indicadores cuja estimativa depende da inspeção.

| Objetivo | Evidência no trabalho | Critério observável |
|:--|:--|:--|
| Integração e temporalidade | Arquitetura UC–NIO; contratos e tabelas históricas | Grão e vigência explícitos, rastreabilidade da origem |
| Extração em escala | Motor Oracle, partições e `sink_*` | Reprocessamento controlado, lotes persistidos, auditoria |
| Qualidade da telemetria | `S_comm`, cobertura e calendário | Datas/canais distinguíveis de consumo nulo |
| Caracterização temporal | CPD, dias típicos e formas de curva | Transições e padrões recorrentes descritos |
| Detecção multivariada | IF, LOF, ECOD, PCA e composição | Scores, casos arquetípicos e diferenças de ranking |
| Interpretação física/contextual | Regras elétricas, alarmes, topologia | Hipóteses compatíveis, sem conclusão causal automática |
| Comparação metodológica | Baseline e ablação M0–M5 | Modificações do funil e estabilidade de prioridades |
| Suporte operacional | TUI, Cockpit, carteira e modelo econômico | Fluxo de decisão e métricas definidas; yield apenas com campo |

Essa correspondência estabelece dois tipos de conclusão: propriedades técnicas que o trabalho consegue demonstrar diretamente e hipóteses de impacto operacional cuja confirmação depende de resultados independentes de campo.

## Hipótese de pesquisa {#hipuxf3tese-de-pesquisa}

A hipótese central é que uma arquitetura que separa explicitamente as
fontes de anomalia e incorpora estrutura temporal produz uma lista de
candidatos **mais interpretável e operacionalmente mais seletiva** que
uma abordagem global baseada apenas em raridade multivariada. Essa
hipótese será testada comparando a metodologia original com refinamentos
sucessivos, preservando o mesmo conjunto de dados, horizonte temporal e
protocolo de avaliação sempre que possível.

# Fundamentação metodológica {#fundamentauxe7uxe3o-metodoluxf3gica}

## Encadeamento do argumento e lacuna enfrentada

Há quatro decisões científicas sucessivas neste problema: definir **qual observação é confiável**, estabelecer **com qual histórico e com quais pares a UC deve ser comparada**, identificar **que forma de desvio ocorreu** e determinar **quais inferências são permitidas por essa evidência**. Tratar essas decisões como uma única classificação binária gera um atalho conceitual: uma saída numérica passa a representar simultaneamente integridade do canal, diferença estatística, evento elétrico e irregularidade comercial. Essas dimensões não são equivalentes.

Trabalhos sobre perdas não técnicas demonstram que a seleção e a ordenação de consumidores são problemas relevantes, e não apenas a classificação em conjuntos experimentais [@sau2023; @han2026]. A literatura de análise de outliers, por sua vez, distingue raridade global, densidade local e desvios em séries temporais, com diferentes hipóteses sobre a distribuição dos dados [@aggarwal2017]. Já a interpretação de distúrbios elétricos requer reconhecer quando regras físicas fornecem informações que não deveriam ser aprendidas cegamente a partir de dados [@bollen2007; @bollen2006].

A lacuna metodológica tratada neste trabalho está na **articulação dessas três perspectivas sobre dados AMI heterogêneos e temporalmente versionados**. O mesmo registro deve ser visto como medição física, evento temporal, observação estatística e evidência potencial para uma decisão operacional. Por isso, a decomposição multivisão não é somente uma opção de organização de software: evita que uma falha de comunicação seja aprendida como padrão elétrico e que uma troca de medidor seja confundida com uma mudança de regime da instalação.

## Anomalia, perda não técnica e confirmação de campo {#anomalia-perda-nuxe3o-tuxe9cnica-e-confirmauxe7uxe3o-de-campo}

Neste trabalho, os termos **anomalia**, **padrão compatível com perda
não técnica** e **fraude confirmada** não são equivalentes. Anomalia
corresponde a uma observação, sequência ou regime que se afasta de uma
referência estatística ou física. Perda não técnica é uma hipótese
operacional que pode ser reforçada por múltiplas evidências. Fraude
confirmada exige validação por processo de campo ou por fonte
corporativa considerada ground truth.

Essa distinção é importante porque os modelos não supervisionados operam
sem rótulos de fraude. Um score elevado indica raridade segundo a
representação utilizada, e não probabilidade causal de furto. A
metodologia procura, portanto, deslocar a interpretação de um único
score para uma composição de evidências independentes.

## Resolução temporal dos dados AMI e limite de inferência {#resoluuxe7uxe3o-temporal-dos-dados-ami-e-limite-de-inferuxeancia}

A literatura clássica de qualidade da energia trabalha frequentemente
com formas de onda de alta frequência, apropriadas para transitórios,
afundamentos, elevações e harmônicos [@bollen2006]. Dados AMI em
cadência de minutos ocupam outro domínio. Fenômenos subcíclicos que não
foram registrados explicitamente como eventos pelo medidor já foram
integrados ou filtrados antes de chegar ao Data Lake.

A contribuição da literatura de processamento de sinais para este
trabalho é, portanto, principalmente estrutural: segmentação temporal,
extração de representações e classificação devem ser tratadas como
etapas distintas; métodos determinísticos e estatísticos podem coexistir
[@bollen2007]. A análise proposta concentra-se em escalas de horas, dias
e semanas, nas quais mudanças de comportamento, persistência e
recorrência são observáveis.

## Estatística robusta e representação do histórico {#estatuxedstica-robusta-e-representauxe7uxe3o-do-histuxf3rico}

Médias simples são sensíveis a picos e podem ocultar heterogeneidade.
Por isso, a metodologia utiliza medidas robustas como mediana, MAD e
quantis para descrever o comportamento histórico. Essas estatísticas são
especialmente úteis quando o interesse é reconhecer uma mudança no
patamar ou na dispersão sem permitir que poucos eventos extremos
redefinam o próprio baseline.

Para uma série diária de energia $E_d$, a mediana histórica é dada por:

$$\tilde{E}=\operatorname{median}(E_d),$$

e a dispersão robusta pode ser descrita por:

$$MAD=\operatorname{median}(|E_d-\tilde{E}|).$$

Essas medidas não substituem a série temporal. Elas funcionam como uma
representação complementar, utilizada tanto para construir features
quanto para gerar bandas de referência interpretáveis nos gráficos de
acompanhamento.

## Detecção de anomalias e raridade multivariada {#detecuxe7uxe3o-de-anomalias-e-raridade-multivariada}

A detecção de outliers busca observações incompatíveis com uma
distribuição de referência [@aggarwal2017]. Neste projeto, métodos como
Isolation Forest, LOF e PCA residual são avaliados como formas distintas
de representar raridade. A abordagem original combinava diferentes
detectores sobre um espaço global de features. A arquitetura atual
mantém esse baseline, mas restringe o subespaço de cada especialista
para impedir que o modelo elétrico aprenda, por exemplo, que ausência de
pacote é uma propriedade elétrica da UC.

Essa separação é uma mudança de representação, não de objetivo. O score
continua sendo utilizado para localizar casos raros, mas passa a ser
interpretado dentro do domínio ao qual as features pertencem.

## Change Point Detection e mudança de regime {#change-point-detection-e-mudanuxe7a-de-regime}

Detecção de mudança de regime (*Change Point Detection* - CPD) responde
a uma pergunta diferente da detecção de outliers: **quando a dinâmica da
série mudou?** Métodos de CPD segmentam uma série em regiões com
propriedades estatísticas relativamente homogêneas [@truong2020].

Uma UC pode não ser rara em determinado dia e ainda assim ter migrado
para um novo patamar persistente. Da mesma forma, uma fraude ou defeito
estabelecido antes da janela de observação pode já parecer estável. Por
isso, CPD e anomaly detection são complementares.

Para reduzir falsos alarmes produzidos pela sazonalidade, a detecção de
mudança deve preferencialmente operar sobre resíduos ou séries
condicionadas por horário e tipo de dia. Em monitoramento causal,
somente informações anteriores ao instante de decisão podem compor a
referência; em auditoria retrospectiva, janelas posteriores podem ser
utilizadas desde que o uso seja explicitamente descrito.

## Perfis diários e recorrência comportamental {#perfis-diuxe1rios-e-recorruxeancia-comportamental}

O consumo de uma UC possui forma intradiária. Dois consumidores podem
apresentar a mesma energia diária e, ainda assim, ter curvas
completamente diferentes. Uma representação diária normalizada permite
separar **magnitude** de **morfologia**.

Para um dia $d$ com $T$ intervalos, define-se o vetor:

$$\mathbf{x}_d=[x_{d,1},x_{d,2},\ldots,x_{d,T}].$$

Uma representação de forma pode ser obtida por normalização da curva,
enquanto a energia total diária é preservada como feature independente.
PCA e clustering podem então construir arquétipos de dia. A sequência
dos arquétipos atribuídos à mesma UC permite distinguir um dia isolado
de uma alteração persistente na rotina.

Assim, um domingo sempre diferente pode ser parte do padrão normal da
instalação; o sinal de interesse aparece quando a frequência, a posição
semanal ou a morfologia desses dias começa a mudar. Essa camada,
denominada $S_{behavior}$, complementa o CPD: o primeiro descreve **qual
tipo de dia está ocorrendo e com que frequência**; o segundo procura
**quando a distribuição temporal mudou**.

## Determinístico versus estatístico {#determinuxedstico-versus-estatuxedstico}

Nem todo fenômeno deve ser aprendido. Grandezas cuja interpretação é
diretamente definida pela física ou por critérios regulatórios devem
permanecer em uma camada determinística. O PRODIST Módulo 8 é utilizado
como referência para contextualização de tensão e qualidade do
fornecimento [@aneel2026], sempre respeitando a diferença entre a
cadência disponível no AMI e o procedimento formal de avaliação
regulatória.

O papel do ML concentra-se onde a fronteira não é conhecida a priori:
comportamento, raridade, combinação de features, recorrência e mudança
de regime. Essa divisão reduz opacidade e permite que cada evidência
tenha significado técnico explícito.

## Heterogeneidade topológica e comparação entre pares {#heterogeneidade-topoluxf3gica-e-comparauxe7uxe3o-entre-pares}

A população analisada não é homogênea. Consumidores conectados a
alimentadores com composição residencial, comercial ou industrial
distintas podem apresentar distribuições de consumo igualmente
distintas. Por isso, o alimentador e a vizinhança elétrica são usados
como contexto de comparação, e não como rótulo causal.

A mesma lógica vale para transformadores: eventos simultâneos entre
várias UCs eletricamente próximas aumentam a plausibilidade de causa
comum; desvios persistentes restritos a uma única instalação aumentam a
relevância de hipóteses individuais. A corroboração topológica deve,
portanto, modificar a prioridade e a interpretação, mas não produzir
sozinha um veredicto de fraude.

## Justificativa comparativa das técnicas escolhidas

Isolation Forest é útil para desvios globalmente raros, mas não assegura que uma observação pouco extrema na cidade inteira seja normal entre consumidores semelhantes [@liu2008]. LOF introduz uma referência de densidade local, à custa da sensibilidade à métrica, à dimensionalidade e ao tamanho do conjunto de vizinhos [@breunig2000]. ECOD utiliza extremidades de distribuições marginais; PCA, por outro lado, explicita subespaços dominantes e resíduos de reconstrução. Nenhuma dessas medidas, isoladamente, resolve *quando* um comportamento mudou, motivando o uso separado de métodos de detecção de ponto de mudança [@truong2020].

O agrupamento de perfis diários com K-Means não é empregado como detector de fraude: seu papel consiste em definir referências comparáveis de forma e, quando há histórico suficiente, estudar alterações na frequência de tipos de dia. Regras elétricas e evidências topológicas funcionam como instrumentos de interpretação e exclusão de hipóteses, preservando a diferença entre uma ocorrência comum de rede e um fenômeno localizado. Assim, o ganho esperado da arquitetura é **redução de ambiguidade**, não aumento automático de uma métrica de classificação.

A superioridade da metodologia refinada deve ser demonstrada por efeito marginal dos componentes e, quando houver rótulos de inspeção, pelo rendimento operacional. A mera redução do número de UCs sinalizadas não garante melhora: também pode excluir casos relevantes. Essa consequência orienta os experimentos de ablação e sensibilidade apresentados adiante.

# Evolução metodológica e comparação com a abordagem original {#evoluuxe7uxe3o-metodoluxf3gica-e-comparauxe7uxe3o-com-a-abordagem-original}

## Abordagem original como baseline

A metodologia original pode ser resumida em quatro etapas:

1.  agregação do histórico em uma janela fixa;

2.  construção de features estatísticas de consumo e grandezas
    elétricas;

3.  aplicação de modelos não supervisionados globais e ensemble de
    scores;

4.  ordenação das UCs pela raridade multivariada.

Essa abordagem continua sendo importante porque fornece um baseline
simples, reproduzível e diretamente relacionado ao objetivo original do
trabalho. Ela responde à pergunta: **quais UCs apresentam combinações de
features incomuns quando toda a população é analisada no mesmo espaço?**

## Motivos para o refinamento

O refinamento surgiu à medida que a aplicação passou a operar sobre
dados reais e históricos mais longos. Foram observadas seis situações
que justificam decompor o problema:

1.  **ausência de telemetria pode produzir aparente queda de consumo**,
    exigindo um especialista de comunicação;

2.  **troca de medidor pode produzir descontinuidade artificial**,
    exigindo reconstrução temporal UC--NIO;

3.  **uma janela fixa pode diluir mudanças recentes ou normalizar
    regimes antigos**, motivando CPD;

4.  **consumidores com energia agregada semelhante podem ter formas
    diárias opostas**, motivando $S_{behavior}$;

5.  **alimentadores e vizinhanças possuem distribuições distintas**,
    motivando comparação contextual;

6.  **certas regras são físicas ou regulatórias**, e não precisam ser
    reaprendidas por um detector estatístico.

Nenhum desses pontos invalida o baseline original. Eles identificam
variáveis de confusão e novas fontes de informação que podem ser
adicionadas progressivamente.

## Arquitetura refinada

A abordagem atual preserva a detecção multivariada, mas organiza as
evidências em subproblemas:

    RAW AMI + CADASTRO + TOPOLOGIA
                |
                +--> qualidade / comunicação --------> S_comm
                |
                +--> série temporal -----------------> CPD
                |
                +--> forma dos dias -----------------> S_behavior
                |
                +--> grandezas elétricas ------------> S_elec
                |
                +--> alarmes ------------------------> S_tamper
                |
                +--> tensão / regras ----------------> determinístico
                |
                +--> alimentador / transformador ----> contexto
                                          |
                                          v
                                evidências integradas
                                          |
                                          v
                                  ranking / inspeção

A principal mudança é epistemológica: cada score precisa responder a uma
pergunta específica. O sistema deixa de exigir que um único modelo
aprenda simultaneamente qualidade de dados, comportamento, física e
contexto.

## Desenho de ablação {#desenho-de-ablauxe7uxe3o}

A superioridade da abordagem refinada será avaliada de forma
incremental, utilizando a metodologia original como referência:

| Versão | Componentes adicionados | Pergunta de avaliação |
|:--|:--|:--|
| $M_0$ | Baseline global original | Qual desempenho da formulação inicial? |
| $M_1$ | Saneamento temporal UC–NIO | Quais eventos coincidem com trocas de equipamento? |
| $M_2$ | $S_{comm}$ e gate de qualidade | Quantos candidatos exigem triagem de comunicação? |
| $M_3$ | CPD e atributos pré/pós | A mudança de regime é localizada com maior consistência? |
| $M_4$ | $S_{behavior}$ e perfis diários | A recorrência acrescenta informação além da energia agregada? |
| $M_5$ | Regras físicas, $S_{tamper}$ e topologia | A contextualização reduz ambiguidade e altera a fila? |

A comparação será realizada com métricas diferentes conforme o estágio.
Antes do ground truth de campo, serão utilizadas redução do funil,
estabilidade, consistência entre especialistas, qualidade dos dados e
capacidade de explicar cada candidato. Após a disponibilidade de
inspeções, entram rendimento@K, precisão das categorias, taxa de
confirmação e impacto sobre a carteira operacional.

## Critério de superioridade {#crituxe9rio-de-superioridade}

A nova abordagem será considerada superior se demonstrar simultaneamente
três propriedades:

1.  **preservar ou aumentar a capacidade de localizar casos
    confirmados**, em comparação ao baseline;

2.  **reduzir casos explicáveis por fatores não relacionados a
    irregularidade individual**, como falha de comunicação, troca de
    medidor ou evento coletivo;

3.  **produzir evidências mais interpretáveis e uma fila mais compatível
    com a capacidade operacional**.

Isso evita definir superioridade apenas pelo número bruto de anomalias.
Um sistema que sinaliza menos UCs pode ser melhor se a redução remover
predominantemente casos espúrios e aumentar o rendimento das inspeções.

## Delimitação experimental {#delimitacao-experimental}

A área de estudo é considerada em sua população completa de unidades
consumidoras, e não por extrapolação de recortes exploratórios. A
unidade de análise é a UC em um instante de referência; os medidores
físicos são relacionados a ela por vínculos de instalação válidos no
tempo. Dessa forma, os procedimentos de preparação, treinamento e
pontuação preservam consistência entre identidade, período de observação
e objetivo da análise.

Duas fontes de evidência são examinadas em conjunto, com finalidades
distintas: dados AMI reais, apresentados em estudos de caso com
identificação mascarada, e uma população sintética controlada destinada
a demonstrar o funcionamento das técnicas de aprendizado de máquina. As
curvas sintéticas permitem conhecer o mecanismo gerador e estudar pontos
cegos dos detectores; os casos reais permitem interpretar sinais e
limitações do processo analítico em situações observadas.

Estatísticas de prevalência de irregularidades e rendimento operacional
dependem de resultados de inspeção em campo. Por essa razão, não são
inferidas a partir de scores não supervisionados nem extrapoladas dos
exemplos ilustrados. A comparação entre a metodologia original e a
arquitetura multivisão é organizada por um protocolo de ablação
reprodutível e métricas apropriadas à disponibilidade das evidências.

# Arquitetura de dados

## UC como entidade de análise e NIO como ativo físico {#uc-como-entidade-de-anuxe1lise-e-nio-como-ativo-fuxedsico}

A telemetria é produzida por um medidor físico, identificado por NIO,
mas o objeto comercial e operacional é a Unidade Consumidora. A hipótese
simplificadora

> **Hipótese simplificadora:** `1 UC = 1 NIO = 1 série`.

não se mantém ao longo do tempo. Medidores são substituídos, podem mudar
de cadência, podem reiniciar registradores e podem ser reutilizados.

A série da UC deve, portanto, ser reconstruída a partir do histórico de
instalações:

$$X_u(t)=\sum_k \mathbf{1}[t\in I_{u,k}]Y_k(t),$$

onde $Y_k(t)$ é a série do medidor $k$ e $I_{u,k}$ é seu intervalo de
vigência na UC $u$.

No dia de troca, a solução precisa admitir a coexistência de dois
equipamentos durante partes distintas do dia. Registradores acumulados
devem ser diferenciados por medidor antes da fusão em nível de UC.

## Camadas semânticas {#camadas-semuxe2nticas}

A arquitetura de dados é organizada em entidades com granularidades
explícitas:

-   `uc_context`: atributos cadastrais e contextuais da UC;

-   `meter_installation_history`: vínculo UC-NIO com vigência;

-   `electrical_hierarchy`: subestação, alimentador, transformador/posto
    e localização;

-   `ami_interval`: grandezas intervalares;

-   `ami_instantaneous`: grandezas instantâneas;

-   `ami_registers`: registradores acumulados;

-   `alarm_events`: alarmes e eventos assíncronos.

O dataset de modelagem utiliza o grão conceitual

$$UC \times cutoff\_date.$$

Cada linha representa o estado conhecido de uma UC até uma data de
corte. As colunas são separadas em:

-   `id__*`: identificadores, fora de $X$;

-   `meta__*`: tempo, qualidade, gating e controle;

-   `x__*`: features permitidas aos modelos;

-   `y__*`: alvos, quando existirem e forem validados.

Essa convenção reduz a chance de inclusão acidental de identificadores,
metadados futuros ou targets dentro do vetor de treinamento.

## Temporalidade e leakage

A validação do pipeline exige distinguir tempo de medição, tempo de
ocorrência do evento, tempo de recepção e tempo de disponibilidade para
a análise. Em uma avaliação online, uma observação só pode ser utilizada
se estivesse disponível no cutoff simulado.

As principais formas de leakage consideradas são:

-   uso de dados posteriores ao cutoff;

-   uso de cadastro atual para representar estado histórico sem
    vigência;

-   ajuste de scaler, imputer ou encoder com dados do teste;

-   utilização de target ou proxy direto do target nas features;

-   construção de janelas que atravessam o período reservado para
    avaliação;

-   sobreposição excessiva entre janelas que cria dependência não
    tratada.

## Extração inteligente e processamento em escala {#extrauxe7uxe3o-inteligente-e-processamento-em-escala}

A plataforma não pressupõe que todos os dados estejam disponíveis em uma
única tabela ou que uma extração de longo período possa ser mantida
integralmente em memória. A ingestão foi desenhada para trabalhar em
ciclos independentes por dia e por lote de medidores, permitindo que o
estado do Data Lake determine o trabalho ainda necessário.

A chamada **sincronização inteligente** realiza três funções: identifica
as partições temporais já consolidadas, detecta lacunas intermediárias e
calcula o intervalo que ainda precisa ser extraído até o último dia
completo. Com isso, uma execução interrompida não precisa ser reiniciada
desde o início. Dias corretamente persistidos continuam sendo parte
válida do histórico.

A extração pode ser acionada em quatro granularidades operacionais:

-   dia de referência isolado;

-   período histórico definido pelo usuário;

-   alimentador específico;

-   *auto-backfill* até o último dia completo.

No acesso ao Oracle, os identificadores são divididos em lotes e as
conexões dos *workers* são reutilizadas. Essa estratégia evita tanto
consultas monolíticas que pressionam memória do banco quanto o custo de
criar uma nova sessão a cada lote. O acompanhamento é apresentado em
tempo real pela TUI por uma fila geral e barras individuais de
*workers*.

## Polars, execução lazy e persistência por sinking {#polars-execuuxe7uxe3o-lazy-e-persistuxeancia-por-sinking}

O volume de telemetria torna inadequado depender de concatenações
sucessivas de DataFrames Python materializados integralmente em RAM. O
pipeline utiliza Polars para privilegiar operações colunares e um plano
de transformação *lazy*. Sempre que a transformação permite, o resultado
é materializado diretamente em disco por mecanismos de *sinking*, como
`sink_parquet()` e `sink_csv()`.

O propósito de `sink` não é apenas acelerar o processamento. Ele altera
a tolerância a falhas do sistema. Em vez de exigir que um período
completo termine para produzir um único artefato, cada partição diária
consolidada torna-se imediatamente persistente e reutilizável. Assim,
uma falha de rede, interrupção manual ou reinício da máquina não
invalida os dias já concluídos.

Essa estratégia estabelece três propriedades importantes:

1.  **bounded memory:** o tamanho do histórico não precisa crescer
    proporcionalmente ao uso de RAM;

2.  **checkpoint natural:** o próprio particionamento diário funciona
    como ponto de recuperação;

3.  **resultados parciais auditáveis:** partes concluídas podem ser
    verificadas antes de toda a campanha terminar.

A escolha por Parquet fornece compressão e leitura colunar eficiente,
enquanto a exportação opcional em CSV preserva interoperabilidade com
ferramentas externas. A TUI também permite sincronizar formatos ausentes
sem repetir a consulta ao banco corporativo.

## Continuidade temporal e auditoria do Data Lake

Uma série temporal só pode ser analisada corretamente quando se conhece
também sua ausência. Por isso, a plataforma mantém um calendário mensal
de continuidade. Cada dia é classificado conforme a existência das
partições esperadas e, quando aplicável, conforme a presença dos
formatos Parquet e CSV.

O calendário tem função metodológica: ele impede que uma lacuna de
extração seja confundida com ausência de medição da UC. Também tem
função operacional, pois permite ao usuário identificar rapidamente
períodos incompletos e acionar *backfill* ou sincronização de formatos.

Além do calendário, a aplicação executa auditorias de saúde do Data
Lake. As verificações incluem presença de partições, contratos de
schema, contagens, chaves esperadas, disponibilidade das camadas
necessárias ao treinamento e execução da suíte de testes. O resultado é
persistido em relatório de saúde com estados como `HEALTHY`, `WARNING`,
`INCOMPLETE` e `CRITICAL`.

## Arquitetura geral da aplicação {#arquitetura-geral-da-aplicauxe7uxe3o}

A Figura a seguir resume o encadeamento entre ingestão, governança,
modelagem e uso operacional. A TUI atua como camada de orquestração da
engenharia; a interface web é tratada como camada de consumo analítico e
acompanhamento das prioridades.

![Arquitetura geral da aplicação, da extração à retroalimentação de
campo.](figures/app_architecture.pdf){#fig:app_arch width="95%"
height="0.78\\textheight"}

A arquitetura representada na figura separa fontes transacionais, artefatos persistidos, treinamento e apresentação. Essa organização é relevante porque uma anomalia exibida no Cockpit precisa ser reconstituída até o arquivo e o instante de origem; a figura não representa um fluxo único e atômico, pois cada estágio possui seus próprios checkpoints e verificações.

# Engenharia do pipeline: das fontes ao dataset analítico

## Requisitos, restrições e unidade de processamento

A engenharia de dados é um componente científico deste estudo porque erros de identidade, tempo ou persistência modificam diretamente o resultado do detector. Há requisitos simultâneos: extrair a população selecionada sem depender de um arquivo monolítico; suportar interrupções e reexecuções; preservar o grão original das fontes; separar observação ausente de valor físico nulo; reconstruir a relação entre UC e equipamento no instante correto; e produzir um dataset de treinamento reproduzível.

A implementação concentra a extração no script `run_araucaria_full.py` e utiliza `src/pipeline.py` e os módulos `src/datasets`, `src/features`, `src/checks`, `src/models` e `src/tui`. O script orquestra campanhas de extração; as rotinas de normalização distinguem medições intervalares, instantâneas e registradores; os escritores persistem camadas semanticamente identificadas; a TUI oferece acesso coordenado a extração, auditoria e treinamento. Essa separação torna possível estudar cada estágio sem assumir que sucesso de uma consulta significa qualidade das séries usadas pelos modelos.

O processamento é particionado prioritariamente por **dia de referência**, com subconjuntos de NIOs consultados por lotes. O grão de armazenamento varia conforme a fonte, enquanto o grão de treinamento é uma UC em uma data de corte. Essa distinção impede que registros de medidores diferentes ou amostras de dias distintos sejam tratados como observações estatisticamente independentes de uma mesma entidade.

## Inventário das fontes e seus contratos

O CIS fornece situação e identidade da UC, vínculos de equipamentos, datas de instalação e retirada e atributos comerciais; GEO fornece alimentador, subestação e hierarquia espacial; MDM/ORCA fornece as observações de medição; eventos do equipamento permitem examinar alarmes. Tabelas de clima e outras informações contextuais podem enriquecer comparações, mas devem manter o carimbo temporal que comprova sua disponibilidade no momento da decisão.

| Camada | Grão | Conteúdo | Verificação essencial |
|:--|:--|:--|:--|
| `uc_context` | UC e versão/instantâneo | classe, situação, características cadastrais | vigência e duplicidade |
| `meter_installation_history` | UC–NIO–intervalo de validade | instalação, retirada, substituição | sobreposição temporal e identificação |
| `electrical_hierarchy` | UC/nó e referência topológica | alimentador, posto, transformador | chaves e versão da topologia |
| `ami_interval` | NIO–instante | energia intervalar e grandezas médias | cadência, duplicados, faltantes |
| `ami_instantaneous` | NIO–instante | tensão/corrente instantâneas | amostragem, fonte e unidade |
| `ami_registers` | NIO–instante | contadores acumulados | reset e delta por equipamento |
| `alarm_events` | NIO–evento | código, horário de origem e recepção | latência, relógio e duplicidade |
| `meter_day_features` | NIO–dia–versão | descritores diários | fonte e versão da transformação |
| `uc_window_features` | UC–corte–janela | histórico consolidado | censura, vazamento e sobreposição |
| `training_dataset` | UC–cutoff | matriz e metadados para treino | divisões temporais e rastreabilidade |

A Tabela anterior explicita por que não é adequado reunir todas essas fontes por uma chave estática. A energia intervalar pertence a um equipamento e a um instante, enquanto a decisão operacional recai sobre a UC. Alarmes exigem atenção especial: horário de ocorrência e horário de chegada ao sistema podem diferir. Um evento recebido depois do instante de corte não deve entrar retroativamente nas features de uma decisão operacional passada.

## Descoberta da população, seleção por alimentador e vigência

A campanha começa com a seleção do universo de UCs e a recuperação dos vínculos com equipamentos e alimentadores. O mecanismo admite a população completa da área de estudo ou a restrição a um alimentador, útil tanto para depuração quanto para modelos contextuais. Uma população filtrada não pode, entretanto, substituir o universo completo na inferência de prevalência de irregularidades.

Para cada UC e instante $t$, a associação ao medidor exige:

$$t_{\mathrm{inst},j}\le t\quad\land\quad(t_{\mathrm{ret},j}=\varnothing\ \lor\ t\le t_{\mathrm{ret},j}).$$

A igualdade na data de retirada expressa a regra de inclusão do vínculo histórico; quando há troca no mesmo dia, a resolução por data não decide o período intradiário de cada NIO. Nesses casos, timestamps mais precisos ou uma política explícita de desambiguação são necessários. O sistema não deve somar dois registradores acumulados ou tratar o salto de contador entre medidores como energia consumida. Primeiramente são calculadas diferenças válidas **dentro de cada NIO**; depois, a relação de vigência permite reconstruir a série UC.

Esse tratamento reduz dois erros opostos: excluir telemetria antiga por consultar somente o medidor atual e incorporar medidas de equipamento não mais instalado. Como consequência, alterações no cadastro podem ser representadas como contexto de mudança, não como prova automática de alteração na carga elétrica.

## Seleção inteligente de datas e recuperação de lacunas

O motor examina as partições previamente materializadas para descobrir quais dias já possuem arquivo de medição. A rotina `find_datalake_date_coverage()` identifica datas a partir de partições de `ami_instantaneous` em Parquet ou CSV; `determine_smart_extraction_dates()` combina dias após o último arquivo disponível com lacunas internas identificadas no calendário, até o dia anterior ao da execução.

É importante distinguir **presença física de arquivo** de **integridade semântica de uma partição**. No snapshot de código analisado, a primeira verificação utiliza existência e tamanho superior a um limite mínimo; por isso, uma partição presente não recebe automaticamente certificado de completude de todas as tabelas e todos os NIOs. O segundo controle é fornecido pelas auditorias do Data Lake, que comparam formatos, datas, contagens e disponibilidade. Essa distinção é central para a interpretação do calendário: uma data marcada como presente não implica que cada medidor esperado tenha transmitido todos os intervalos.

O modo manual aceita dia específico, intervalo histórico ou filtro de alimentador. Isso permite recuperação direcionada, reprocessamento de datas problemáticas e experiências controladas sem descarregar novamente toda a população. A finalidade operacional do auto-backfill é reduzir trabalho repetitivo e conservar uma sequência histórica contínua, sempre respeitando a retenção efetiva da fonte de dados.

## Extração Oracle em lotes e paralelismo de consultas

O script de execução carrega as consultas SQL e divide a lista de identificadores em lotes. O uso do tipo Oracle `SYS.ODCIVARCHAR2LIST` permite enviar coleções de NIOs como parâmetro de consulta, evitando montar comandos SQL de tamanho variável por concatenação de strings. As conexões de consulta ao MDM são mantidas por worker, reduzindo a necessidade de abrir uma sessão para cada lote. O progresso da fila geral e das tarefas individuais é visualizado com barras do Rich.

No código disponível, cada worker executa uma consulta e recebe os resultados do **lote** com `fetchall()`, transforma esse lote em Polars e o persiste em Parquet. Consequentemente, a estratégia é **materialização limitada por lote seguida de processamento lazy de arquivos**, e não um fluxo contínuo byte a byte diretamente do cursor Oracle ao arquivo final. Essa precisão importa: o tamanho do lote controla o pico de memória do estágio de aquisição; o `sink_*` controla, sobretudo, a etapa posterior de consolidação/normalização sem concatenar todos os lotes em um único DataFrame residente.

A paralelização concentra-se nas operações de I/O de banco de dados e em tarefas independentes. Ela não elimina limitações do banco, do link ou do Python; aumento indiscriminado de workers pode piorar latência, aumentar sessões concorrentes e pressionar recursos corporativos. Parâmetros como tamanho do lote, workers e formato são, portanto, variáveis de desempenho a medir, não constantes universais.

## Persistência intermediária, Polars e `sink_*`

Cada lote MDM concluído produz um arquivo temporário compactado. Depois do término da fila do dia, o sistema constrói um `LazyFrame` a partir das leituras `pl.scan_parquet(...)` e as combina com `pl.concat` com concatenação vertical flexível. A escrita com `sink_parquet` ou `sink_csv` permite executar o plano sobre os arquivos de entrada sem exigir a materialização integral do dia como um único objeto de memória. A mesma estratégia aparece na consolidação das três saídas normalizadas.

Há duas formas de parcialidade com consequências diferentes: **checkpoint de lote**, que evita uma consulta repetida ao retomar parte da campanha, e **partição diária consolidada**, que serve de entrada auditável para etapas posteriores. A primeira é recuperável, mas pode exigir nova validação; a segunda deve possuir esquema, contagem, janela, formato e manifesto compatíveis. A escrita por sinking, isoladamente, não fornece transação atômica, garantia de ausência de duplicados ou integridade do conteúdo. Essas propriedades exigem verificações adicionais, particularmente quando há falhas durante a escrita ou mudanças de esquema.

Arquivos Parquet comprimidos são favorecidos na computação por coluna, preservação de tipos e leitura seletiva; CSV é útil para inspeção, compatibilidade e intercâmbio. O módulo de conversão permite criar o formato ausente a partir da cópia existente, evitando uma consulta desnecessária à origem. Ainda assim, igualdade de formato não equivale a igualdade de linhas: a auditoria precisa comparar cardinalidade, esquema e, quando aplicável, checksums.

## Normalização dos três canais de medição

O MDM entrega curvas que podem estar representadas como mapas ou sequências serializadas em JSON. A função de normalização identifica slots temporais, detecta ou interpreta cadência e produz relações longas com uma observação por instante. Na operação foram consideradas cadências heterogêneas; portanto, não é correto impor 96 amostras por dia a todo medidor. O número esperado deve ser calculado por canal e pela cadência válida:

$$N_{\mathrm{esp}}(d,c)=\frac{1440}{\Delta t_{d,c}},\qquad C(d,c)=\frac{N_{\mathrm{obs}}(d,c)}{N_{\mathrm{esp}}(d,c)}.$$

A razão é uma simplificação para um dia completo em horário local de 24 horas; duplicações, mudanças de horário e diferentes convenções de timestamp exigem política própria. A normalização separa `ami_interval`, `ami_instantaneous` e `ami_registers`, preservando a semântica de cada variável e evitando agregar medidas de natureza diferente como se fossem uma série homogênea.

Um exemplo ilustra o problema: se uma curva intervalar registra 80 pontos válidos de uma grade de 96, a cobertura é $80/96=0{,}8333$. Oito registros de energia nula continuam sendo **oito observações válidas**, não oito faltantes. Se a consulta retorna zero pontos, a causa ainda é indeterminada: pode decorrer de retenção, indisponibilidade da fonte, filtro incorreto ou ausência de comunicação. A classificação dessa falta pertence ao especialista de qualidade e aos metadados da extração.

## Construção e separação dos datasets de aprendizado

Depois da normalização, o pipeline calcula descritores diários e agregações por janelas históricas. O contrato recomendado é uma linha por UC e instante de corte $t_c$. Seu conjunto de entrada contém apenas observações anteriores ou disponíveis em $t_c$:

$$\mathbf{x}_{u,t_c}=f\big(\mathcal H_u[t_c-W,t_c],\mathcal C_u(t_c)\big),$$

em que $\mathcal H_u$ representa histórico de medição com vínculo válido, $\mathcal C_u$ o contexto disponível à época e $W$ a janela de observação. Colunas `x__` entram no estimador; `id__` identifica unidade e observação; `meta__` fornece contexto, roteamento e rastreabilidade; `y__` armazena rótulos externos quando existentes.

O mesmo consumidor pode aparecer em cortes sucessivos, de modo que a divisão aleatória de linhas criaria exemplos altamente correlacionados em treino e teste. O experimento deve considerar **separação temporal**, impedir sobreposição indevida entre janelas e, quando necessário, medir generalização em UCs não vistas. Imputadores, escalonadores, PCA, clusterizadores, scores de referência e thresholds precisam ser ajustados no conjunto de treinamento e aplicados posteriormente sem recalcular parâmetros a partir dos dados de avaliação.

## Manifestos, saúde do Data Lake e rastreabilidade da execução

A execução gera identificadores de campanha e utiliza manifestos para registrar operações, arquivos e contagens. A auditoria de integridade pode inspecionar dimensões de tabelas, continuidade diária, versões dos formatos, falhas de execução e divergências entre artefatos. O calendário Rich resume o status visual por dia e permite separar datas completamente ausentes das presentes em apenas um dos formatos. O relatório de saúde amplia essa leitura para a consistência do conjunto.

O controle mínimo reprodutível de uma partição inclui: data e canal, consulta ou versão do SQL, lote ou conjunto de NIOs, momento da extração, esquema, cardinalidade, intervalo temporal e estado do processamento. Para os modelos, acrescentam-se versão do dataset, catálogo de features, data de corte, pré-processadores, semente, hiperparâmetros, detector, política de limiar e resultado de pontuação. Sem esse encadeamento, dois rankings produzidos em dias distintos podem divergir sem que se saiba se a causa é mudança do consumo, correção histórica, variação do conjunto de referência ou atualização do código.

## Falhas recuperáveis e limites de consistência

O projeto utiliza cache de lotes e partições por dia para evitar perder toda a campanha por uma falha de rede ou interrupção. A utilidade desse mecanismo está na possibilidade de preservar resultados intermediários e reiniciar a partir de arquivos existentes. Entretanto, tolerância a interrupção não significa ausência de risco de arquivos truncados: checkpoints precisam ser validados antes do reaproveitamento. De forma semelhante, uma data sem linhas retornadas não demonstra que nenhuma medição existiu no sistema fonte.

Há também uma restrição de retenção histórica no MDM. A preservação local de dados extraídos é importante para permitir análises em janelas mais longas do que a janela mantida no banco de origem. Isso não autoriza reconstruir medições não coletadas: a lacuna histórica anterior à primeira extração permanece uma limitação do conjunto de evidências.

## Encadeamento operacional de uma campanha completa

O procedimento reproduzível pode ser descrito por onze estágios: (1) carregar configuração/SQL; (2) selecionar UCs e alimentadores; (3) obter histórico de instalação e hierarquia; (4) calcular datas faltantes; (5) dividir NIOs em lotes; (6) consultar Oracle com sessões por worker; (7) gravar cache Parquet por lote; (8) consolidar e normalizar intervalos, instantâneos e registradores com Polars; (9) extrair alarmes e computar descritores diários; (10) construir janelas UC–cutoff e auditar integridade; (11) treinar, versionar e pontuar modelos pela TUI. Cada transição possui artefato persistente ou verificação explícita.

A principal contribuição da plataforma não é simplesmente acelerar uma consulta: **preserva o caminho entre uma medição do equipamento e a evidência exibida ao analista**. O dado anômalo só se torna interpretável quando a unidade, o medidor instalado, o instante, o canal, a qualidade da transmissão e o modelo empregado podem ser recuperados conjuntamente.

# Análise exploratória do conjunto de dados {#anuxe1lise-exploratuxf3ria-do-conjunto-de-dados}

A análise exploratória de dados (EDA) ocupa posição estrutural no
trabalho. Ela não é tratada apenas como etapa visual anterior ao
treinamento, mas como mecanismo de validação das hipóteses que permitem
construir o dataset. Os exemplos reais mascarados demonstram diferenças
de perfil, análise semanal e interpretação de uma unidade destacada pelo
pipeline. As estatísticas de cobertura, heterogeneidade topológica e
distribuição de anomalias são definidas para o universo completo da área
de estudo, sem extrapolação de recortes exploratórios descontinuados.

## Objetivos da EDA

A EDA deverá responder, antes de qualquer comparação de modelos, às
seguintes perguntas:

-   qual é a população efetivamente elegível em cada data;

-   quantas UCs possuem medição inteligente e histórico suficiente para
    cada análise;

-   quais cadências aparecem em cada família de medidor e grandeza;

-   onde estão as lacunas de extração e onde estão as lacunas reais de
    telemetria;

-   como a população se distribui entre alimentadores, transformadores,
    classes e configurações de fase;

-   quantas trocas de NIO ocorrem dentro do horizonte;

-   qual é a distribuição da idade do vínculo UC--NIO;

-   quais grandezas apresentam caudas, saturação, zeros estruturais ou
    assimetrias relevantes;

-   quão diferentes são os formatos das curvas diárias mesmo quando o
    consumo diário agregado é semelhante;

-   quais alimentadores apresentam distribuição suficientemente distinta
    para justificar especialistas locais.

## EDA de integridade e cobertura temporal

A primeira família de gráficos deve reproduzir, em escala analítica, a
mesma lógica do calendário da TUI. Para cada fonte e data serão
avaliados: existência da partição, contagem de registros, cobertura
temporal esperada, quantidade de UCs com histórico suficiente e
divergências entre as camadas necessárias ao cruzamento.

Um *heatmap* `data x fonte` permitirá separar três fenômenos que, sem
essa camada, poderiam parecer equivalentes: ausência de extração,
ausência de telemetria no sistema fonte e ausência estrutural de
determinada grandeza. A mesma distinção será realizada por UC utilizando
cobertura por canal.

Para uma cadência esperada com $N_d$ pontos por dia, a completude diária
básica é

$$C_d = \frac{N_{obs,d}}{N_d}.$$

Entretanto, $N_d$ não será assumido globalmente. Ele será associado ao
canal e à cadência esperada da família de dados. Isso evita transformar
diferenças legítimas de aquisição em falsos gaps.

## EDA da relação UC--NIO {#eda-da-relauxe7uxe3o-ucnio}

A população será descrita a partir da UC como entidade de análise e do
NIO como ativo físico temporal. As estatísticas previstas incluem número
de vínculos por UC, duração dos vínculos, quantidade de substituições
dentro do período, UCs com troca intradia e quantidade de registros
afetados por mudança de equipamento.

A análise é especialmente importante porque uma troca de NIO pode
produzir descontinuidade nos registradores, mudança de cadência e
diferenças de disponibilidade. Sem a reconstrução temporal, o próprio
processo de manutenção criaria *change points* artificiais.

## EDA topológica por alimentador e transformador {#eda-topoluxf3gica-por-alimentador-e-transformador}

As distribuições serão avaliadas por alimentador antes de assumir que
uma única população global representa adequadamente a cidade. Serão
comparadas magnitude de carga, dispersão, composição
residencial/comercial, presença de GD, cobertura de telemetria,
quantidade de UCs por transformador e características das curvas
diárias.

O objetivo não é afirmar previamente que todo alimentador exige um
modelo próprio. A hipótese a testar é se a estratificação topológica
reduz heterogeneidade intragrupo e aumenta a capacidade de reconhecer
padrões locais. Alimentadores com pouca amostra poderão utilizar o
modelo global ou ser agrupados em coortes semelhantes.

## EDA de grandezas elétricas e features {#eda-de-grandezas-eluxe9tricas-e-features}

Serão avaliadas distribuições de energia ativa direta e reversa, tensão,
corrente, fator de carga, dispersão robusta, razões diurno/noturno, dias
úteis/finais de semana e variáveis de qualidade. A análise deverá
identificar escalas muito diferentes, features redundantes e efeitos de
zeros ou nulos estruturais antes da entrada em modelos baseados em
distância ou densidade [@zheng2018; @aggarwal2017].

A EDA também incorpora a comparação de UCs com consumo agregado
semelhante e formatos de curva distintos, documentada nos estudos de
caso. O objetivo é demonstrar por que a morfologia intradiária contém
informação que se perde na soma mensal ou mesmo diária.

## EDA comportamental por dia

Cada dia válido poderá ser representado como um vetor de carga. Em dados
de 15 minutos, por exemplo,

$$\mathbf{x}_{u,d}=[x_1,x_2,\ldots,x_{96}].$$

Para separar forma e magnitude serão mantidas duas representações. A
primeira preserva o total diário $E_{u,d}=\sum_t x_t$. A segunda
normaliza a curva:

$$p_{u,d,t}=\frac{x_{u,d,t}}{E_{u,d}+\epsilon}.$$

A exploração dos formatos intradiários, da regularidade semanal e das
transições entre padrões fundamenta a construção do especialista
comportamental. Os gráficos individuais e o experimento de agrupamento
permitem observar essas propriedades sem pressupor que cada cluster
represente uma causa física.

## Eixos de exploração e critérios de interpretação {#eixos-eda}

| Dimensão analítica | Pergunta respondida | Uso metodológico |
|:--|:--|:--|
| Calendário de cobertura | Quais períodos são utilizáveis? | Auditoria de integridade |
| Histórico disponível por UC | Há janela suficiente? | Elegibilidade temporal |
| Trocas UC–NIO | Quantas mudanças físicas ocorrem? | Controle de falsos CPDs |
| População por alimentador | Como a população se distribui? | Referências locais |
| Completude por canal | Que séries são confiáveis? | $S_{comm}$ |
| Quantis elétricos | Quais escalas e caudas existem? | Escalonamento e detectores |
| Curvas diárias | Que arquétipos aparecem? | $S_{behavior}$ |
| Arquétipo por dia da semana | Qual é a rotina recorrente? | Referência de calendário |
| Dias atípicos frequentes | A rotina se modifica? | Persistência |
| Interseção entre especialistas | Quais evidências convergem? | Funil operacional |

A separação entre disponibilidade de dados, integridade da medição e
comportamento elétrico assegura que a interpretação da EDA não confunda
ausência de observações com consumo nulo ou irregularidade. Estimativas
agregadas são definidas sobre a população consolidada e não são
substituídas por amostras exploratórias.

## Critérios de interpretação das distribuições exploratórias

Cada figura da EDA deve ser apresentada com seu denominador. Número de arquivos, número de NIOs e número de UCs não são intercambiáveis; equipamentos substituídos podem inflar contagens de NIOs sem ampliar o universo comercial. Da mesma forma, um alimentador mais populoso tende a produzir mais registros e mais candidatos absolutos, mesmo sem maior taxa de atipicidade. Por isso, comparações topológicas devem apresentar tanto contagens quanto taxas por UC elegível ou por unidade de exposição.

O exame da distribuição elétrica distingue variáveis de magnitude, variáveis de forma e variáveis de qualidade. É útil comparar distribuições com e sem separação por classe ou alimentador, pois uma grande dispersão global pode resultar apenas da mistura de populações heterogêneas. A presença de extremos deve ser investigada antes da winsorização ou imputação, preservando informação sobre eventos reais e evitando transformar um procedimento de limpeza em mecanismo de supressão de anomalias.

A recorrência de dias atípicos deve usar um denominador de **dias efetivamente válidos**. Para uma UC $u$, com indicador de atipicidade $A_{u,d}\in\{0,1\}$ e elegibilidade $Q_{u,d}$, calcula-se:

$$R_u(W)=\frac{\sum_{d\in W}A_{u,d}Q_{u,d}}{\sum_{d\in W}Q_{u,d}},\quad \sum_{d\in W}Q_{u,d}>0.$$

Essa forma evita interpretar como dias normais aqueles para os quais não foi possível reconstruir a curva. A comparação entre janelas deve ser estratificada por dia da semana e, quando possível, por calendário/estação; uma mudança legítima de rotina é uma hipótese concorrente de anomalia persistente, não um rótulo operacional inferido automaticamente.

A distribuição por alimentador fundamenta dois testes distintos: (i) ganho de explicabilidade da comparação por pares e (ii) ganho efetivo de detecção ou priorização. O primeiro pode ser demonstrado com ranking global versus local e vizinhanças coerentes. O segundo exige comparação contra resultados de inspeção e controle de tamanhos de amostra. Alimentadores pequenos devem adotar referência regional/global para evitar conclusões baseadas em poucos pares.

# Arquitetura analítica multivisão {#arquitetura-analuxedtica-multivisuxe3o}

A arquitetura atual evita a concentração de todos os sinais em um único
score caixa-preta. Cada família produz uma evidência com semântica
própria.

                         AMI + CONTEXTO
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
      Qualidade / Comunicação             Contexto cadastral
              |                                 |
            S_comm                              |
              +----------------+----------------+
                               v
                        Série elegível
                               |
                               v
                       Change Point Engine
                               |
                       mudança de regime
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
          S_behavior         S_elec          S_tamper
              |                |                |
              +--------+-------+-------+--------+
                       |               |
                       v               v
                  Regras físicas   Contexto espacial
                       +-------+-------+
                               v
                      Corroboração topológica
                               |
                               v
                    Prioridade de investigação

## Especialista de comunicação - $S_{comm}$ {#especialista-de-comunicauxe7uxe3o---s_comm}

O objetivo de $S_{comm}$ é responder se a telemetria possui integridade
suficiente e se o próprio comportamento de comunicação é atípico.
Features candidatas incluem cobertura, quantidade de gaps, maior gap,
concentração noturna de faltantes e latência de eventos.

Essa separação é essencial porque missingness pode ser informativa para
comunicação e, ao mesmo tempo, contaminante para uma inferência física.
O gating de qualidade deve ser parametrizável. Um valor como 70% de
cobertura em 30 dias pode ser utilizado inicialmente como hipótese de
engenharia, mas não deve ser apresentado como limiar validado antes da
análise de sensibilidade.

## Change Point Engine

O bloco de mudança de regime produz features como:

-   instante estimado da mudança $t^*$;

-   magnitude relativa $\Delta\mu$;

-   direção da mudança;

-   persistência do novo patamar;

-   tempo desde a mudança;

-   alteração da dispersão antes e depois do evento.

Essas features acrescentam informação que estatísticas agregadas de
janelas fixas não capturam adequadamente.

## Especialista elétrico - $S_{elec}$ {#especialista-eluxe9trico---s_elec}

O especialista elétrico opera sobre dados aprovados pelas regras de
qualidade e utiliza features de consumo, corrente, tensão e
comportamento temporal. A arquitetura privilegia modelos cuja função
possa ser explicada, como Isolation Forest para isolamento multivariado
ou PCA residual para erro de reconstrução.

O objetivo não é somar arbitrariamente diversos detectores. Cada score
precisa responder uma pergunta e possuir um plano de validação.

## Especialista comportamental - $S_{behavior}$

O comportamento diário é tratado como uma camada intermediária entre as
medições intervalares e as janelas históricas longas. O objetivo é
responder não apenas **quanto** uma UC consumiu, mas **como** esse
consumo se distribuiu ao longo do dia e com que recorrência esse formato
aparece.

A representação começa com dias de cobertura adequada. Cada curva é
decomposta em duas componentes: magnitude diária e forma normalizada.
Essa separação permite que dois consumidores de escalas distintas sejam
reconhecidos como semelhantes quanto à rotina intradiária, sem eliminar
a informação de consumo total.

Após regularização da cadência, as curvas normalizadas podem ser
reduzidas por PCA e agrupadas com K-Means ou método equivalente
[@geron2025]. Os centroides não recebem significado causal
automaticamente; são inicialmente **arquétipos matemáticos de dia**. A
interpretação posterior pode identificar, por exemplo, perfis
predominantemente diurnos, noturnos, constantes ou bimodais.

![Fluxo proposto para o especialista comportamental
diário.](figures/behavior.pdf){#fig:behavior width="95%"
height="0.75\\textheight"}

O diagrama comportamental distingue uma observação de um dia, o arquétipo atribuído à sua forma e a sequência desses arquétipos ao longo do tempo. A passagem de um cluster a outro é uma descrição de mudança de padrão; sua interpretação exige contexto de calendário, qualidade da série e persistência. Clusterização, isoladamente, não atribui causa.

### Recorrência e dias atípicos {#recorruxeancia-e-dias-atuxedpicos}

Um dia diferente não é, por si só, uma anomalia operacional. Uma UC pode
apresentar sistematicamente um sábado distinto dos dias úteis. O modelo
deve aprender a recorrência condicional ao calendário.

Defina $c_{u,d}$ como o arquétipo atribuído ao dia $d$ da UC $u$. Para
cada dia da semana $w$, pode-se estimar a distribuição histórica

$$q^{base}_{u,w}(c)=P(c_{u,d}=c\mid weekday(d)=w).$$

Um dia recebe score de atipicidade quando seu perfil tem baixa
compatibilidade com a distribuição histórica correspondente. A proporção
de dias atípicos em uma janela $W$ é

$$R_A(u,W)=\frac{1}{|W|}\sum_{d\in W}A_{u,d},$$

onde $A_{u,d}=1$ para dia comportamentalmente atípico e zero caso
contrário.

O sinal de interesse não é apenas $R_A$ elevado, mas sua mudança:

$$\Delta R_A=R_A(u,W_{recent})-R_A(u,W_{base}).$$

Também pode ser monitorada a mudança da distribuição de arquétipos entre
janelas, por exemplo por divergência de Jensen--Shannon. Dessa forma,
uma rotina semanal estável pode permanecer normal mesmo contendo dias
diferentes, enquanto o aparecimento de novos tipos de dia ou o aumento
persistente de sua frequência gera evidência de mudança comportamental.

### Interpretação operacional {#interpretauxe7uxe3o-operacional}

$S_{behavior}$ não deve diagnosticar férias, troca de morador, veículo
elétrico, geração ou fraude. Ele produz uma afirmação mais restrita: **a
rotina intradiária mudou de forma persistente**. A explicação é buscada
em cadastro, troca de titularidade, GD, topologia, alarmes e demais
especialistas.

Essa camada complementa o CPD. O CPD identifica mudanças de nível ou
distribuição ao longo do tempo; $S_{behavior}$ identifica mudanças na
composição dos tipos de dia, inclusive quando a energia diária total
permanece aproximadamente constante.

### Especialistas por alimentador

Os arquétipos poderão ser treinados globalmente e comparados com modelos
locais por alimentador. A hipótese é que alimentadores com composição de
carga distinta produzam espaços comportamentais também distintos. O uso
de especialista local só será mantido quando houver amostra suficiente e
ganho mensurável sobre o modelo global; caso contrário, utiliza-se a
referência global ou uma coorte agregada.

## Especialista de eventos físicos - $S_{tamper}$ {#especialista-de-eventos-fuxedsicos---s_tamper}

Alarmes de tampa, campo magnético, falha de fase, alteração de relógio e
troca de medidor são tratados como evidências contextuais. Eles não
devem, isoladamente, constituir target de fraude. Seu valor aumenta
quando coincidem temporalmente com mudança de regime ou anomalia
elétrica.

## Corroboração espacial e topológica {#corroborauxe7uxe3o-espacial-e-topoluxf3gica}

Uma alteração individual ganha significado diferente de uma alteração
compartilhada por várias UCs do mesmo transformador ou alimentador. O
pós-processamento deve comparar cada candidato com sua vizinhança
elétrica para distinguir fenômenos locais de eventos coletivos.

## Formalização da pontuação, fusão e roteamento

A expressão ``fusão dos especialistas'' descreve **operações diferentes** no software, que precisam ser separadas para reprodução: consenso entre detectores estatísticos elétricos, regras diagnósticas entre visões e combinação hierárquica entre modelos treinados em diferentes populações. Misturar esses três níveis em uma única expressão $S=\sum w_m S_m$ faria parecer que há um modelo unificado calibrado para fraude, o que não corresponde ao código disponível.

### Primeiro nível: consenso dos detectores elétricos

Na implementação de `src/models/composite.py` presente no snapshot de código inspecionado, a matriz numérica $X_{\mathrm{elec}}\in\mathbb R^{n\times p}$ passa por imputação mediana, `RobustScaler` e quatro detectores: Isolation Forest, LOF, ECOD e PCA residual. Os scores têm sinais e escalas diferentes. Para cada detector $m$, a rotina `_min_max_normalize` aplica, no conjunto pontuado, a transformação:

$$z_{i,m}=\frac{s_{i,m}-\min_j s_{j,m}}{\max_j s_{j,m}-\min_j s_{j,m}},\qquad z_{i,m}\in[0,1],$$

com tratamento numérico de amplitude nula no código. Para IF e LOF, o sinal bruto é invertido para que maior valor represente maior anomalia. O consenso da versão analisada é:

$$S^{(4)}_{i}=\operatorname{clip}_{[0,1]}\!\left(0{,}30z_{i,IF}+0{,}25z_{i,LOF}+0{,}25z_{i,ECOD}+0{,}20z_{i,PCA}+0{,}20R_i\right),$$

em que $R_i\in\{0,1\}$ indica ativação de uma das regras físicas críticas implementadas. Os limiares extremos que ativam $R_i$ são heurísticas computacionais de triagem presentes no código, não equivalem automaticamente às faixas regulatórias do PRODIST Módulo 8. O valor $0{,}20$ é **bônus aditivo**, e não peso de um quinto detector; portanto, não integra a soma unitária dos quatro pesos estatísticos.

A documentação técnica mais recente registra também uma configuração estendida com **seis detectores**, incorporando COPOD e HBOS, e pesos $(0{,}22;0{,}20;0{,}18;0{,}18;0{,}10;0{,}12)$, com bônus físico documentado de $0{,}15$. Essa configuração é aqui registrada como variante **documentada** $S^{(6)}$, não apresentada como execução verificada no mesmo snapshot Python utilizado para reconstituir $S^{(4)}$. Antes de comparar experimentalmente as variantes, devem ser congelados o commit de código e os pesos efetivamente executados. Tabelas e figuras do demonstrador sintético constituem outra execução, com seu próprio manifesto, e não identificam por si só o artefato implantado.

O corte binário do detector composto é calculado como quantil da distribuição de scores da própria coorte avaliada:

$$\tau=Q_{1-c}(\{S^{(4)}_j\}_{j=1}^{n}),\qquad D_i=\mathbb I(S^{(4)}_i\ge\tau),$$

com $c$ representando a contaminação nominal (por exemplo, $0{,}05$). Esse procedimento controla aproximadamente a proporção selecionada **na coorte pontuada** e não estima probabilidade de fraude. Empates e amostras pequenas podem alterar a fração efetiva. Como a normalização e o quantil dependem do lote, a prioridade de uma UC pode mudar se a população comparada mudar; por isso, a definição de uma política temporalmente estável requer referência e thresholds congelados após o ajuste no treinamento.

### Segundo nível: gate de confiabilidade e diagnóstico multivisão

O `ElectricalAnomalyDetector` calcula o score elétrico e marca sua confiabilidade por:

$$G_i=\mathbb I(q_i\ge g),\qquad g=0{,}70\ \text{na configuração padrão do snapshot},$$

em que $q_i$ é o indicador de qualidade disponibilizado ao módulo. A condição elétrica positiva utiliza $D_iG_i$. O **score bruto continua disponível** no objeto de retorno mesmo quando $G_i=0$; o código restringe a declaração de anomalia elétrica, não impede necessariamente a avaliação computacional do score. Por sua vez, o orquestrador hierárquico emprega `meta__is_clean_for_train` para escolher amostras de ajuste e quarentena de saída, com fallback quando há poucas observações limpas. Assim, existem controles de qualidade em **locais distintos** do programa que não devem ser confundidos.

O `CommunicationAnomalyDetector` utiliza Isolation Forest e ECOD, combinando os dois scores normalizados como:

$$S_{comm,i}=0{,}5z_{i,IF}^{comm}+0{,}5z_{i,ECOD}^{comm}.$$

O especialista de eventos físicos aplica saturação da contagem de alarmes $a_i$, penalidade da inclinação negativa $b_i$ e contexto de substituição $m_i\in\{0,1\}$:

$$S_{tamper,i}=\operatorname{clip}_{[0,1]}\!\left[0{,}70\min(a_i/3,1)+0{,}30\operatorname{clip}_{[0,1]}(-b_i/5)+0{,}15\mathbb I(a_i>0)m_i\right].$$

As unidades da inclinação e os pesos devem corresponder ao cálculo de features efetivamente executado; os valores são heurísticos de projeto, não coeficientes estimados de uma distribuição de irregularidades confirmadas. O `BehavioralClusterer` disponível ajusta PCA e K-Means e retorna **rótulos de cluster**. Portanto, nessa implementação, não existe um $S_{behavior}$ probabilístico único que possa ser diretamente somado às demais visões. A recorrência de dias atípicos amplia a interpretação comportamental como representação temporal e não deve ser descrita como score final já calibrado sem uma função e validação próprias.

A integração `MultiViewAnomalyPipeline` constrói um **vetor de diagnóstico**:

$$\mathbf e_i=(S_{comm,i},S_{elec,i},S_{tamper,i},G_i,C_i,R_i),$$

em que $C_i$ é o identificador do cluster comportamental. Em seguida aplica uma árvore de regras com precedência explícita. A sequência implementada prioriza: (1) $S_{tamper}\ge0{,}70$; (2) não confiável com $S_{comm}\ge0{,}70$; (3) não confiável; (4) confiável com $S_{elec}\ge0{,}70$; (5) comunicação anômala; (6) demais casos. Algumas *strings* legadas usam palavras como `CONFIRMADA` ou `OPERACAO_REGULAR`; neste trabalho, elas são tratadas como **códigos internos de roteamento**, jamais como confirmação de fraude, de ausência de irregularidade ou de conformidade regulatória.

Essa árvore **não produz um escore contínuo global único** para todas as visões. Caso a carteira necessite de uma prioridade escalar posterior, deve-se especificar versão da função operacional, critérios de elegibilidade, tratamento de valores ausentes, calibração dos scores e estratégia de desempate. A definição dessa política de fila não pode ser retroativamente atribuída ao orquestrador multivisão como se já fosse parte de seu treinamento.

### Terceiro nível: MoE global versus alimentador

Em `src/models/moe_router.py`, o roteamento hierárquico é diferente da fusão entre $S_{comm}$ e $S_{elec}$. O primeiro ajusta um detector composto global e detectores compostos por alimentador para coortes com pelo menos 30 registros; o segundo calcula pesos de referência de acordo com o tamanho do alimentador. As regras de referência do código são:

| Registros no alimentador $n_f$ | $w_f$ | $w_r$ | $w_g$ |
|:--|--:|--:|--:|
| $n_f<150$ | 0,15 | 0,45 | 0,40 |
| $150\le n_f<1000$ | interpolação linear de 0,35 a 0,70 | interpolação de 0,40 a 0,20 | interpolação de 0,25 a 0,10 |
| $n_f\ge1000$ | 0,70 | 0,20 | 0,10 |

Aqui $f$, $r$ e $g$ representam alimentador, regional/municipal e global. Havendo penetração de GD no perfil do circuito superior ao limiar configurado, a heurística amplia o peso local, respeitando limites e renormalização. Na rotina de treino inspecionada, a chamada de combinação usa scores de **alimentador e global**, mas passa o score municipal como `None`; nesse caso, o operador renormaliza apenas os pesos dos especialistas disponíveis:

$$S^{MoE}_i=\frac{\sum_{m\in\mathcal A_i}w_mS_{i,m}}{\sum_{m\in\mathcal A_i}w_m},\qquad \mathcal A_i\subseteq\{f,r,g\}.$$

Exemplo: com $n_f\ge1000$ e apenas alimentador/global disponíveis, os pesos nominais $(0{,}70,0{,}20,0{,}10)$ tornam-se efetivamente $0{,}875$ local e $0{,}125$ global. O limiar de decisão pode ser obtido por quantis locais sobre registros elegíveis. Isso mostra que o componente ``Mixture of Experts'' atualmente implementado é um **roteador hierárquico heurístico de modelos**, não uma rede neural com *gating network* treinada por gradiente.

### Exemplo numérico e análise de propagação de evidências

Considere uma UC hipotética com quatro scores elétricos normalizados, nesta ordem: $0{,}80$, $0{,}60$, $0{,}30$ e $0{,}40$. Seu consenso de quatro detectores é $0{,}545$. Se uma regra física crítica é ativada, o bônus eleva a pontuação a $0{,}745$. Se, entretanto, a qualidade vale $q=0{,}65$, a declaração elétrica permanece **não confiável** pelo gate de $0{,}70$. O score existe, mas a fila deve indicar insuficiência de evidência elétrica.

Se houver dois alarmes, inclinação $b=-2$ nas unidades de entrada e troca recente de medidor, o score heurístico de eventos será aproximadamente $0{,}737$. Pela precedência literal das regras do código, isso pode levar à categoria interna de investigação de tamper antes do teste de qualidade. Essa decisão deve ser mostrada ao operador como **sinal físico a investigar com ressalva de telemetria**, não como prova de manipulação. O exemplo evidencia por que explicitar precedência, pesos e estado de confiabilidade é mais importante do que relatar apenas o valor final.

### Sensibilidade, estabilidade e critérios de reprodutibilidade

A configuração descrita é reprodutível apenas quando também se registram: o conjunto usado no `fit`, os hiperparâmetros, a versão de cada detector, o pré-processamento, o esquema de scores, a coorte de normalização, a política de threshold e as regras físicas executadas. Como o `fit_and_score` multivisão executa ajustes e pontuação no mesmo conjunto passado pelo chamador, sua execução descritiva não constitui, isoladamente, teste de generalização fora da amostra. A avaliação comparativa deve preservar separação temporal e parâmetros treinados congelados.

A discussão acima satisfaz a necessidade de formalização matemática em três níveis sem atribuir ao sistema uma função de fusão que não exista. A validação de pesos e limites requer testes de ablação e, para conclusões sobre furto, rótulos externos de inspeção.

# Demonstração experimental das técnicas de aprendizado de máquina {#demonstracao-experimental-das-tecnicas-de-ml}

Para tornar explícito o papel de cada técnica utilizada no pipeline, foi
desenvolvido um experimento reproduzível no qual os mesmos dados são
processados por métodos com geometrias distintas. O objetivo não é
declarar um algoritmo universalmente superior, mas verificar a hipótese
de que técnicas diferentes respondem a perguntas diferentes e, por isso,
podem produzir evidências complementares.

O experimento possui duas frentes. A primeira utiliza uma população
sintética controlada com 1.000 curvas diárias de 96 pontos,
correspondentes a intervalos de 15 minutos, incluindo perfis
residenciais noturnos, comerciais diurnos, cargas contínuas, geração
distribuída e diferentes tipos de outlier. Nesse conjunto, a categoria
geradora de cada caso é conhecida e funciona como *ground truth*
didático. A segunda frente utiliza os estudos de caso reais mascarados
apresentados no capítulo seguinte, sem assumir rótulo de fraude.

> **Mesmos dados $\rightarrow$ técnicas diferentes $\rightarrow$
> perguntas diferentes $\rightarrow$ efeitos diferentes $\rightarrow$
> contribuição complementar.**

## K-Means: construção de populações comportamentalmente comparáveis {#demo-kmeans}

K-Means responde à pergunta: **quais curvas possuem morfologia
semelhante?** A entrada é a curva diária normalizada, de modo que o
agrupamento privilegie a forma do consumo em vez de apenas sua
magnitude. O método não detecta fraude; ele constrói arquétipos que
permitem comparar uma UC com pares de comportamento semelhante.

![Perfis diários crus e normalizados no experimento controlado. A
normalização expõe diferenças de forma que a energia agregada pode
ocultar.](figures/ml_method_demo/01_raw_profiles.png){#fig:ml-raw-profiles
width="98%" height="0.80\\textheight"}

A primeira leitura dessa figura deve comparar dois painéis: as curvas brutas conservam diferenças de magnitude, enquanto as curvas normalizadas enfatizam distribuição intradiária. A normalização melhora a comparação de forma, mas elimina informação útil sobre escala; por isso o dataset mantém separadamente energia diária e perfil relativo. O exemplo demonstra informação perdida pela média agregada, não irregularidade confirmada.

![Arquétipos obtidos por K-Means na população sintética controlada. Os
centróides representam referências comportamentais, não classes
causais.](figures/ml_method_demo/02_kmeans_profiles.png){#fig:ml-kmeans
width="98%" height="0.80\\textheight"}

Os centróides sintetizam formas recorrentes da população controlada. Curvas afastadas do próprio centróide podem ser atípicas dentro do grupo, mas o valor desse afastamento depende de dispersão e tamanho da coorte. A figura evidencia que a segmentação cria referências comparáveis; não sustenta converter nomes interpretativos de cluster em classes de fraude.

No experimento, quatro famílias principais aparecem separadas por
morfologia. Essa representação oferece a base para o especialista
comportamental e para comparações locais: um desvio passa a ser
interpretado em relação à rotina esperada do grupo, e não apenas em
relação à cidade inteira.

## PCA e PCA residual: estrutura dominante e deformação da curva {#demo-pca}

A Análise de Componentes Principais (PCA) responde a duas perguntas
sucessivas: **quais combinações lineares explicam a maior parte da
variação?** e **quanto de cada curva deixa de ser explicado por esse
subespaço?** A primeira pergunta produz uma representação compacta; a
segunda origina o PCA residual.

![Projeção, variância explicada e reconstrução por PCA no benchmark
controlado.](figures/ml_method_demo/03_pca_projection.png){#fig:ml-pca
width="98%" height="0.80\\textheight"}

A projeção em duas componentes facilita visualizar proximidades, mas não preserva necessariamente toda a distância no espaço original. A curva de variância acumulada explicita a fração da variação representada e ajuda a decidir quantas componentes manter. A reconstrução no terceiro painel permite observar a diferença entre compressão útil e perda de informação relevante.

Para uma observação $x_i$ e sua reconstrução $\hat{x}_i$, utiliza-se o
erro

$$E_i=\lVert x_i-\hat{x}_i\rVert_2.$$

Uma curva pode manter energia total semelhante ao padrão e, ainda assim,
apresentar erro elevado se sua morfologia não pertence à estrutura
dominante.

![PCA residual: comparação entre uma curva bem reconstruída e uma
anomalia de
morfologia.](figures/ml_method_demo/04_pca_residual.png){#fig:ml-pca-residual
width="98%" height="0.80\\textheight"}

A comparação entre curva original e reconstrução mostra em quais horários aparece maior discrepância. Um resíduo elevado indica que a morfologia não foi bem representada pelo subespaço ajustado; não especifica se a causa é fraude, nova rotina ou um evento válido. A interpretação depende de a PCA ter sido treinada em uma referência temporalmente apropriada.

## Isolation Forest: raridade global {#demo-iforest}

Isolation Forest procura observações globalmente fáceis de separar por
partições aleatórias [@liu2008]. Casos extremos tendem a apresentar
caminhos médios de isolamento menores. Essa propriedade o torna útil
para triagem multidimensional, mas também cria um ponto cego importante:
um pequeno agrupamento de observações anômalas semelhantes entre si pode
deixar de parecer isolado.

![Isolation Forest no benchmark controlado: raridade global e exemplo do
ponto cego produzido por um microagrupamento
compacto.](figures/ml_method_demo/05_isolation_forest.png){#fig:ml-iforest
width="98%" height="0.80\\textheight"}

O mapa de score deve ser lido pela facilidade de isolamento dos pontos. Observações muito afastadas do conjunto tendem a maior anomalia; já um pequeno agrupamento consistente pode alterar as profundidades de isolamento. A figura torna visível um mecanismo que não pode ser resumido por um limiar sobre consumo total.

## LOF: raridade local {#demo-lof}

LOF responde a uma pergunta diferente: **esta observação possui
densidade incompatível com a de seus vizinhos?** [@breunig2000]. Uma UC
pode ser pouco extrema globalmente e, ao mesmo tempo, muito incomum
quando comparada apenas com consumidores de comportamento ou topologia
semelhantes.

![Comparação entre LOF e Isolation Forest para casos globais, locais e
microagrupamentos no conjunto
controlado.](figures/ml_method_demo/06_lof.png){#fig:ml-lof width="98%"
height="0.80\\textheight"}

A divergência entre LOF e Isolation Forest aparece nas regiões onde a densidade dos vizinhos difere da população global. O exemplo não estabelece que LOF seja superior em qualquer alimentador: mostra que uma referência de proximidade pode recuperar situações não destacadas pelo isolamento global, desde que a métrica e a população de comparação sejam adequadas.

Essa propriedade justifica avaliar LOF em coortes reduzidas, por exemplo
por alimentador ou perfil comportamental, em vez de assumir que a mesma
vizinhança estatística representa toda a área de estudo.

## ECOD e detectores de cauda {#demo-ecod}

ECOD mede extremidade por funções de distribuição empíricas e não exige
hipótese gaussiana [@li2022ecod]. Sua contribuição é particularmente
clara quando uma UC não é extrema em consumo total, mas ocupa
simultaneamente regiões de cauda em medidas como coeficiente de
variação, fator de carga ou razão dia/noite.

![ECOD: posicionamento de uma observação nas caudas empíricas de
diferentes features.](figures/ml_method_demo/07_ecod.png){#fig:ml-ecod
width="98%" height="0.80\\textheight"}

As curvas de distribuição mostram a posição relativa de um caso em cada característica. A possibilidade de energia diária pouco extrema coexistir com medidas em cauda justifica conservar features de variabilidade e forma. Como ECOD considera extremidade marginal, dependências entre variáveis precisam ser examinadas por outros métodos; cauda não equivale a mecanismo físico.

Na variante estendida documentada, ECOD é acompanhado por COPOD e HBOS; o snapshot executável examinado mantém quatro detectores. COPOD
acrescenta uma leitura baseada em dependência de caudas [@li2020copod];
HBOS fornece uma aproximação por histogramas independentes
[@goldstein2012hbos]. Esses métodos não são tratados como equivalentes:
sua permanência no ensemble deve ser sustentada pela contribuição
marginal observada em ablações.

## Comparação direta e ensemble composto {#demo-ensemble}

A Figura [figura](#fig:ml-comparison) apresenta os scores normalizados dos
detectores para diferentes casos controlados. O padrão de resposta não é
idêntico entre os métodos. No CSV gerado pelo experimento, a
sobreposição do Top-100 de cada detector com a carteira do ensemble
variou de 33% a 72%, evidenciando que os métodos não estão simplesmente
reproduzindo a mesma ordenação.

![Matriz de respostas dos detectores e decomposição do score composto no
experimento
controlado.](figures/ml_method_demo/08_detector_comparison.png){#fig:ml-comparison
width="98%" height="0.80\\textheight"}

A matriz permite comparar a resposta dos métodos ao **mesmo conjunto de exemplos**; as barras tornam rastreável a contribuição dos scores após a transformação de escala. A diferença de ordenação indica complementaridade potencial e também risco de incorporar detectores redundantes. O efeito útil deve ser estimado por ablação, não deduzido automaticamente de divergência visual.

A fusão utiliza scores colocados em escala comparável por calibração
empírica. De forma genérica,

$$S_{comp}(i)=\sum_m w_m z_{i,m},$$

em que $z_{i,m}$ é o score calibrado do método $m$ para a observação $i$
e $w_m$ é seu peso. Os pesos atualmente implementados são parâmetros de
projeto e não devem ser interpretados como probabilidades ou importância
causal.

## Ablation funnel e contextualização por alimentador {#demo-ablation}

O experimento também produz uma ablação controlada $M_0\rightarrow M_5$.
O funil ilustra o efeito de adicionar saneamento cadastral, gating de
comunicação, contextualização, persistência e ensemble. Os números da
Figura [figura](#fig:ml-ablation) pertencem ao benchmark sintético e não
devem ser confundidos com contagens finais de Araucária; sua finalidade
é demonstrar como cada camada modifica a população elegível.

![Ablação operacional controlada e exemplo de alteração do ranking
quando o contexto do alimentador é
introduzido.](figures/ml_method_demo/09_ablation_funnel.png){#fig:ml-ablation
width="98%" height="0.80\\textheight"}

O funil do experimento controlado ilustra a consequência de aplicar filtros e contextualização sobre a quantidade de candidatos. Como as etapas podem se sobrepor, a redução marginal precisa ser calculada sobre conjuntos de UCs, evitando somar exclusões duplicadas. O painel de alimentadores mostra mudança de referência estatística; sua interpretação exige normalizar por tamanho da coorte e comparar estabilidade de ranking.

O segundo painel mostra que uma UC pode ter baixa prioridade no ranking
global e alta prioridade dentro do seu alimentador. Essa comparação será
reproduzida na base real por meio de ranking global, ranking local e
sobreposição Top-$K$.

## Casos arquetípicos e relação com CPD {#demo-cases}

O painel final reúne seis situações: comportamento normal, outlier
global, outlier local, extremidade marginal, anomalia de morfologia e
quebra temporal. O objetivo é tornar explícito que não existe um único
score capaz de resumir todos esses fenômenos sem perda de significado.

![Seis casos arquetípicos usados para demonstrar a especialização dos
detectores e a complementaridade entre anomalia estática e mudança de
regime.](figures/ml_method_demo/10_case_studies.png){#fig:ml-cases
width="98%" height="0.80\\textheight"}

Os seis arquétipos exibem falhas de decisão diferentes: extremo global, desvio local, cauda marginal, deformação estrutural e mudança de regime. A utilidade do painel está em mostrar quais algoritmos são informativos em cada situação conhecida. Os casos são gerados artificialmente; etiquetas do experimento não podem ser transferidas como verdade de campo para UCs reais.

O caso temporal reforça a distinção entre detecção de raridade e CPD. Um
detector de outlier responde se um estado é incomum; CUSUM ou outro
método de change point procura o instante em que o regime mudou. No
pipeline, ambas as evidências podem coexistir sem serem tratadas como a
mesma variável.

## Custo computacional do benchmark {#demo-runtime}

O script registra tempo de ajuste e escoragem para 1.000 curvas
sintéticas. Os valores medidos na execução versionada são apresentados
na Tabela [tabela](#tab:runtime-demo). Eles servem para comparar ordens de
grandeza no ambiente do experimento, não como benchmark universal de
produção.

::: {#tab:runtime-demo}
  Método                Tempo medido (ms)
  ------------------- -------------------
  K-Means                          441,40
  PCA residual                      37,03
  Isolation Forest                1931,24
  LOF                               14,29
  ECOD                               1,39
  Ensemble composto               1983,95

  : Tempo medido no experimento demonstrativo com 1.000 curvas.
:::

A tabela evidencia uma diferença de escala de execução no ambiente controlado. O maior custo do IF nesta configuração não demonstra que ele seja intrinsecamente mais lento em toda arquitetura: paralelização, número de árvores e implementação alteram o tempo. A conclusão operacional válida é a necessidade de medir custo por coorte, por ajuste e por pontuação em condições equivalentes, e não estimar capacidade de produção por extrapolação de 1.000 curvas.

A leitura desses valores deve considerar que implementações, número de
árvores, vizinhos, paralelização e tamanho da coorte alteram
substancialmente o custo. No ambiente demonstrativo, ECOD e LOF foram
leves; Isolation Forest dominou o tempo do ensemble. A extrapolação para
carga operacional requer profiling por tamanho de coorte e registro de
memória de processo; os tempos aqui informados caracterizam
exclusivamente o experimento controlado.

## O que o experimento demonstra e o que não demonstra {#demo-limits}

O benchmark controlado demonstra o **mecanismo** de cada técnica, seus
pontos cegos e a não equivalência dos rankings. Ele também fornece
figuras reproduzíveis para explicar a arquitetura perante banca e
operação. Entretanto, não estima a taxa de acerto em fraude, não
substitui inspeção de campo e não permite concluir que determinado peso
de ensemble é ótimo para a população real. Essas perguntas pertencem ao
estudo prospectivo com ground truth operacional.

# Sensibilidade do gating e consequências para a seleção

A cobertura mínima não pode ser escolhida apenas por conveniência numérica. Elevar o limiar $g$ reduz a exposição do detector elétrico a janelas degradadas, mas também reduz o conjunto de consumidores analisáveis e pode excluir casos realmente relevantes. Seja $E_g=\{i:q_i\ge g\}$ e seja $P$ o conjunto de positivos independentes de referência. Dois indicadores distintos são:

$$C(g)=\frac{|E_g|}{N},\qquad R_{\max}(g)=\frac{|P\cap E_g|}{|P|}.$$

$C(g)$ mede elegibilidade da população e $R_{\max}(g)$ é o teto de recuperação possível após o gate, mesmo com detector perfeito entre os elegíveis. Para uma carteira Top-$K$, o indicador controlado de recuperação é $R_K(g)=|P\cap\mathrm{TopK}(E_g)|/|P|$. O risco relevante é que elevar $g$ pode reduzir falsos alertas de telecomunicação ao mesmo tempo que limita $R_K$. Sem inspeções independentes, nenhum desses indicadores deve ser denominado *recall de fraude real*.

## Ensaio controlado reproduzível

Para tornar o compromisso visível, foi executado um experimento de **sensibilidade sintética** sobre os 1.000 exemplos do demonstrador de ML. Os oito casos com rótulo de anomalia **injetada** foram mantidos como referência exclusivamente didática. Atribuiu-se uma variável de qualidade artificial com gerador pseudoaleatório fixado em 20261118: distribuição Beta(8,2) nos demais casos e Beta(7,3) nos casos injetados. Esses parâmetros são hipóteses do exercício e **não descrevem a distribuição de telemetria da concessionária**. O ranking base foi mantido a partir de `composite_score`, selecionando-se até 50 UCs elegíveis em cada limiar.

![Sensibilidade do limiar de qualidade no exercício sintético: a seletividade reduz a população elegível e pode excluir anomalias previamente injetadas.](figures/gating_sensitivity_synthetic.png){#fig:gating-synthetic width="94%" height="0.72\\textheight"}

A Figura [sensibilidade do gating](#fig:gating-synthetic) mostra três efeitos distintos. A curva de elegibilidade mede perda de cobertura analítica, a curva de retenção estabelece um limite máximo de casos positivos que ainda podem ser detectados e a terceira mostra quantos casos injetados permanecem no Top-50 após a exclusão. O exemplo produz perda acentuada a partir de $g=0{,}70$ porque a variável de qualidade foi artificialmente associada a distribuições diferentes; por essa razão, **não autoriza concluir que 70% esteja inadequado nos dados reais**, mas evidencia matematicamente que o efeito depende da associação entre qualidade e evento de interesse.

| Limiar $g$ | UCs elegíveis | Anomalias injetadas retidas / 8 | Anomalias injetadas no Top-50 / 8 |
|:--|--:|--:|--:|
| 0,40 | 1000 | 8 | 6 |
| 0,50 | 985 | 6 | 4 |
| 0,60 | 940 | 6 | 4 |
| 0,70 | 806 | 2 | 1 |
| 0,80 | 563 | 0 | 0 |
| 0,90 | 200 | 0 | 0 |

A tabela permite conferir os dados subjacentes ao gráfico e mostra que um limiar não deve ser avaliado isoladamente pela taxa de descarte. O número de oito anomalias controladas é pequeno; a figura tem função **metodológica e ilustrativa**, não fornece intervalo de confiança robusto nem estimativa da eficácia operacional.

## Protocolo para avaliação na população real

O estudo de sensibilidade na base AMI deverá aplicar um conjunto fixo de cutoffs, características e parâmetros treinados; variar somente $g$ em uma grade predefinida; registrar contagens elegíveis por fabricante, canal, alimentador, classe e histórico de instalação; e comparar interseção e estabilidade das carteiras Top-$K$. Casos de telemetria degradada não devem ser assumidos negativos, mas encaminhados à categoria específica de qualidade. Quando houver resultados independentes de campo, a análise poderá estimar perda de positivos por gate, rendimento@K com e sem roteamento e incerteza por bootstrap estratificado.

Essa distinção entre análise controlada e validação operacional evita tanto ignorar o risco do gate quanto escolher limiares por desempenho aparente sobre rótulos produzidos pelo próprio modelo.

# Saídas como padrões de compatibilidade {#sauxeddas-como-padruxf5es-de-compatibilidade}

A saída analítica evita diagnósticos causais não sustentados. Em vez de
emitir "fraude confirmada", o sistema utiliza classes como:

-   padrão compatível com redução abrupta de carga;

-   padrão compatível com geração local;

-   padrão compatível com problema de medição;

-   padrão compatível com degradação de telemetria;

-   comportamento estatisticamente atípico;

-   comportamento corroborado ou não corroborado pela vizinhança.

A qualificação operacional ocorre por cruzamento de evidências.

Exemplo:

    queda persistente de FA
    + tensão estável
    + boa cobertura
    + ausência de mudança coletiva no transformador
    + alarme físico próximo ao t*
            -> prioridade elevada para análise

Essa arquitetura human-in-the-loop é coerente com a necessidade de
tornar o resultado acionável sem substituir a decisão de campo.

# Estudo de caso e anatomia da detecção {#estudo-de-caso-e-anatomia-da-detecuxe7uxe3o}

Este capítulo utiliza exemplos reais mascarados para mostrar como as
diferentes representações do pipeline alteram a interpretação de uma UC.
Os casos foram selecionados automaticamente a partir do Data Lake e têm
finalidade **didática e metodológica**. Eles demonstram mecanismos que o
sistema consegue observar, mas não constituem, isoladamente, comprovação
de fraude nem resultado populacional final.

## Ato I - Mesma energia, comportamentos diferentes

A primeira comparação evidencia uma limitação de qualquer método baseado
apenas em consumo agregado. As UCs mascaradas `934***50` e `116****57`
apresentam medianas diárias próximas, respectivamente 20,38 e 19,22
kWh/dia. Entretanto, a dispersão e a forma horária são profundamente
distintas.

![Comparação de duas UCs com consumo diário mediano semelhante e
morfologias de carga distintas. Identificadores
mascarados.](figures/eda/comparacao_duas_ucs_mesmo_consumo.png){#fig:comparacao-ucs-mesmo-consumo
width="98%"}

A semelhança de mediana diária não implica semelhança intradiária. A figura confronta duas UCs mascaradas com escalas agregadas próximas e curvas de forma diferente, sustentando o uso de razão dia/noite, fator de carga, dispersão e agrupamento por morfologia. O estudo de caso não determina qual UC é irregular; indica que um detector baseado apenas em soma diária poderia perder essa distinção.

Na UC estável, o MAD diário é próximo de zero e a carga horária
permanece praticamente plana. Na UC volátil, o MAD é elevado e a curva
apresenta forte concentração diurna. O exemplo mostra que uma
estatística de magnitude não é suficiente para caracterizar o
consumidor. A arquitetura precisa preservar simultaneamente **quanto a
UC consome**, **como distribui esse consumo ao longo do dia** e **quão
estável é esse padrão no tempo**.

Esse resultado sustenta duas decisões metodológicas. A primeira é
utilizar medidas robustas de dispersão e tendência em conjunto com a
mediana. A segunda é manter um especialista comportamental separado, no
qual curvas de 24 horas são normalizadas e agrupadas por morfologia.
Dessa forma, uma UC pode ser comparada com dias ou consumidores de
comportamento semelhante, em vez de ser julgada somente pela escala
absoluta de consumo.

## Ato II - A dimensão semanal revela a rotina {#ato-ii---a-dimensuxe3o-semanal-revela-a-rotina}

A Figura [figura](#fig:comparacao-semanal) acompanha o mesmo par durante 168
horas. A diferença entre as UCs deixa de ser apenas estatística e se
torna temporalmente evidente.

![Acompanhamento semanal do par de UCs. A série azul mantém carga
aproximadamente constante; a série laranja apresenta regime diurno,
picos e variação
intra-semanal.](figures/eda/acompanhamento_semanal_comparacao_par.png){#fig:comparacao-semanal
width="98%"}

Ao estender a comparação para vários dias, a figura permite examinar repetição, estabilidade e alternância de comportamento. O padrão semanal pode ser uma propriedade legítima da rotina, enquanto uma mudança persistente da distribuição dos tipos de dia merece investigação. Essa leitura fundamenta o especialista comportamental e a separação entre atipicidade pontual e mudança de regime.

A UC de carga plana preserva praticamente o mesmo nível durante toda a
semana. A outra UC repete um ciclo diurno, mas com amplitudes diferentes
entre os dias. Para o modelo comportamental, o ponto relevante não é
simplesmente que existe variabilidade. É a **regularidade dessa
variabilidade**.

Se determinado consumidor apresenta historicamente um dia atípico sempre
no mesmo ponto da semana, essa recorrência pode fazer parte de sua
rotina. O sinal de interesse surge quando a frequência, a posição ou a
forma desses dias muda. Assim, a sequência de clusters diários constitui
uma nova série temporal discreta, sobre a qual podem ser calculadas
persistência, transições e mudanças de distribuição.

Esse mecanismo também evita interpretar férias, mudança de ocupação,
trabalho remoto ou alteração de atividade como fraude automática. Esses
eventos são hipóteses compatíveis com mudança de rotina. O sistema
detecta a transição e a descreve; outras camadas determinam se existem
evidências adicionais que justifiquem elevar a prioridade operacional.

## Ato III - Da anomalia multivariada ao Raio-X da UC

A terceira figura mostra uma UC mascarada selecionada pelo pipeline com
score de anomalia elevado e categoria operacional compatível com
desequilíbrio de carga ou *bypass*. A série superior apresenta o consumo
real juntamente com baseline horário e banda robusta; o painel inferior
apresenta a visão de disponibilidade de telemetria utilizada durante a
investigação.

![Exemplo real mascarado de UC destacada pelo pipeline. O gráfico
combina série de consumo, baseline robusto e diagnóstico de
disponibilidade de telemetria para apoiar interpretação do
score.](figures/eda/acompanhamento_semanal_outlier.png){#fig:outlier-real
width="98%"}

O Raio-X da UC superpõe medição observada, referência robusta e marcações de anomalia. A presença de desvios sustentados em uma janela pode justificar prioridade, mas a interpretação exige cobertura adequada e consulta ao histórico de instalação, alarmes e vizinhança. A designação de bypass é categoria investigativa compatível com hipóteses concorrentes; a figura, por si, não comprova manipulação de medidor.

No snapshot utilizado para gerar o exemplo, a UC `685***20` apresentava
mediana histórica de 81,08 kWh/dia, MAD de 24,46 kWh/dia, tendência
negativa de aproximadamente -1,37 kWh/dia e indicador elevado de
desequilíbrio de corrente. O ensemble produziu score 0,621. Esses
valores explicam por que a UC é rara no espaço multivariado, mas **não
provam a causa física da irregularidade**.

A figura evidencia o papel da arquitetura multivisão: o score deve ser
interpretado ao lado da qualidade da série, do comportamento temporal e
das grandezas elétricas. Mesmo quando o classificador pós-processa o
caso como `DESEQUILIBRIO_CARGA_OU_BYPASS`, essa categoria deve ser lida
como um padrão compatível que orienta a investigação. A confirmação
permanece dependente de contexto cadastral, topologia, alarmes e
eventual inspeção de campo.

Há ainda uma diferença importante entre o exemplo visual e a avaliação
populacional. A figura representa uma UC concreta e ajuda a verificar se
a cadeia de evidências faz sentido. Ela não informa, por si só, quantas
UCs semelhantes existem nem qual a taxa de acerto do classificador. A
extrapolação de ocorrências individuais para prevalência ou acurácia
populacional requer denominadores e resultados de campo independentes,
não fornecidos pelo exemplo.

## Da abordagem original à arquitetura atual no mesmo caso {#da-abordagem-original-uxe0-arquitetura-atual-no-mesmo-caso}

O estudo de caso permite enxergar concretamente a diferença entre as
duas formulações. Na abordagem original, a UC seria representada por um
vetor agregado e ordenada pelo score global. Na arquitetura refinada, o
mesmo candidato percorre etapas adicionais:

    score global
        |
        +--> a série possui qualidade suficiente?
        |
        +--> houve mudança temporal ou apenas variabilidade recorrente?
        |
        +--> o perfil diário mudou?
        |
        +--> as grandezas elétricas são raras entre pares adequados?
        |
        +--> existem alarmes físicos ou explicações determinísticas?
        |
        +--> outras UCs do mesmo transformador/alimentador mudaram juntas?
        |
        v
    prioridade e explicação operacional

A vantagem esperada não é necessariamente produzir um score maior. É
transformar um alerta genérico em um caso cuja origem estatística,
qualidade dos dados e evidências de contexto possam ser auditadas.

## Relação com o Cockpit Web {#relauxe7uxe3o-com-o-cockpit-web}

As figuras estáticas reproduzem a mesma lógica já utilizada no protótipo
do Cockpit Web. A navegação por candidato parte da carteira geral e
chega ao "Raio-X" da UC, consultando a série real sob demanda e
apresentando baseline, cobertura, scores, categoria e persistência. A
TUI permanece responsável pela operação do pipeline, enquanto o Cockpit
funciona como interface de investigação dos resultados.

    TUI -> extrair, auditar, treinar e versionar
    Modelos -> produzir evidências e ranking
    Web -> investigar candidatos e apoiar decisão

Essa separação de responsabilidades reforça que a aplicação não é apenas
um conjunto de scripts de Machine Learning, mas uma plataforma completa
de engenharia, análise e apoio operacional.

# Aplicação operacional e TUI {#aplicauxe7uxe3o-operacional-e-tui}

A aplicação constitui parte do produto do TCC e não apenas uma interface
auxiliar. Seu papel é tornar executável, auditável e reproduzível a
sequência completa de extração, validação, treinamento e comparação dos
modelos. A interface atual é uma TUI baseada em Rich, executada em
ambiente Python gerenciado por `uv`.

## Filosofia da TUI

A TUI mantém a lógica de negócio nos módulos Python e utiliza a
interface apenas como orquestrador. Isso preserva duas formas de
execução: interativa, para engenharia e pesquisa, e CLI, para automação
ou depuração. A mesma operação disparada pelo menu pode ser reproduzida
por comando de terminal.

O menu principal agrupa seis funções operacionais:

    [1] Gerar Dataset
    [2] Treinar & Validar Modelos
    [3] Visualizar Estatísticas do Data Lake
    [4] Comparar Especialistas vs Modelo Global
    [5] Executar Auditoria de Saúde & Tests
    [6] Configurações Globais

## Geração do dataset {#gerauxe7uxe3o-do-dataset}

A opção de geração de dataset oferece quatro modos: cidade completa em
um dia de referência, alimentador específico, período histórico e
sincronização inteligente. Em todos eles o operador escolhe o formato de
saída (`parquet`, `csv` ou ambos) e acompanha a fila de lotes do MDM.

No modo de sincronização inteligente, a aplicação consulta a cobertura
já existente no Data Lake e determina automaticamente quais dias
precisam ser extraídos. Se houver lacuna interna, ela também entra na
fila. Esse comportamento transforma a extração em processo idempotente:
repetir o comando não significa repetir o trabalho concluído.

Durante a execução, cada *worker* apresenta estado individual -
aguardando, consultando Oracle, recebendo registros, normalizando ou
gravando - além da barra de progresso global. O desenho favorece
diagnóstico de gargalos e evita que uma execução longa pareça um
processo opaco.

## Estatísticas e calendário do Data Lake {#estatuxedsticas-e-calenduxe1rio-do-data-lake}

A opção de estatísticas apresenta dimensões das camadas, distribuição
por alimentador e um calendário mensal de continuidade. O usuário pode
auditar ambos os formatos ou selecionar apenas Parquet ou CSV. Dias
completos, parciais e ausentes são distinguidos visualmente.

A mesma área permite sincronizar formatos faltantes. Uma partição
disponível apenas em Parquet pode ser convertida para CSV e vice-versa
sem reconsultar o Oracle. Essa funcionalidade reduz custo de banco e
separa claramente **aquisição** de **formatação para consumo**.

## Auditoria de saúde e testes {#auditoria-de-sauxfade-e-testes}

A auditoria executa validações estruturais e a suíte de testes do
projeto. O relatório diário registra a condição do Data Lake e deve
atuar como *gate* anterior ao treinamento. Um modelo não deve ser
treinado simplesmente porque arquivos existem; as camadas precisam
satisfazer os contratos mínimos esperados.

Entre os controles relevantes estão: schemas, chaves, presença das
partições, consistência de datas, cardinalidade dos joins, histórico
UC--NIO e disponibilidade das features exigidas. Esse estágio reduz o
risco de treinar modelos sobre datasets silenciosamente degenerados.

## Ordem de treinamento

A TUI organiza o treinamento em níveis. A ordem representa dependências
metodológicas e permite comparação controlada, sem pressupor que todas
as técnicas tenham contribuição marginal positiva.

![Ordem operacional de treinamento e
governança.](figures/training_order.pdf){#fig:training width="85%"
height="0.72\\textheight"}

A ordem ilustrada é uma ordem de dependências analíticas: qualidade antes de inferência elétrica, referência comportamental antes de comparação local e contexto antes de decisão de campo. Ela não deve ser interpretada como demonstração de que todas as etapas foram validadas com inspeções; representa sequência reproduzível de construção e testes.

O primeiro nível contém regras e classificadores físicos. O segundo
contém os modelos de comportamento e anomalia: modelo global,
especialistas locais por alimentador e arquitetura multivisão. A fusão
ou roteamento entre especialistas é avaliada somente depois que o
desempenho de cada componente isolado é conhecido. Um baseline simples
permanece como referência para impedir que maior complexidade seja
confundida com melhoria.

A unidade canônica de modelagem é `UC x cutoff_date`. Identificadores
(`id__`) e metadados de controle (`meta__`) são separados das features
(`x__`) e dos futuros rótulos (`y__`). Transformações que aprendem
parâmetros são ajustadas somente no conjunto de treino, e o split deve
respeitar temporalidade e agrupamentos para reduzir leakage
[@geron2025].

## Especialistas locais e distribuição por alimentador {#especialistas-locais-e-distribuiuxe7uxe3o-por-alimentador}

A possibilidade de treinar por alimentador tem dois objetivos. O
primeiro é científico: verificar se populações topologicamente mais
homogêneas produzem limites e arquétipos mais estáveis. O segundo é
operacional: permitir que a equipe examine um circuito específico sem
reprocessar a cidade inteira.

O modelo local não substitui automaticamente o modelo global. A
avaliação deve comparar ambos no mesmo recorte e adotar o local somente
quando houver volume suficiente e benefício mensurável. Caso contrário,
o modelo global funciona como âncora.

## Model Registry e comparação de versões {#model-registry-e-comparauxe7uxe3o-de-versuxf5es}

Cada treinamento deve gerar uma versão identificável, associada ao
intervalo de dados, catálogo de features, parâmetros, seed e código. O
Model Registry permite comparar versões e impedir que a última execução
seja automaticamente tratada como a melhor.

A TUI oferece comparação entre especialistas e modelo global no conjunto
de teste. A comparação deve ser acompanhada não somente por métricas
clássicas, mas também por estabilidade do ranking, cobertura da
população, número de UCs encaminhadas a cada fila e impacto operacional.

## Configurações globais e reprodutibilidade {#configurauxe7uxf5es-globais-e-reprodutibilidade}

Parâmetros operacionais como formato de saída, *random state* e taxa de
contaminação podem ser alterados pela interface. Para o trabalho
científico, essas configurações devem ser persistidas junto ao
experimento. O ambiente é gerenciado com `uv`, reduzindo divergência de
dependências entre execuções.

Credenciais de bancos não fazem parte do código ou dos artefatos
científicos. A aplicação utiliza configuração externa por ambiente ou
arquivo local ignorado pelo controle de versão.

## Papel da futura interface web

A TUI permanece orientada à engenharia do pipeline. O cockpit web, em
evolução, possui finalidade diferente: consumir resultados já
processados, filtrar a carteira, acompanhar indicadores e abrir o
detalhe de cada UC. A arquitetura prevista utiliza backend Python e
consultas colunares em Polars, com paginação do lado do servidor para
não transferir a população inteira ao navegador.

A visão web deverá apresentar filtros de alimentador, período,
categoria, score e status operacional; presets de investigação; seleção
dinâmica de colunas; exportação de listas; e um painel de detalhe
contendo série temporal, evidências por especialista, regras acionadas,
topologia, histórico do NIO e status da inspeção.

A distinção entre as interfaces é, portanto:

    TUI -> executar, auditar, treinar e governar
    WEB -> interpretar, priorizar, acompanhar e retroalimentar

## Cobertura funcional completa da TUI e interfaces associadas

A TUI baseada em Rich foi concebida como interface de operação da pesquisa, sobretudo nos ambientes onde comandos de terminal são mais adequados do que uma aplicação gráfica pesada. O menu organiza seis áreas: extração, treinamento, governança do Data Lake, comparação de modelos, diagnóstico de saúde e configuração. A capacidade de inspecionar a disponibilidade dos dados antes de executar o treino é parte do método de controle de qualidade, e não somente uma facilidade de interação.

| Recurso | Entrada ou escolha | Resultado e significado operacional |
|:--|:--|:--|
| Extração por `days_back` | um dia relativo | reprocessamento pontual e comparação de artefatos |
| Intervalo histórico | datas inicial/final | recuperação de janelas e composição do histórico |
| Auto-backfill | calendário e horizonte temporal | identificação de dias ausentes e atualização incremental |
| Filtro por alimentador | nome/identificador topológico | execução local e comparação com população global |
| Formato de saída | Parquet, CSV ou ambos | persistência tipada e intercâmbio para inspeção |
| Progresso por worker | lotes em execução | acompanhamento de gargalos, consultas e falhas |
| Calendário mensal | partições diárias e canais | observação de datas ausentes e formatos parciais |
| Auditoria de contagens | tabelas e períodos | inspeção da cardinalidade e divergências |
| Sincronização de formatos | Parquet e CSV | recuperação de cópias sem nova consulta à fonte |
| Treinamento baseline | matriz temporal de features | referência comparativa IF/K-Means |
| Treinamento composto | features elétricas | consenso entre detectores e regras |
| Treinamento hierárquico | dataset e alimentador | modelos locais, score global e roteamento |
| Classificador de tensão | dados/etiquetas de referência | experimento auxiliar de classificação elétrica |
| Filtro temporal de treino | janela selecionada | controle de corte e repetibilidade |
| Registro de modelos | artefatos e métricas | versionamento e comparação de versões |
| Comparação especialista/global | mesmo conjunto de teste | análise de diferenças de decisão |
| Health audit | diretórios e manifestos | identificação de falhas e inconsistências |
| Hiperparâmetros | contaminação, seed, formato | execução controlada e reprodutível |

A tabela cobre os comandos documentados da TUI e esclarece que algumas operações são orquestradas pela interface enquanto outras dependem de scripts externos. O pipeline completo com meteorologia é um caminho de pesquisa por linha de comando, não uma opção necessariamente presente no menu. De forma semelhante, a existência de uma opção de *Model Registry* não garante que toda governança de modelos esteja automatizada; o registro e a comparação precisam ser distinguidos das interfaces parciais e de futuras decisões de promoção de versões.

O calendário fornece dois diagnósticos diferentes que devem ser apresentados juntos: o **estado materializado da partição** e a **completude efetiva da telemetria**. Um dia com Parquet presente pode continuar contendo lacunas para uma parcela dos NIOs; uma falha de conversão entre formatos não equivale a ausência do evento no MDM. A visualização permite decidir se a ação correta é converter uma cópia, reextrair um dia, inspecionar o vínculo UC–NIO ou encaminhar medidores ao especialista de comunicação.

A comparação de modelos deve ocorrer sobre o mesmo cutoff, mesma coorte elegível e mesmas features, exceto pela alteração deliberada no ablation study. Sem esse controle, diferenças de F1, contaminação ou quantidade de alertas podem decorrer apenas de alterações de amostra. Ao salvar o modelo, o repositório de artefatos deve ser interpretado em conjunto com o manifesto de dados e o pré-processamento correspondente.

A aplicação Web complementa a TUI, mas atende outro usuário: o analista que precisa investigar a UC no tempo. Seu Raio-X apresenta curva de energia, referência histórica, indicador de cobertura, categoria e evolução da anomalia, com consulta temporal sob demanda e navegação entre candidatos. Uma eventual posição na lista deve remeter ao mesmo instante de corte e à mesma versão de modelo registrada pela execução de treinamento; do contrário, TUI e Cockpit exibiriam decisões de campanhas distintas sem identificação.

# Operacionalização da detecção e priorização de inspeções

## Mudança do paradigma operacional {#mudanuxe7a-do-paradigma-operacional}

A introdução da ferramenta altera a maneira como a carteira de inspeções
pode ser formada. O sistema não adiciona apenas uma nova fonte de
suspeitas a uma fila existente; ele cria a possibilidade de uma **fila
dinâmica, explicável e limitada pela capacidade de campo**.

Em um processo estático, a seleção é construída em determinado instante
e as equipes executam a carteira até sua renovação. Na proposta deste
trabalho, a população é reavaliada periodicamente e cada UC mantém
estado, histórico de evidências e posição relativa. Casos podem entrar,
ganhar ou perder prioridade, ser desviados para outra área técnica,
permanecer em observação ou ser encerrados.

![Mudança do paradigma operacional: monitoramento, priorização, inspeção
e feedback.](figures/operational_cycle.pdf){#fig:operational_cycle
width="95%" height="0.75\\textheight"}

O ciclo representa a ligação entre produção de evidências, restrição de capacidade, inspeção e retroalimentação. A realimentação só gera rótulos de verdade operacional quando o resultado de campo é registrado com data e protocolo; a ausência de inspeção não equivale a caso negativo. Assim, o esquema fundamenta a governança da decisão sem pressupor ganhos financeiros medidos.

A principal consequência é que o *threshold* deixa de ser apenas um
parâmetro estatístico. Ele passa a responder também à capacidade
disponível. Se a quantidade de casos acionáveis excede sistematicamente
o número de inspeções possíveis, o produto não está operacionalmente
dimensionado; se produz poucos candidatos fortes, pode ampliar de
maneira controlada o universo investigado.

A operação fornece ainda o principal mecanismo de aprendizado futuro.
Uma inspeção concluída adiciona informação que os dados AMI isolados não
possuem: confirmação ou não da irregularidade, categoria encontrada,
valor associado, tempo de execução e eventual causa alternativa. Esses
resultados permitem recalibrar o ranking e, com histórico suficiente,
treinar modelos supervisionados sobre rótulos operacionais reais.

O ciclo deixa de ser apenas `dados -> modelo` e passa a ser:

$$\text{observar}\rightarrow\text{priorizar}\rightarrow\text{inspecionar}
\rightarrow\text{avaliar}\rightarrow\text{recalibrar}.$$

Essa transformação é uma contribuição do trabalho porque explicita o
impacto organizacional da ferramenta: o algoritmo precisa caber na
operação, e a operação passa a produzir o ground truth que permite
melhorar o algoritmo.

### Ciclo de vida de um caso

Cada UC deve possuir estado operacional independente do score
instantâneo. Uma máquina de estados mínima é:

    normal
      -> observação
      -> acionável
      -> em fila
      -> inspecionada
           -> positiva
           -> negativa
           -> inconclusiva
      -> encerrada / monitorada

Persistência do score não cria automaticamente novos casos. Um novo
episódio só é aberto após critério explícito de reabertura, evitando
multiplicar a demanda pela repetição diária da mesma condição.

### Abstenção e evidência insuficiente {#abstenuxe7uxe3o-e-eviduxeancia-insuficiente}

O sistema deve possuir a opção de **não decidir**. Histórico curto,
baixa qualidade de telemetria, conflito entre fontes ou ausência de
contexto suficiente podem levar a um estado de evidência insuficiente.
Essa abstenção é preferível a forçar toda UC a receber um diagnóstico
elétrico.

### Trilha de auditoria da decisão {#trilha-de-auditoria-da-decisuxe3o}

Para cada prioridade gerada, a trilha de auditoria deve preservar pelo
menos:

-   `cutoff_date`;

-   versão do dataset e do modelo;

-   features disponíveis;

-   scores dos especialistas;

-   regras determinísticas acionadas;

-   posição no ranking e threshold vigente;

-   contexto utilizado na decisão.

Essa trilha permite reproduzir por que uma UC ocupou determinada vaga de
inspeção em determinado momento.

### Drift e validade temporal dos modelos

A população não é estacionária. Novos medidores, expansão de geração
distribuída, mudanças de hábitos, sazonalidade e alterações da rede
podem modificar a distribuição das features. Portanto, versões de modelo
devem ser acompanhadas por janela de treinamento e indicadores de drift.
Uma queda de desempenho pode representar envelhecimento do modelo, e não
necessariamente problema no código.

## Funil operacional de isolamento por camadas

### Por que medir isolamento e não apenas anomalia {#por-que-medir-isolamento-e-nuxe3o-apenas-anomalia}

Um detector que retorna milhares de UCs sem indicar qual ação cada grupo
exige transfere o problema para a operação. A arquitetura deve
quantificar **quanto cada camada reduz, separa ou redireciona a
população**.

Seja $U^{(0)}$ o conjunto de UCs elegíveis no início do processo. Para
uma camada operacional $k$, define-se:

$$I_k = \{u\in U^{(k-1)}: u\;\text{é isolada para uma ação específica na camada }k\},$$

$$U^{(k)} = U^{(k-1)}\setminus I_k.$$

A taxa de isolamento da camada é

$$\rho_k = \frac{|I_k|}{|U^{(k-1)}|},$$

e a razão residual acumulada é

$$R_k = \frac{|U^{(k)}|}{|U^{(0)}|}.$$

Essas métricas devem ser apresentadas em conjunto com os conjuntos
brutos de evidências, pois uma mesma UC pode ser marcada por mais de um
especialista. Para evitar dupla contagem, o relatório operacional deve
manter dois planos:

1.  **evidência multilabel**, em que uma UC pode aparecer em vários
    detectores;

2.  **fila de ação deduplicada**, em que cada episódio recebe uma
    prioridade e uma ação principal.

### Filas de ação {#filas-de-auxe7uxe3o}

O objetivo não é encaminhar toda anomalia para fiscalização de PNT. O
sistema deve separar, por exemplo:

| Evidência predominante | Ação primária sugerida |
|:--|:--|
| Baixa cobertura ou gaps | Triagem de comunicação e dados |
| Mudança compartilhada no transformador | Análise de rede e manutenção |
| Regra de qualidade elétrica | Avaliação técnica de QEE |
| Troca de medidor coincidente | Auditoria cadastral e de medição |
| Tamper, mudança individual e boa telemetria | Inspeção PNT por evidências |
| Anomalia comportamental isolada | Monitoramento e investigação complementar |

A qualidade do sistema aumenta quando consegue **retirar casos da fila
errada**, não apenas quando reduz o número absoluto de candidatos.

### Estatísticas obrigatórias do funil {#estatuxedsticas-obrigatuxf3rias-do-funil}

Para cada execução mensal deverão ser calculados:

-   população elegível inicial;

-   UCs desviadas por problemas de dados;

-   UCs desviadas para análise de rede;

-   UCs com mudança de regime;

-   UCs anômalas em $S_{elec}$;

-   UCs com evidência de tamper;

-   UCs com duas ou mais evidências convergentes;

-   UCs finais elegíveis para inspeção;

-   percentual de redução acumulado;

-   interseções entre detectores;

-   UCs novas no mês;

-   UCs persistentes do mês anterior;

-   UCs resolvidas;

-   UCs reabertas.

## Incidência mensal, backlog e capacidade de inspeção {#inciduxeancia-mensal-backlog-e-capacidade-de-inspeuxe7uxe3o}

### Estoque versus fluxo

Uma anomalia persistente não deve ser contada como nova todos os dias. O
sistema precisa distinguir **estoque de casos** e **entrada
incremental**.

Defina:

-   $A_m$: número de novas UCs que entram em estado acionável no mês
    $m$;

-   $B_m$: backlog não resolvido no início do mês;

-   $K_m$: capacidade operacional de inspeções no mês;

-   $C_m$: casos concluídos no mês.

Com simplificação de uma inspeção por caso:

$$B_{m+1}=\max(0,B_m + A_m - K_m).$$

A razão de carga operacional é

$$\lambda_m = \frac{B_m+A_m}{K_m}.$$

Interpretação:

-   $\lambda_m>1$: demanda superior à capacidade;

-   $\lambda_m\approx1$: carteira compatível com a capacidade;

-   $\lambda_m<1$: há margem para ampliar a seleção ou absorver backlog.

A sustentabilidade de longo prazo exige que a incidência média de novos
casos acionáveis não ultrapasse sistematicamente a capacidade média.

### Episódios e deduplicação temporal {#episuxf3dios-e-deduplicauxe7uxe3o-temporal}

Cada UC deve possuir estado operacional, por exemplo:

    normal -> observação -> acionável -> em fila -> inspecionada -> resolvida
                                          |               |
                                          +----persistente+

Uma UC só volta a ser contabilizada como nova quando um episódio
anterior foi encerrado e um novo evento satisfaz a política de
reabertura. Essa estrutura evita inflar a demanda pela repetição diária
da mesma anomalia.

## Controle adaptativo da granularidade da seleção {#controle-adaptativo-da-granularidade-da-seleuxe7uxe3o}

A quantidade de candidatos não deve ser independente da capacidade
operacional. O produto precisa ser suportado pela operação.

Se a quantidade de candidatos fortes excede a capacidade, o sistema deve
**estreitar** a seleção. Se a capacidade está ociosa e a taxa de sucesso
permanece adequada, o sistema pode **ampliar** a seleção.

### Quando estreitar

Se $\lambda_m>1$, alternativas incluem:

-   elevar o threshold do score;

-   exigir duas ou mais evidências convergentes;

-   exigir persistência mínima;

-   priorizar casos sem explicação topológica coletiva;

-   aumentar peso de tamper ou CPD coincidente;

-   priorizar maior valor econômico esperado;

-   reduzir o horizonte de candidatos muito antigos;

-   estabelecer cotas por classe ou região para evitar concentração
    indevida.

### Quando ampliar

Se $\lambda_m<1$, o sistema pode:

-   reduzir moderadamente thresholds;

-   aceitar uma única evidência muito forte;

-   incluir casos de monitoramento com persistência crescente;

-   ampliar classes de comportamento ou regiões;

-   recuperar candidatos do backlog com boa evidência;

-   utilizar uma política mais generalista para preencher capacidade sem
    forçar alvos fracos.

O limite de 40% de substituição proposto neste trabalho é tratado como
**meta máxima de piloto**, não como obrigação. Caso não existam
candidatos novos com evidência suficiente, as vagas não devem ser
preenchidas artificialmente por UCs fracas; elas retornam à carteira
operacional convencional.

## Política piloto de 40% de substituição {#poluxedtica-piloto-de-40-de-substituiuxe7uxe3o}

### Separação entre repriorização e descoberta {#separauxe7uxe3o-entre-repriorizauxe7uxe3o-e-descoberta}

Um ponto essencial é não atribuir ao modelo o mérito de "descobrir" uma
UC que já constava na lista de inspeção da operação.

A avaliação distingue três coortes:

#### Coorte O - Overlap / repriorização {#coorte-o---overlap-repriorizauxe7uxe3o}

UCs que já estavam presentes na lista operacional no instante da decisão
e que também recebem score elevado do sistema.

O ganho do modelo é **ordenação**, não descoberta.

#### Coorte A - antecipação {#coorte-a---antecipauxe7uxe3o}

UCs que não estavam na lista no instante da decisão do modelo, mas que
entram posteriormente na lista operacional dentro de um horizonte de
comparação definido.

Se confirmadas, representam um ganho de **antecipação temporal**, não
uma descoberta exclusiva.

#### Coorte N - descoberta incremental

UCs que não estavam na lista operacional no instante da decisão, não
entram nela dentro do horizonte de comparação e são confirmadas após
inspeção originada pelo sistema.

Essa é a coorte que representa descoberta efetivamente incremental.

### Composição da carteira mensal {#composiuxe7uxe3o-da-carteira-mensal}

Se $K_m$ é a capacidade mensal, define-se como hipótese de projeto:

$$K_{new,m}\le 0.40K_m,$$

$$K_{legacy,m}=K_m-K_{new,m}.$$

O modelo pode repriorizar a carteira legada inteira, mas no máximo 40%
das vagas são inicialmente reservadas a candidatos externos à lista
operacional. Esse desenho permite aprender com o modelo sem remover
completamente o mecanismo atual.

A política é:

$$K_{new,m}=\min(D_{new,m},0.40K_m),$$

onde $D_{new,m}$ é o número de candidatos novos que satisfazem os
critérios mínimos. Vagas experimentais não ocupadas retornam à carteira
legada.

## Métricas de sucesso operacional {#muxe9tricas-de-sucesso-operacional}

### Por que não utilizar somente accuracy {#por-que-nuxe3o-utilizar-somente-accuracy}

A inspeção atua sobre uma carteira selecionada majoritariamente para
casos suspeitos. Não existe, nesse recorte, uma amostra balanceada de
positivos e negativos representativa de toda a cidade. Assim, "accuracy"
pode ser enganosa.

A métrica primária proposta é a **taxa de sucesso da inspeção**,
equivalente à precisão entre os casos inspecionados:

$$Y = \frac{N_{positivo}}{N_{inspecionado}}.$$

O objetivo de projeto é obter ganho de pelo menos 5 **pontos
percentuais** em relação à referência operacional:

$$\Delta Y = Y_{policy}-Y_{baseline}\ge0.05.$$

Essa interpretação em pontos percentuais deve ser confirmada com a
operação; se a empresa preferir ganho relativo de 5%, a fórmula deverá
ser alterada.

### Repriorização da lista existente {#repriorizauxe7uxe3o-da-lista-existente}

Para UCs já presentes na lista, o efeito é medido por curvas de ganho
acumulado:

$$Yield@k = \frac{TP@k}{k},$$

$$Lift@k = \frac{Yield_{modelo}@k}{Yield_{ordem\;base}@k}.$$

Se ambas as políticas eventualmente inspecionariam as mesmas UCs, a
principal vantagem é encontrar casos positivos mais cedo. Pode-se medir:

$$\Delta t_i = t_{baseline,i}-t_{modelo,i},$$

para cada caso positivo comum às duas listas.

Quando houver estimativa confiável de perda diária persistente $L_i$, o
valor de antecipação pode ser aproximado por

$$V_{antecipacao,i}=\max(0,\Delta t_i)\cdot L_i.$$

Essa parcela só deve ser utilizada quando a hipótese de perda continuada
for tecnicamente defensável.

### Descoberta de novas UCs

Para a coorte N, as métricas mínimas são:

-   taxa de sucesso de novos alvos;

-   quantidade de positivos incrementais;

-   valor cobrado/recuperável associado aos positivos;

-   custo de inspeções infrutíferas;

-   oportunidade perdida pela substituição de vagas da lista legada.

# Avaliação econômica e protocolo de validação

## Modelo econômico {#modelo-econuxf4mico}

### Valor do caso confirmado

O projeto utiliza como base econômica o valor de cobrança/recuperação
atribuído pela companhia ao caso confirmado. Denote esse valor por

$$V_{claim,i}.$$

Ex post, ele é conhecido após o processo de apuração. Ex ante, antes da
inspeção, precisa ser estimado por $\widehat V_{claim,i}$ a partir das
informações disponíveis, caso o valor seja utilizado no ranking.

É importante distinguir valor cobrado de valor efetivamente realizado.
Se a análise econômica quiser considerar risco de recuperação, pode-se
introduzir $\theta_i\in[0,1]$, representando a fração esperada de
realização. No cenário operacional simplificado solicitado pela empresa,
pode-se iniciar com $\theta_i=1$ e reportar o valor bruto de cobrança
separadamente.

### Valor esperado absoluto de uma inspeção nova {#valor-esperado-absoluto-de-uma-inspeuxe7uxe3o-nova}

Se $p_i$ é a probabilidade calibrada de confirmação e $C_i$ é o custo
total da inspeção, o valor monetário esperado é

$$EMV_i = p_i\theta_i\widehat V_{claim,i}-C_i.$$

Esse resultado decorre de:

$$EMV_i=p_i(\theta_i\widehat V_{claim,i}-C_i)+(1-p_i)(-C_i).$$

Uma inspeção nova que não gera cobrança contribui com $-C_i$, exatamente
como desejado no balanço econômico.

**Importante:** score de anomalia não é probabilidade. Enquanto
$S_{elec}$, CPD e outros componentes não estiverem calibrados contra
resultados de inspeção, $p_i$ não pode ser substituído diretamente por
um score normalizado. Na fase inicial, o ranking deve ser avaliado por
bins de score e taxas empíricas de sucesso. A calibração probabilística
só passa a ser usada após acumular ground truth suficiente.

### Custo de oportunidade da substituição {#custo-de-oportunidade-da-substituiuxe7uxe3o}

Quando a capacidade mensal é fixa, uma nova inspeção geralmente
**substitui**, e não adiciona, uma inspeção legada. Nesse caso, avaliar
apenas $EMV_i$ é insuficiente. Existe uma alternativa deslocada $j$.

O valor incremental correto é

$$\Delta EMV_{i\leftarrow j}
=
\left(p_i\theta_i\widehat V_i-C_i\right)
-
\left(p_j^{(0)}\theta_j^{(0)}\widehat V_j-C_j\right).$$

Se os custos de inspeção forem semelhantes, a diferença econômica é
dominada pela expectativa de confirmação e pelo valor do caso. Se a
inspeção nova exige deslocamento maior, equipe especializada ou custo
adicional, essa diferença deve permanecer explícita.

Essa formulação evita um erro comum: contabilizar o valor de uma nova
fraude encontrada sem considerar que, para inspecioná-la, outra UC
deixou de ser visitada.

### Valor ex post do piloto

Após as inspeções, o resultado realizado de uma carteira $P_m$ pode ser
calculado por

$$V^{real}_m = \sum_{i\in P_m}\left(y_iV_{claim,i}-C_i\right),$$

onde $y_i=1$ para inspeção positiva e $0$ para negativa.

O impacto incremental mensal é

$$\Delta V_m=V^{real}_{policy,m}-V^{real}_{baseline,m}.$$

A avaliação deve também reportar separadamente:

-   valor de casos da coorte O;

-   valor antecipado da coorte A;

-   valor de descobertas realmente incrementais da coorte N;

-   custo das inspeções negativas em N;

-   valor esperado/realizado dos casos deslocados da lista legada.

### Cenários econômicos e análise de sensibilidade {#cenuxe1rios-econuxf4micos-e-anuxe1lise-de-sensibilidade}

O modelo econômico deve ser utilizado como estrutura de comparação, não
como promessa de recuperação. Três cenários serão apresentados na versão
final:

1.  **conservador:** menor taxa de confirmação e custo de inspeção
    elevado;

2.  **base:** parâmetros observados durante o piloto;

3.  **favorável:** maior confirmação e maior valor médio por caso.

Para cada cenário serão avaliados o valor esperado da carteira, custo
das inspeções negativas, valor dos casos positivos e custo de
oportunidade das vagas deslocadas. A sensibilidade deverá mostrar quais
parâmetros dominam o resultado.

Se $K$ é fixo, o ponto central não é maximizar o valor de cada caso
isoladamente, mas o valor do portfólio sujeito à capacidade:

$$\max_{P:|P|\le K}\sum_{i\in P}EMV_i.$$

A formulação aproxima o problema da própria decisão de inspeção
orientada a retorno discutida por trabalhos recentes [@han2026].

### Valor pleiteado, realizado e recuperado

Nesta etapa, o valor econômico positivo será associado ao valor que a
companhia atribui/cobra no processo decorrente da irregularidade
confirmada. Esse valor deve ser distinguido de duas quantidades futuras:
valor efetivamente recebido e valor líquido após custos jurídicos ou de
cobrança.

Assim, o TCC reportará inicialmente **valor pleiteado associado às
inspeções positivas**. Se a operação fornecer posteriormente taxa de
realização, o modelo poderá introduzir um fator $\theta_i$ entre zero e
um.

Essa distinção impede que o trabalho trate automaticamente cobrança
emitida como caixa efetivamente recuperado.

### Dados necessários para fechar o modelo econômico {#dados-necessuxe1rios-para-fechar-o-modelo-econuxf4mico}

A documento dependerá de parâmetros operacionais ainda a levantar:

-   custo médio de uma inspeção;

-   custo adicional por deslocamento ou equipe especializada;

-   capacidade mensal por equipe/região;

-   valor pleiteado por inspeção positiva;

-   tempo médio entre seleção e inspeção;

-   destino das inspeções inconclusivas;

-   taxa histórica de sucesso da carteira convencional;

-   quando disponível, proporção efetivamente recuperada do valor
    pleiteado.

A formulação separa, assim, benefício bruto pleiteado, realização
financeira e custo de oportunidade, sem pressupor valores operacionais
não observados.

### Condição de equilíbrio e exemplo econômico transparente

Quando a capacidade é fixa, o valor marginal de uma nova inspeção corresponde à diferença entre o valor esperado do caso selecionado $i$ e o valor esperado do caso convencional deslocado $j$:

$$\Delta EMV_{i\leftarrow j}=p_i\theta_iV_i-C_i-\left(p_j\theta_jV_j-C_j\right).$$

Em uma simulação **puramente ilustrativa**, considere um candidato novo com probabilidade calibrada hipotética $p_i=0{,}35$, valor bruto pleiteável $V_i=\mathrm{R\$}\,2.000$, realização $\theta_i=0{,}80$ e custo total $C_i=\mathrm{R\$}\,180$. Seu valor esperado é $\mathrm{R\$}\,380$. Se a vaga deslocada teria $EMV_j=\mathrm{R\$}\,250$, o incremento esperado seria $\mathrm{R\$}\,130$. Se a mesma vaga convencional tivesse $EMV_j=\mathrm{R\$}\,500$, a substituição destruiria $\mathrm{R\$}\,120$ de valor esperado mesmo com o novo candidato apresentando benefício bruto positivo.

Esses valores não são extraídos das inspeções da concessionária: demonstram por que comparar recuperação bruta com custo da nova visita é insuficiente. Se a operação utiliza o valor pleiteado como benefício de referência, deve identificá-lo como valor bruto; realização e recuperação financeira pertencem a etapas posteriores. A definição de $p_i$ requer calibração contra resultados confirmados, não transformação direta do score não supervisionado.

A condição de equilíbrio para substituir $j$ por $i$ é $p_i\theta_iV_i-C_i>EMV_j$. Essa desigualdade fornece critério auditável para simulação de capacidade sem atribuir ao projeto metas financeiras como fatos observados.

## Ranking em duas filas

Os casos já conhecidos pela operação e os casos novos não devem receber
exatamente a mesma política.

### Fila de repriorização {#fila-de-repriorizauxe7uxe3o}

Para a coorte O, o risco de "criar trabalho novo" é baixo, porque a
inspeção já estava prevista. O ranking pode dar maior peso a:

-   convergência entre evidências;

-   severidade;

-   magnitude da mudança;

-   persistência;

-   valor estimado;

-   idade na fila;

-   impacto de antecipação.

### Fila de descoberta

Para a coorte N, a decisão cria uma inspeção que não existiria pela
seleção operacional naquele horizonte. O critério deve ser mais
rigoroso. Pode exigir:

-   qualidade de dados suficiente;

-   pelo menos duas visões convergentes ou uma evidência determinística
    forte;

-   ausência de explicação topológica coletiva;

-   persistência mínima;

-   EMV positivo ou score acima de um threshold superior;

-   limite mensal de até 40% da capacidade.

Essa assimetria protege a operação contra a expansão indiscriminada da
carteira.

## Protocolo de avaliação em campo {#protocolo-de-avaliacao-em-campo}

A validação do rendimento de inspeção exige resultados observados em
campo e uma carteira convencional de comparação registrada antes da
decisão do modelo. Scores calculados sem rótulos são úteis para
descrever raridade, convergência entre evidências e seletividade do
funil; não representam, isoladamente, acertos de detecção de furto.

### Desenho do piloto

O protocolo mensal distingue a lista existente, os casos antecipados e
as descobertas incrementais. Antes do início de cada ciclo são
congelados o instante de corte dos dados, a versão do modelo, os
thresholds, a carteira convencional e a capacidade disponível.

1.  Registrar a lista operacional de referência e a capacidade mensal
    antes dos resultados de campo.

2.  Gerar scores, evidências e categorias exclusivamente com informações
    anteriores ao *cutoff*.

3.  Identificar sobreposição (O), antecipação (A) e descoberta
    incremental (N).

4.  Reservar até 40% das vagas para candidatos incrementais que
    satisfaçam critérios mínimos de qualidade e evidência.

5.  Utilizar o ranking também para ordenar os candidatos presentes na
    carteira convencional.

6.  Registrar execução, confirmação, resultado inconclusivo, custo e
    valor pleiteado e recuperado de cada inspeção.

7.  Calcular rendimento@K, *lift*, antecipação, valor incremental e
    custo de oportunidade por coorte.

8.  Calibrar políticas apenas para ciclos subsequentes, preservando as
    decisões históricas para auditoria.

### Controle do viés de seleção {#controle-do-vies-de-selecao}

Quando o modelo influencia quem será inspecionado, a ausência de
inspeção não pode ser interpretada como ausência de irregularidade. O
protocolo admite uma parcela de controle selecionada pela rotina
convencional ou por amostragem dentro de estratos definidos previamente.
Isso permite estimar diferenças de rendimento com menor dependência das
escolhas do próprio detector, embora não elimine toda a incerteza
causal.

Os indicadores de campo são, portanto, distintos das métricas do
experimento sintético: a primeira avaliação mede resultado operacional;
a segunda verifica mecanismos e complementaridade entre algoritmos.

# Governança e interfaces de investigação

## Rastreabilidade, feedback e governança do produto {#rastreabilidade-feedback-e-governanuxe7a-do-produto}

A plataforma precisa preservar a ligação entre dado bruto,
transformação, feature, modelo, prioridade e resultado de campo. Essa
rastreabilidade é necessária tanto para pesquisa quanto para uso
operacional.

### Contratos de dados

As camadas normalizadas possuem grãos diferentes e não devem ser
achatadas prematuramente. Contexto cadastral, histórico de instalação,
hierarquia elétrica, medições intervalares, instantâneos, registradores
e alarmes permanecem em tabelas semanticamente separadas. A matriz de
treino é um produto derivado e versionado, não a fonte original.

### Catálogo de features {#catuxe1logo-de-features}

Cada feature deve registrar origem, janela, unidade, regra de missing,
transformação, domínio físico e modelos que a consomem. Isso reduz a
possibilidade de incluir acidentalmente variáveis de comunicação no
especialista elétrico ou aplicar atributos cadastrais futuros a datas
passadas.

### Feedback de inspeção {#feedback-de-inspeuxe7uxe3o}

O resultado de campo deve ser persistido como evento associado à UC e ao
instante da decisão. Para treinamento futuro, é necessário distinguir
`não inspecionada`, `inspeção negativa`, `positiva`, `inconclusiva` e
`não executável`. Ausência de inspeção não é rótulo negativo.

### Versionamento ponta a ponta

O relatório de uma UC deve ser reconstruível a partir de quatro versões:
dados, features, modelo e política de ranking. Alterar qualquer uma
delas pode mudar a posição da UC. Esse controle também permite comparar
versões do sistema sem reescrever retrospectivamente decisões já
tomadas.

## Cockpit analítico e apoio à inspeção {#cockpit-analitico-e-apoio-a-inspecao}

O Cockpit Web complementa a TUI ao fornecer uma interface de
investigação centrada na UC. O protótipo funcional permite navegar entre
candidatos, consultar séries temporais sob demanda e confrontar consumo
medido, linha de base, cobertura de telemetria e categorias analíticas.
Essa separação mantém a TUI como interface de operação da engenharia de
dados e o Cockpit como interface de leitura e interpretação das
evidências.

### Investigação individual e Raio-X da UC

A inspeção visual começa pela posição da unidade na carteira e avança
para seus indicadores temporais e elétricos. A sobreposição entre
consumo, referência robusta e disponibilidade dos dados torna explícito
se um desvio está acompanhado por observações suficientes para
interpretação. Os estudos de caso deste trabalho reproduzem a mesma
leitura em figuras estáticas. Alarmes e metadados topológicos funcionam
como contexto, não como confirmação automática de irregularidade.

### Visão de carteira e funil analítico

A estrutura analítica mantém destinos distintos para baixa qualidade de
telemetria, eventos coletivos de rede, explicações cadastrais e
candidatos a investigação individual. Um caso não precisa
necessariamente desaparecer do universo analisado quando recebe uma
explicação alternativa: ele pode ser encaminhado a outra fila ou
permanecer em observação. A contagem por etapa deve registrar inclusive
sobreposição entre especialistas.

    População elegível
          |
          +-> telemetria insuficiente -> qualidade/comunicação
          +-> evento coletivo -------> rede/manutenção
          +-> contexto cadastral ----> saneamento/monitoramento
          +-> evidência individual --> carteira de inspeções
                                      |
                                      +-> repriorização
                                      +-> descoberta incremental

### Capacidade, retorno e governança de carteira

A integração operacional prevista para o Cockpit reúne capacidade
mensal, *backlog*, incidência de novos episódios, composição das coortes
O/A/N e valor econômico esperado. Simulações de threshold mostram como a
seletividade modifica volume e custo de oportunidades. Esses indicadores
pertencem à camada de decisão e utilizam as definições formalizadas nos
capítulos de capacidade e modelo econômico, sem serem confundidos com
resultados experimentais já medidos.

A rastreabilidade por versão de dados, features, modelo e política
garante que a apresentação de uma UC possa ser reproduzida
posteriormente.

# Resultados e discussão {#resultados-e-discussao}

Os resultados obtidos são apresentados em três escalas complementares:
infraestrutura de processamento e interfaces de análise; experimento
metodológico controlado; e estudos de caso com telemetria real
mascarada. Essa separação impede que observações ilustrativas sejam
confundidas com desempenho de fiscalização em toda a população.

## Resultados da plataforma e da engenharia de dados

A aplicação reúne extração por data e alimentador, sincronização
inteligente de períodos ausentes, escrita por partições, auditoria de
continuidade temporal e orquestração de treinamento pela TUI. O emprego
de processamento colunar em Polars e persistência por
`sink_parquet`/`sink_csv` estabelece uma arquitetura na qual períodos
concluídos são preservados independentemente do término de uma campanha
inteira de extração. O calendário de disponibilidade torna observável a
integridade das partições e diferencia falhas de processamento de
lacunas de telemetria.

A reconstrução temporal da relação UC--NIO corrige uma limitação
estrutural de datasets construídos apenas com o medidor atualmente
instalado. Trocas de medidor, recomeço de registradores e mudanças de
cadência passam a constituir eventos cadastrais rastreáveis, evitando
que sejam automaticamente interpretados como alterações físicas de
consumo. O contrato `UC x cutoff` organiza os dados utilizados pelo
modelo e torna explícita a separação entre identificadores, metadados,
features e rótulos de inspeção.

## Resultados do experimento metodológico controlado

Na população sintética de 1.000 curvas diárias, os algoritmos
apresentaram respostas distintas para morfologia, raridade global,
densidade local e extremidades de distribuição. O K-Means produziu
agrupamentos de curvas; a PCA mostrou estrutura dominante e resíduos de
reconstrução; Isolation Forest, LOF e ECOD atribuíram prioridades
diferentes aos mesmos casos. A sobreposição dos cem primeiros casos de
cada detector com a seleção composta ficou entre 33% e 72%, conforme a
configuração executada. Trata-se de evidência de diversidade de ranking
no experimento, não de superioridade em detecção de furto.

A Tabela [tabela](#tab:runtime-demo) fornece também um primeiro retrato de
custo computacional: 1.931,24 ms para Isolation Forest, 14,29 ms para
LOF e 1,39 ms para ECOD na execução documentada. O resultado não deve
ser extrapolado diretamente para os registros da distribuidora, pois
depende das dimensões do conjunto, implementação e configuração dos
algoritmos. O valor do experimento é permitir comparação controlada,
reprodução e inspeção visual das decisões.

O ensaio de ablação ilustra a variação do número de candidatos quando se
acrescentam qualidade de dados, referências comportamentais e contexto.
Como a população controlada possui mecanismos de geração conhecidos, a
interpretação de seus gráficos não requer rotular irregularidades reais
por inferência.

## Resultados dos estudos de caso AMI

As curvas mascaradas apresentadas nas Figuras
[figura](#fig:comparacao-ucs-mesmo-consumo) e
[10.2](#fig:comparacao-semanal){reference-type="ref"
reference="fig:comparacao-semanal"} demonstram que UCs com energia
diária mediana semelhante podem diferir substancialmente em dispersão,
horário de uso e repetição semanal. O par estudado apresenta medianas de
20,38 e 19,22 kWh/dia; a diferença de morfologia justifica manter
representação de forma independente da magnitude agregada.

Na Figura [figura](#fig:outlier-real), uma UC destacada pelo pipeline apresenta
simultaneamente comportamento temporal, desvio de referência robusta e
indicadores elétricos. O score documentado de 0,621 representa raridade
segundo o ensemble aplicado; a descrição `DESEQUILIBRIO_CARGA_OU_BYPASS`
orienta a hipótese de investigação, mas não equivale a constatação de
furto. Essa distinção é essencial para que a camada de aprendizado de
máquina opere como suporte técnico e não como substituto da inspeção.

## Síntese das contribuições observadas

-   A plataforma operacionaliza a passagem de medições heterogêneas para
    séries e datasets temporalmente auditáveis.

-   A demonstração controlada evidencia o motivo de manter técnicas
    complementares e permite avaliar cada uma por sua pergunta
    estatística.

-   Os casos reais mostram a utilidade de observar consumo agregado,
    forma, sazonalidade semanal e qualidade de telemetria conjuntamente.

-   O modelo de capacidade e valor esperado transforma as evidências
    analíticas em um protocolo de decisão compatível com restrições de
    campo.

A hipótese de ganho de rendimento em inspeção e a meta de substituição
de até 40% da carteira convencional são parâmetros de avaliação
operacional, não medidas extraídas dos experimentos. O estudo não
atribui melhora efetiva de detecção de PNT sem confirmações de campo.
Seu resultado demonstrado consiste na integração da plataforma de dados,
na explicitação das funções dos detectores e na interpretação auditável
de casos representativos.

## Discussão transversal: o que muda da abordagem original

A comparação entre a metodologia global original e a decomposição atual não pode ser reduzida a ``mais algoritmos''. O baseline M0 identifica raridade agregada; M1 assegura identidade válida no tempo; M2 restringe inferências quando faltam dados; M3 inclui referências estatísticas locais; M4 acrescenta persistência e mudança de regime; M5 incorpora evidência elétrica e topológica. Cada etapa altera **o significado** de uma observação sinalizada, além de alterar a contagem de casos.

Os gráficos do demonstrador permitem verificar que os detectores escolhem candidatos distintos. No entanto, a mera discrepância entre rankings não demonstra que a composição seja superior. Ganho metodológico exige evidência de redução de erro de interpretação, robustez temporal ou melhor correspondência com observações externas; ganho operacional exige incremento em resultados positivos e benefício econômico por capacidade fixa. A ablação proposta permite observar ambas as dimensões sem declarar sucesso antes de sua medição.

Outra contribuição demonstrável é a capacidade de reproduzir a cadeia de evidência. Uma observação exibida no Cockpit pode ser vinculada à curva original, NIO instalado à época, condições de qualidade, janela estatística, versão do detector e categoria de encaminhamento. A explicabilidade, nesse contexto, não se resume a visualizar a importância de uma feature: ela depende de recuperar **qual dado existia quando a decisão foi tomada**.

Finalmente, a análise de sensibilidade do gate mostra por que separar comunicação de comportamento elétrico é necessário, mas insuficiente se a política de elegibilidade não for explicitada. Uma regra capaz de reduzir alertas espúrios pode também impedir análise de uma parte relevante da população. O equilíbrio entre confiabilidade e cobertura deve ser reportado como resultado próprio da ferramenta, não escondido dentro do score final.

# Limitações e riscos metodológicos {#limitacoes-e-riscos-metodologicos}

A principal restrição na interpretação de um detector não supervisionado
é a distinção entre raridade estatística e irregularidade confirmada. Os
exemplos de AMI e os experimentos controlados demonstram mecanismos e
interpretações, mas não fornecem uma estimativa populacional de
incidência de furto. Medidas de rendimento de inspeção exigem resultados
independentes de campo e controle do viés introduzido pela própria
seleção dos candidatos.

A infraestrutura de medição apresenta limitações físicas e
informacionais. Cadências e grandezas diferem entre equipamentos,
lacunas podem decorrer de falhas de comunicação, e dados de baixa
frequência não descrevem eventos de qualidade de energia em escalas
subcíclicas. Além disso, substituições de NIO, alterações cadastrais e
reinicializações de registradores podem produzir descontinuidades
artificiais quando a temporalidade do vínculo não é preservada.

A heterogeneidade do consumo também afeta o aprendizado estatístico.
Padrões raros podem ser legítimos, e perfis recorrentes podem sofrer
mudanças por sazonalidade, ocupação, atividade comercial ou geração
distribuída. A comparação por pares, a separação de especialistas e a
análise do histórico reduzem ambiguidades, mas não constituem evidência
causal suficiente. Scores não supervisionados tampouco podem ser
interpretados como probabilidades de fraude sem calibração com rótulos
confiáveis.

Por fim, o valor operacional depende de parâmetros externos ao detector:
custo de deslocamento e inspeção, capacidade mensal, taxas de
confirmação e valores efetivamente pleiteados ou recuperados. A reserva
de até 40% da capacidade e a meta de melhora de 5 pontos percentuais no
rendimento compõem o protocolo de avaliação, não resultados inferidos
das figuras do experimento. Essas restrições delimitam as conclusões
econômicas e preservam a função da ferramenta como apoio técnico à
decisão.

# Conclusão {#conclusao}

Este trabalho apresenta uma metodologia para detecção, localização e
caracterização de padrões anômalos em redes de distribuição a partir de
dados AMI, com ênfase na investigação de perdas não técnicas. A
contribuição não se limita à seleção estatística de observações raras:
compreende a reconstrução da informação de medição no tempo, a
organização de evidências especializadas e sua apresentação em um fluxo
analítico reproduzível.

A plataforma desenvolvida incorpora extração histórica por lotes,
sincronização inteligente, persistência particionada, processamento
colunar, auditoria de continuidade e treinamento por TUI. O histórico
UC--NIO e a separação entre falhas de comunicação e sinais elétricos são
componentes metodológicos fundamentais, pois evitam que problemas de
disponibilidade ou trocas cadastrais sejam confundidos com alterações
reais do comportamento de consumo.

O experimento sintético demonstra que K-Means, PCA residual, Isolation
Forest, LOF e ECOD capturam estruturas diferentes sobre as mesmas
curvas. As figuras de estudo de caso extraídas do Data Lake mostram, por
sua vez, que consumidores com consumo agregado semelhante podem ter
rotinas opostas e que um candidato destacado pelo ensemble precisa ser
analisado em conjunto com seu histórico, a qualidade de telemetria e o
contexto elétrico. Nenhum desses indicadores, isoladamente, confirma
fraude.

A evolução do modelo global para uma arquitetura multivisão é
apresentada como hipótese técnica passível de verificação por ablação e
avaliação de campo. O sistema preserva a distinção entre a detecção de
mudanças, a identificação de raridade, a aplicação de regras físicas e a
atribuição de categorias compatíveis com múltiplas causas. Essa
estrutura favorece decisões auditáveis e a investigação de ocorrências
individuais sem ignorar eventos coletivos de rede.

A conexão com a operação acrescenta um segundo critério de desempenho: a
qualidade de uma lista priorizada depende da capacidade disponível, do
custo de inspeção e do benefício incremental em relação à seleção
convencional. Para isso, o trabalho formaliza funil por especialista,
incidência mensal, backlog, coortes de
repriorização/antecipação/descoberta e valor monetário esperado com
custo de oportunidade. A política de até 40% de substituição e o alvo de
ganho de 5 pontos percentuais no rendimento constituem parâmetros do
protocolo de campo e não resultados já alcançados.

Dessa forma, o principal resultado é uma base integrada de engenharia de
dados e análise de anomalias, com evidências reproduzíveis e critérios
explícitos para avaliação operacional. A confirmação de perdas não
técnicas continua associada à inspeção e ao processo técnico da
distribuidora; a contribuição dos modelos é tornar a seleção de casos
mais informada, rastreável e comparável.
