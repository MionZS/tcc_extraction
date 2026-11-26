# Apostila de Domínio Específico e Guia Completo do Repositório — Plataforma Analítica AMI (TCC Engenharia Elétrica UFPR)

> **Para quem é esta apostila?**  
> Para qualquer pessoa — de um leigo curioso a um engenheiro ou membro de banca — entender **o problema elétrico no mundo real**, **cada técnica de Ciência de Dados e Machine Learning usada**, **exatamente em qual arquivo do repositório está cada peça**, **como as ferramentas se combinam**, **as melhores descobertas (takes) do projeto** e **os usos mais criativos dessa arquitetura**.

---

## 1. O Problema no Mundo Real (Explicado sem Jargão)

### O que é um Medidor Inteligente (AMI)?
Imagine a diferença entre tirar uma **foto por mês** do hidrômetro da sua casa e ter um **relógio cardíaco** medindo seu pulso a cada 5 ou 10 minutos, 24 horas por dia.
- **Medição convencional:** um leiturista passa 1 vez por mês e anota um único número acumulado (kWh do mês). Se algo estranho aconteceu no dia 3 ou se o consumo caiu pela metade no meio do mês, você só descobre semanas depois — e sem saber *em que hora do dia* mudou.
- **AMI (*Advanced Metering Infrastructure*):** medidores inteligentes registram energia, tensão e corrente em intervalos curtos (5, 10 ou 15 minutos) e transmitem esses pacotes por rádio (**rede RF Mesh**) até concentradores e daí para os bancos da distribuidora (**MDM / ORCA**).

### Glossário Essencial do Setor Elétrico (Tradução para Leigos)
| Termo | O que significa na prática | Analogia simples |
|---|---|---|
| **UC** (Unidade Consumidora) | O "CPF da conta de luz" de um imóvel/cliente. | Endereço de entrega. |
| **NIO** (Número de Identificação do Objeto) | O número de série físico do medidor instalado na parede. Uma UC pode trocar de NIO ao longo do tempo! | O número de série do aparelho celular (você troca de aparelho, mas o número da linha continua o mesmo). |
| **FA** (*Forward Active*, kWh) | Energia ativa que a rede **entrega** ao cliente (consumo normal). | Água entrando pelo cano da rua. |
| **RA** (*Reverse Active*, kWh) | Energia ativa que o cliente **injeta de volta** na rede (ex.: painéis solares / Geração Distribuída). | Água sendo bombeada de volta para a rua. |
| **PNT** (Perdas Não Técnicas) | Energia consumida mas não faturada (fraude/desvio, erro cadastral, falha de medição/medidor travado). | Vazamento no cano *antes* do relógio, ou relógio emperrado. |
| **GD** (Geração Distribuída) | Microgeração própria (quase sempre solar fotovoltaica). Quando não homologada no cadastro comercial, aparece como fluxo reverso inesperado. | Cliente gerando energia própria sem avisar a concessionária. |
| **Alimentador / Transformador (Posto)** | A "árvore" elétrica que leva energia da subestação até a rua e às casas. | Avenida principal (alimentador) e rua/esquina (transformador). |
| **PRODIST (Módulo 8)** | Norma da ANEEL que define limites aceitáveis de qualidade de tensão (afundamento, elevação, desequilíbrio entre fases). | Código de trânsito da qualidade de energia. |
| **CIS / GEO / ORCA** | Os 3 mundos de dados da distribuidora: **CIS** = cadastro comercial; **GEO** = mapa elétrico (alimentadores/postes); **ORCA/MDM** = telemetria pesada dos medidores. | Cadastro de clientes, mapa das ruas e central de sensores. |
| **TOI / OS** | Termo de Ocorrência e Inspeção / Ordem de Serviço em campo. **Importante:** este TCC **não** teve acesso a vistorias de campo; o sistema gera *hipóteses priorizadas* para análise humana, nunca "prova de fraude". | Triagem de exames laboratoriais antes do diagnóstico médico presencial. |

### Por que detectar anomalias em AMI é difícil?
1. **Sinal ruim parece fraude:** se o rádio RF Mesh falhar e perder 50% dos pacotes, a soma diária de energia cai pela metade — um algoritmo ingênuo gritaria "desvio!", quando na verdade é só falta de sinal de rádio.
2. **Troca de medidor mistura histórias:** se a UC trocou o medidor NIO-A pelo NIO-B no meio do mês, olhar só pelo NIO quebra o histórico.
3. **Cada cliente tem um perfil:** 80 kWh/dia é normal para uma padaria e absurdo para uma kitnet.
4. **Capacidade de campo é limitada:** de nada adianta apontar 28.000 "suspeitos" se a distribuidora consegue inspecionar no máximo ~2.000 imóveis por mês. É preciso **separar filas** (Telecom, Metrologia, GD, PNT) e **ordenar por prioridade**.

---

## 2. Mapa Completo do Repositório — Onde Está Cada Coisa

### A. Núcleo de Dados, Ingestão e Limpeza (`src/`)
| Arquivo | O que faz | Técnica / Regra principal |
|---|---|---|
| [src/db.py](file:///d:/Projects/tcc_extraction/src/db.py) | Conexão segura com o banco Oracle/ORCA. | SQLAlchemy `QueuePool(pool_size=5, max_overflow=0)` + fechamento explícito de cursores (corrigiu vazamento de 142 sessões abertas). |
| [src/datasets/normalize.py](file:///d:/Projects/tcc_extraction/src/datasets/normalize.py) | Normaliza tabelas brutas para `ami_interval`, `ami_instantaneous`, `ami_registers`. | Detecção automática de cadência (5, 10 ou 15 min) e cálculo exato de cobertura por denominador nativo. |
| [src/datasets/retroactive_cleaner.py](file:///d:/Projects/tcc_extraction/src/datasets/retroactive_cleaner.py) | **Coração da engenharia de dados:** reconstrói o vínculo histórico `UC ↔ NIO` por vigência temporal e gera features diárias e de janela (30 dias) sem vazamento futuro. | Polars vetorizado + saneamento retroativo + flags `meta__is_clean_for_train` e `meta__data_quality`. |
| [src/datasets/build_training_dataset.py](file:///d:/Projects/tcc_extraction/src/datasets/build_training_dataset.py) | Junta features de janela com cadastro CIS (`uc_context`) e topologia GEO (`electrical_hierarchy`). | Separação estrita de prefixos (`id__*`, `meta__*`, `x__*`, `y__*`) para impedir vazamento de variáveis administrativas no ML. |
| [src/features/meter_day.py](file:///d:/Projects/tcc_extraction/src/features/meter_day.py) & [src/features/electrical.py](file:///d:/Projects/tcc_extraction/src/features/electrical.py) | Calcula métricas físicas diárias por medidor. | FA/RA sum, fator de carga, taxa de reversão, desequilíbrio de tensão e corrente sob carga (`>= 1 A`), rampas, autocorrelação lag-1, hora de pico cíclica (`sin/cos`). |
| [src/features/uc_window.py](file:///d:/Projects/tcc_extraction/src/features/uc_window.py) | Agrega 30 dias em estatísticas robustas até a data de corte (*cutoff*). | Mediana, MAD (*Median Absolute Deviation*), slope linear OLS, dias confiáveis, idade do medidor, troca em 30/90d. |
| [src/features/neighborhood.py](file:///d:/Projects/tcc_extraction/src/features/neighborhood.py) & [src/features/circuit_profile.py](file:///d:/Projects/tcc_extraction/src/features/circuit_profile.py) | Calcula contraste espacial entre vizinhos do mesmo transformador/alimentador e perfil do circuito (ex.: penetração de GD). | Z-score dentro do transformador e mediana de cobertura dos pares. |
| [src/events/alarm_extractor.py](file:///d:/Projects/tcc_extraction/src/events/alarm_extractor.py) & [src/events/alarm_categories.py](file:///d:/Projects/tcc_extraction/src/events/alarm_categories.py) | Extrai e categoriza alarmes do medidor (tampa aberta, falta de energia, comunicação offline, corrente reversa). | Classificação por regex/padrões sobre `CONTENT` em partições Parquet (`output/events/alarm_events/`). |

### B. Modelos, Especialistas e Regras de Negócio (`src/models/` e `src/eda/`)
| Arquivo | O que faz | Técnica principal |
|---|---|---|
| [src/models/composite.py](file:///d:/Projects/tcc_extraction/src/models/composite.py) | **Comitê Elétrico de 6 Detectores (`CompositeAnomalyDetector`)** + isolamento estrito de features elétricas puras (`select_pure_electrical_features`). | IForest + LOF (`log1p`) + ECOD + COPOD + HBOS + RobustPCA (`StandardScaler`) + normalização congelada `frozen_quantile` (`P01–P99`) + guardrails PRODIST (`strict_voltage_imbalance_rule` na `v2`). |
| [src/models/specialized_detectors.py](file:///d:/Projects/tcc_extraction/src/models/specialized_detectors.py) | **Orquestrador Multivisão (`MultiViewAnomalyPipeline`):** separa `S_comm`, `S_elec`, `S_tamper` e `BehavioralClusterer`. | Gate de qualidade (`cov >= 0,70`, e `reliable_days >= 18` na `v2`) que impede telemetria ruim de virar anomalia elétrica. |
| [src/eda/uc_timeseries.py](file:///d:/Projects/tcc_extraction/src/eda/uc_timeseries.py) | Séries temporais diárias/horárias e **Detector de Mudança de Regime (`_detect_step_change_cusum`)**. | CUSUM (soma acumulada centrada) + teste de magnitude relativa à escala robusta MAD para localizar exatamente o dia de um degrau de consumo. |
| [src/models/moe_router.py](file:///d:/Projects/tcc_extraction/src/models/moe_router.py) & [src/models/hierarchical_trainer.py](file:///d:/Projects/tcc_extraction/src/models/hierarchical_trainer.py) | **Mixture of Experts (MoE) Hierárquico:** pondera especialista do alimentador local vs. modelo global conforme volume de amostras e penetração solar. | Encolhimento empírico (*shrinkage*): circuitos grandes confiam 70–88% no modelo local; circuitos pequenos apoiam-se no modelo global. |
| [src/models/spurious_filter.py](file:///d:/Projects/tcc_extraction/src/models/spurious_filter.py) | Filtra perturbações coletivas de rede (ex.: desligamento/manobra no transformador) e ruídos não isolados. | Se `>= 60%` dos vizinhos no mesmo transformador ficaram anômalos simultaneamente → reclassifica como `GRID_DISTURBANCE` e atenua o escore em 65%. |
| [src/models/category_reporting.py](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py) | **Árvore de Triagem Delimitada (9 categorias não sobrepostas)** + contraste espacial Telecom (Posto → grade 250m → Alimentador) + transições de classe comportamental (`C1..C6`). | Separa fisicamente filas de Telecom, Ciclo de Vida (UC Nova / Troca), GD não cadastrada, Medidor Travado e Anomalia Elétrica/PNT. |
| [src/utils/censorship.py](file:///d:/Projects/tcc_extraction/src/utils/censorship.py) | Blindagem LGPD automática de UCs (`908***00`), NIOs (`NIO-435***63`) e alimentadores (`Circuito F-01..F-23`). | Mascaramento determinístico sem expor clientes ou nomes comerciais em figuras e relatórios. |

### C. Experimentos, Interfaces e Testes (`scripts/`, `src/experiments/`, `tests/`)
| Arquivo | O que faz |
|---|---|
| [src/experiments/oot_pipeline.py](file:///d:/Projects/tcc_extraction/src/experiments/oot_pipeline.py) & [src/experiments/daily_rules.py](file:///d:/Projects/tcc_extraction/src/experiments/daily_rules.py) | Runners oficiais da campanha congelada de 25/11 (`OOT_MULTIVIEW`, `DAILY_RULES`, `ABLATION_M0_M5`, `SCORE_STABILITY`). |
| [scripts/audit_experimental_results.py](file:///d:/Projects/tcc_extraction/scripts/audit_experimental_results.py) | **Auditoria P0:** reconciliação populacional, auditoria de correlações, gating congelado (sem *refit*), sensibilidade de pesos (23 configs), sensibilidade do bônus `+0,15` e matriz de especialistas. |
| [scripts/run_multiview_evaluation.py](file:///d:/Projects/tcc_extraction/scripts/run_multiview_evaluation.py) | **Avaliação Multivisão P1 (Base A):** executa todos os especialistas sobre as 65.686 UCs OOT (incluindo 16 modelos MoE por alimentador sem vazamento e CPD em lote), ablação `V0–V5` sob `K = 2.000`, estudos de caso e projeção operacional. |
| [scripts/run_controlled_perturbations.py](file:///d:/Projects/tcc_extraction/scripts/run_controlled_perturbations.py) | **Perturbações Controladas P1 (Base B):** injeta 23 cenários fisicamente coerentes em cópias isoladas de 100 séries AMI reais (433 mil medições) antes da extração de features para medir detecção, erro temporal CPD e isolamento de eventos coletivos. |
| [scripts/run_v2_temporal_validation.py](file:///d:/Projects/tcc_extraction/scripts/run_v2_temporal_validation.py) | **Validação Temporal Multi-Corte (`19/11 → 25/11`) e Variante `v2` Saneada:** testa persistência real de +6 dias, ativa 1,7M de alarmes reais no `S_tamper` e prova que o gate `v2` zera vazamentos de blocos consecutivos ausentes. |
| [src/tui/tui_app.py](file:///d:/Projects/tcc_extraction/src/tui/tui_app.py) & [scripts/run_web_cockpit.py](file:///d:/Projects/tcc_extraction/scripts/run_web_cockpit.py) | **Interfaces operacionais:** Terminal UI (Textual/Rich) para governança/extração/auditoria e Cockpit Web para raio-X individual de UCs, séries temporais e filas priorizadas. |
| [tests/test_p0_p1_experimental_closure.py](file:///d:/Projects/tcc_extraction/tests/test_p0_p1_experimental_closure.py) (e 44 outros arquivos em `tests/`) | Garante matematicamente zero vazamento temporal, soma unitária dos pesos, conservação de partições, anonimização LGPD e invariantes das figuras/relatórios (**67/67 testes nucleares aprovados**). |

---

## 3. Como Cada Técnica de ML Funciona (Explicação Intuitiva)

### Por que usar 6 detectores elétricos juntos em vez de 1?
Imagine 6 inspetores com "lentes" completamente diferentes olhando a mesma folha de dados elétricos (16 variáveis puras, sem nenhuma variável de comunicação):

1. **Isolation Forest (`IF`, peso 0,22):** *O cortador aleatório.* Tenta isolar cada cliente fazendo cortes aleatórios nos dados. Quem fica sozinho em poucos cortes é um ponto extremo global. **Ponto fraco:** sofre de *masking* quando vários fraudadores/anômalos formam um pequeno grupinho parecido.
2. **Local Outlier Factor (`LOF`, peso 0,20):** *O comparador de vizinhos.* Pergunta: "A densidade ao redor desta UC é muito menor que a dos seus 20 vizinhos mais parecidos?" **Superpoder:** encontra anomalias locais e microgrupos que o IF deixa passar. *(Por isso removê-lo trocou 180 vagas no Top-2.000!)*
3. **ECOD (peso 0,18):** *O fiscal das pontas.* Sem assumir distribuição normal, olha em qual percentil extremo (cauda esquerda ou direita) cada variável do cliente caiu.
4. **COPOD (peso 0,18):** *O fiscal das pontas assimétricas.* Usa cópulas empíricas corrigidas por assimetria (*skewness*) — perfeito para energia elétrica (que tem cauda longa à direita) e desequilíbrios.
5. **HBOS (peso 0,10):** *O contador de histogramas.* Ultrarrápido; pontua quem cai em faixas muito raras (ex.: consumo quase zero com tensão normal).
6. **Robust PCA (peso 0,12):** *O detector de quebra de correlação.* Aprende as relações lineares normais entre variáveis (ex.: alta energia anda junto com certa corrente e demanda). Se alguém "quebra a física" dessas correlações, o erro de reconstrução dispara.

### Por que a Normalização Congelada (`frozen_quantile`) é crucial?
Se você normalizar um lote usando `min` e `max` do próprio dia (`batch_minmax`), basta **1 único medidor defeituoso** com valor absurdo para esmagar o escore de todas as outras 65.685 UCs!  
Com `frozen_quantile`, fixamos os limites `P01` e `P99` **no período de treino (Ago–Out)**. Nos testes de estabilidade (`score_stability`), remover o Top-1% mais extremo causou deslocamento de até **0,2291** no `batch_minmax`, contra **0,000000** (estabilidade absoluta!) no `frozen_quantile`.

### Como os Especialistas se Complementam?
- **`S_comm` + Gate de Qualidade:** "Os dados são confiáveis para julgar física?" Se não → **Quarentena Telecom** (e ainda compara com os vizinhos do transformador/250m para dizer se o defeito é só na antena daquela casa ou no bairro inteiro!).
- **`S_elec` (Transversal):** "Hoje, olhando os últimos 30 dias, esta UC parece esquisita comparada às outras 65 mil?"
- **`CPD / CUSUM` (Longitudinal):** "Esta UC *mudou bruscamente* em relação a **si mesma** em algum dia específico?" *(No experimento controlado, localizou quedas de -50% com erro mediano de **0,0 dia**, enquanto `S_elec` sozinho mal via quedas de -20% a -30%!)*
- **`S_behavior` (Intradiário 24h):** "O consumo total diário não mudou nada, mas a rotina de horários mudou?" *(Detectou deslocamento de +4h mantendo ΔkWh/dia = 0,0!)*
- **`S_tamper` (Eventos Físicos):** "O próprio medidor gritou alarme de tampa aberta, remoção de módulo ou corrente reversa?"
- **`MoE` + `SpuriousFilter` (Topologia):** "Isso é uma anomalia desta casa ou o transformador inteiro sofreu uma queda junto?"

---

## 4. Combinações de Ferramentas (As "Receitas" que Fazem Tudo Rodar Rápido)

| Objetivo | Combinação de Ferramentas | Por que funciona tão bem |
|---|---|---|
| **Processar 117 dias × 8,8 milhões de linhas/dia em um notebook** | **Polars (LazyFrame + predicate pushdown) + Parquet particionado (`year/month/day`)** | Carregar 100 UCs ao longo de 30 dias (433 mil linhas de um universo de ~260 milhões) levou **3,0 segundos** sem estourar memória! |
| **Extrair do Oracle (ORCA) sem derrubar o banco** | **SQLAlchemy `QueuePool(size=5, max_overflow=0)` + `ThreadPoolExecutor(5)` + `result.close()`** | Mantém paralelismo real de 5 consultas reutilizando exatamente 5 sessões em vez de vazar 142 cursores abertos. |
| **Persistir modelos pesados sem quebrar o Git/disco** | **Joblib + `joblib_splitter.py` (chunks de 25 MB comprimidos)** | Permite versionar, carregar e congelar o `MultiViewAnomalyPipeline` inteiro (`frozen_multiview_model.joblib`) com integridade SHA-256. |
| **Avaliar algoritmos sem depender de vistorias de campo** | **Injeção controlada de perturbações sobre cópias em memória de séries reais + `pytest` de invariantes** | Mede observabilidade, atraso temporal e vazamento de lacunas sem tocar 1 byte do Data Lake e sem criar curvas sintéticas artificiais. |
| **Operação humana segura (LGPD)** | **Cockpit Web + TUI Rich + `censorship.py` + contrato Parquet com `NULL` explícito** | O analista vê o raio-X completo (curva 5 min, CUSUM, radar de detectores, vizinhança) sem expor dados sensíveis em relatórios externos. |

---

## 5. Os 8 Melhores "Takes" (Descobertas Reais de Engenharia e Ciência do Projeto)

1. **O mistério da cobertura de 50% vs. 99,49% (UC `685***20`):** O painel antigo dividia 6 leituras/hora por 12 slots de uma grade fixa de 5 min → parecia 50% de perda de pacotes! Na verdade o medidor operava em cadência nativa de **10 minutos** (`6/6 = 100%` nas 168h; `4.298/4.320 = 99,49%` em 30 dias). **Take:** *nunca assuma cadência fixa em parques AMI heterogêneos.*
2. **O colapso do LOF (`1e10`) por empates exatos:** Medidores idênticos (ex.: consumo zero) geravam distância k-NN igual a zero e escore LOF `~10¹⁰`, esmagando toda a normalização (`ρ = 0,02`). Aplicar `log1p(max(-score - 1, 0))` resgatou o LOF (`ρ = +0,46`), transformando-o no detector mais complementar do comitê.
3. **PCA sem padronização interna (`StandardScaler`):** Como a potência instalada (kVA) e consumo (kWh) tinham escalas muito maiores que desequilíbrio (0–2), a PCA reconstruía apenas a variável de maior variância e gerou correlação negativa (`-0,20`) antes do saneamento. Padronizar internamente corrigiu para `+0,41`.
4. **O bug silencioso do bônus `+0,15` (descoberto na auditoria P0):** A regra `v_imb_cols` buscava `"IMBALANCE_MAX"` e acabou pegando **desequilíbrio de corrente** (`> 0,20`), dando `+0,15` para **46,7% da cidade inteira (30.694 UCs)** em vez de apenas **0,19% (124 UCs)** de violações reais de tensão/reversão! Corrigir na `v2` manteve 92,4% do Top-2.000 e **aumentou** a persistência temporal de 6 dias para **91,5%**.
5. **Os 1,7 milhão de alarmes "invisíveis" (descoberto na auditoria P1):** O pipeline de 25/11 lia `alarm_events.parquet` (arquivo único), enquanto os alarmes estavam em **41 partições** dentro de `output/events/alarm_events/`. Ler particionado revelou **1.377 UCs** com `S_tamper >= 0,70` (496 concordantes com anomalia elétrica e 881 sinais complementares exclusivos).
6. **A brecha dos 15 dias de silêncio consecutivo (descoberta nas perturbações P1):** Um medidor que ficou 15 dias mudo teve média de 99% nos 15 dias em que transmitiu — enganando o gate baseado só em `avg_interval_coverage >= 70%` (31% de vazamento na `v1`)! Combinar `cov >= 70%` **E** `reliable_days >= 18` na `v2` zerou o vazamento para **0,0% (100% em quarentena)**.
7. **Separação espacial de falhas de Telecom:** Cruzar a cobertura da UC com a mediana do mesmo transformador (ou grade geográfica de 250 m) permitiu separar automaticamente **910 falhas isoladas de medidor/antena** de **1.829 casos em gargalo coletivo RF Mesh** — evitando enviar equipe comercial para problema de repetidor de rádio.
8. **Persistência real de +6 dias (`19/11 → 25/11`):** Sem precisar de TOI, os 6 dias adicionais de dados provaram que **89% a 92% da carteira priorizada persiste** de uma semana para outra (`Spearman 0,95–0,97`), provando que o algoritmo captura fenômenos estruturais persistentes e não ruído aleatório de um dia.

---

## 6. Usos Mais Criativos da Plataforma (Além de Fiscalização de Fraude)

A mesma arquitetura construída neste repositório pode ser reaproveitada imediatamente para pelo menos **6 aplicações de alto valor** em distribuidoras de energia:

1. **Caça à Geração Distribuída (GD) Não Homologada e Proteção de Rede:** Detectou **100%** das injeções solares reversas no experimento controlado e identificou **1.451 UCs** com fluxo reverso no OOT — essencial para evitar inversão de fluxo em transformadores, sobretensão local e riscos de segurança para eletricistas em manutenção.
2. **Auditoria Automática de Cadastro Comercial (CIS × Campo):** Como visto no caso real `291***96` (medidor constava removido em `11/06/2026`, mas transmitiu ~28 kWh/dia até `05/11`!), o cruzamento temporal `UC–NIO` encontra "medidores fantasmas", ligações ativas sem faturamento e cadastros defasados.
3. **Manutenção Preditiva da Rede de Telecomunicações RF Mesh:** O mapa de `GARGALO_COLETIVO_RF_MESH` por transformador/grade de 250 m funciona como um **radar de saúde de concentradores e repetidores**, direcionando equipes de automação antes que a área inteira fique cega.
4. **Detector de Medidores Travados e Defeitos Metrológicos:** A regra física `FA ~ 0` com `U_L1 > 100 V` e comunicação ativa identifica bobinas de corrente abertas ou registradores congelados (515 UCs no OOT) sem confundir com falta de energia.
5. **Radar de Eventos Coletivos e Contingências em Transformadores:** O `SpuriousFilter` (16/16 eventos coletivos separados no experimento) pode ser invertido para monitorar **quedas de carga ou afundamentos simultâneos por transformador**, ajudando o Centro de Operação (COD) a detectar sobrecargas, fases abertas ou interrupções parciais em baixa tensão.
6. **Planejamento de Capacidade de Equipes sob Orçamento Fixo:** O módulo de projeção operacional (`K = 2.000/mês`) permite simular diferentes políticas de despacho (por severidade estatística, por materialidade em kWh/dia, por persistência CPD ou por qualidade de evidência), transformando ML em ferramenta direta de gestão de carteira.

---

## 7. Transparência Total para a Banca: O que é Derivado vs. Heurístico

Para nunca ser pego de surpresa na arguição, tenha esta divisão na ponta da língua:

- **O que é estritamente empírico / calculado dos dados:** normalização `P01–P99` no treino, limiar `P95` (`0,5131`), todas as correlações, reconciliações, persistência `19/11 → 25/11`, matriz de perturbações controladas e ablação `V0–V5`.
- **O que é escolha heurística inicial de projeto (não otimizada por falta de rótulos de campo):**
  - Pesos do ensemble (`0,22 / 0,20 / 0,18 / 0,18 / 0,10 / 0,12`) → **validados por sensibilidade** (±20% muda `< 1,4%` do Top-2.000; uniformes têm `93,6%` de sobreposição).
  - Contaminação nominal (`5%`), bônus `+0,15`, pesos de `S_tamper` (`0,70/0,30/0,15`), pontos de corte do `MoE` (`150` e `1.000` amostras) e pesos aditivos da variante P1 `V5` (`+0,06/+0,04/+0,05`) → **escolhas de projeto explicáveis fisicamente**, cuja calibração ótima depende de rótulos independentes no piloto prospectivo (Cap. 14).
  - Premissas econômicas (`K=2.000`, `40%`, `+5 p.p.`, `R$ 180 / R$ 1.800`) → **cenário paramétrico hipotético**, jamais resultado medido.
