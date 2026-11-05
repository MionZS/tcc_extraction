# Mudança de Paradigma Metodológico: Saneamento Retroativo de Sinais, Reconciliação Topológica e Janelas Desacopladas em Redes AMI

**Projeto:** Extração, Normalização e Detecção Não-Supervisionada de Anomalias em Smart Grids (Copel Distribuição — Araucária/PR)  
**Camada Arquitetural:** Curadoria Retroativa Pós-Extração e Pré-Modelagem (`src/datasets/retroactive_cleaner.py`)  
**Unidade Amostral Saneada:** $\text{UC} \times \text{cutoff\_date}$ sob janelas desacopladas de ciclo fechado ($W \in \{15, 30\}\text{ dias}$)  
**Data:** Novembro de 2026  

---

## 1. Fundamentação Epistemológica: Por que Modelos Não-Supervisionados Exigem Condicionamento do Espaço de Suporte

Em problemas clássicos de aprendizado supervisionado, a disponibilidade de rótulos confiáveis $Y \in \{0, 1\}$ permite que algoritmos discriminativos (como *Gradient Boosted Trees* ou redes neurais) ponderem automaticamente a relevância de cada atributo, suprimindo variáveis ruidosas que apresentem baixa informação mútua com o alvo $P(Y \mid \mathbf{X})$. No contexto real da distribuição de energia elétrica brasileira, contudo, a modelagem supervisionada de perdas não-técnicas (PNT) sobre telemetria de Medição Inteligente (*Advanced Metering Infrastructure* — AMI) é inviabilizada pela escassez severa de rótulos sincrônicos: procedimentos irregulares (PIs) historicamente lavrados carecem de memória de massa preservada no instante de início da fraude, e o intervalo entre a instalação da irregularidade e sua constatação em campo supera frequentemente o horizonte operacional do histórico de telemetria de alta frequência.

Sob o paradigma **estritamente não-supervisionado** (implementado neste trabalho via comitê composto por *Isolation Forest*, *Local Outlier Factor* — LOF, *Empirical Cumulative Distribution-based Outlier Detection* — ECOD e erro de reconstrução espectral via PCA, orquestrados hierarquicamente por alimentador via *Mixture of Experts* — MoE), o estimador não aprende uma fronteira de decisão condicionada a fraudes; ele estima a **geometria da densidade empírica conjunta** $p(\mathbf{x})$ de todas as fontes de variância presentes na matriz de atributos $\mathbf{X}$.

Em uma infraestrutura AMI real operando sobre comunicação sem fio em malha (*RF Mesh*), a distribuição bruta observada $P_{\text{raw}}(\mathbf{x})$ não reflete puramente o comportamento eletrocomercial das Unidades Consumidoras (UCs), mas sim uma mistura de quatro processos geradores distintos:

$$P_{\text{raw}}(\mathbf{x}) = \pi_1 P_{\text{rede}}(\mathbf{x}) + \pi_2 P_{\text{anom}}(\mathbf{x}) + \pi_3 P_{\text{telecom}}(\mathbf{x}) + \pi_4 P_{\text{artefato}}(\mathbf{x})$$

onde:
1. $P_{\text{rede}}(\mathbf{x})$ representa o regime normal de carga, sazonalidade semanal e impedância topológica do circuito secundário/primário;
2. $P_{\text{anom}}(\mathbf{x})$ representa as anomalias eletrocomerciais de interesse operacional (derivações clandestinas, *bypass* de fase, inversão e adulteração metrológica);
3. $P_{\text{telecom}}(\mathbf{x})$ representa falhas de completude da rede *RF Mesh* (sombra de rádio, saturação de rotas multi-salto, reinicialização de concentradores e recepção parcial de pacotes de 5 minutos);
4. $P_{\text{artefato}}(\mathbf{x})$ compreende distorções determinísticas de integração entre sistemas legados (divergência lexicográfica de identificadores entre sistemas comerciais e coletores de medição) e singularidades matemáticas de grandezas físicas em regime de carga nula.

Quando a amplitude de variância induzida por $P_{\text{telecom}}$ e $P_{\text{artefato}}$ supera em ordens de grandeza os desvios sutis de $P_{\text{anom}}$, qualquer detector não-supervisionado ajustado diretamente sobre os dados brutos converge inevitavelmente para modelar artefatos de comunicação e falhas de vínculo cadastral como se fossem anomalias severas, ou degenera suas partições de densidade. Portanto, a introdução de uma **etapa retroativa e formal de saneamento de sinais e reconciliação de entidades** entre o *Data Lake* bruto e a matriz de treinamento não constitui um ajuste *ad hoc* de hiperparâmetros, mas uma **condição matemática de identificabilidade** para que o suporte de treinamento $\mathcal{X}_{\text{clean}}$ represente legitimamente o domínio elétrico.

---

## 2. Diagnóstico Forense Empírico sobre o Parque de Araucária ($N = 63.356$ Medidores)

A primeira bateria experimental em escala municipal completa sobre os `63.356` medidores inteligentes ativos de Araucária revelou sintomas clássicos de degenerescência distribucional nos especialistas locais por alimentador: circuitos específicos apresentaram taxas aparentes de anomalia de exatamente `100,0%` (*Angélica*, *Gonvarri*) ou `50,0%` (*CSN*), contrastando com a taxa nominal de contaminação parametrizada ($\alpha \in [0,001; 0,01]$).

A auditoria forense conduzida sobre as camadas de contexto (`uc_context`, `electrical_hierarchy`, `meter_installation_history`), partições diárias (`uc_day_features`) e matriz consolidada (`training_dataset`) demonstrou que o problema decorria da interação simultânea de quatro mecanismos estruturais nos dados extraídos:

### 2.1. Assimetria Lexicográfica de Identificadores entre Sistemas Legados (`CIS/GEO` vs. `ORCA/MDM`)

Na arquitetura corporativa da concessionária, o sistema comercial e georreferenciado (`CIS` / `GEO`) armazena o número serial do medidor (`NIO`) com preenchimento fixo de zeros à esquerda em 10 caracteres (e.g., `"0043309810"`), vinculado ao código identificador da Unidade Consumidora (`UC`, de 7 a 9 dígitos, e.g., `"100000070"`). Em contrapartida, o sistema coletor de telemetria (`ORCA` / `MDM`) registra as séries de memória de massa e eventos sob a representação numérica suprimida de 8 caracteres (`"43309810"`).

Na ausência de normalização canônica na ponte temporal `NIO → UC`:
- O dicionário de tradução falhou silenciosamente em `100%` das buscas diretas, propagando o próprio `NIO` de 8 dígitos para o campo `id__uc_id`;
- No *join* subsequente contra a hierarquia elétrica (`electrical_hierarchy.UC`), **`99,985%` das amostras de treinamento (`41.174` de `41.180` linhas) ficaram órfãs de topologia** (`ALIMENTADOR = NULL`, classe tarifária nula, tensão nominal nula e histórico de substituição nulo);
- Apenas **`6` registros em todo o conjunto de treino** obtiveram correspondência aparente por mera **colisão numérica acidental** — casos em que o número serial de 8 dígitos de um medidor coincidia aritmeticamente com o código `UC` de um consumidor distinto pertencente a outro ponto da rede (*Angélica*: 1 amostra = `100%` anômala; *Gonvarri*: 1 amostra = `100%` anômala; *CSN*: 2 amostras = `50%` anômala; *Almeida Torres*: 1 amostra = `0%`; *Fonte Nova*: 1 amostra = `0%`). Como nenhum alimentador atingiu a cardinalidade mínima de segurança ($n_f \ge 30$) para instanciar seu especialista local, todos recaíram no escore global sem calibração de circuito.

### 2.2. O Dilema da Escala Temporal: Colapso de Snapshot Único vs. Autocorrelação por Janela Deslizante Diária

A investigação sobre o dimensionamento amostral confirmou que o treinamento estava simultaneamente exposto a dois extremos metodológicos inadequados para séries temporais em *Smart Grids*:

1. **Sub-agregação por Snapshot Diário Único (Degenerescência de Variância Intra-Janela):** Embora cerca de 90 dias de telemetria bruta houvessem sido extraídos para o *Data Lake*, a rotina final de consolidação materializava `uc_day_features` apenas para a última data processada (`2026-10-28`) e a submetia isoladamente ao agregador de janela de 30 dias. Com apenas $1$ observação diária por UC dentro da janela $(t_c - 30, t_c]$, todas as métricas de dispersão robusta e dinâmica temporal colapsaram identicamente a zero em `100%` das `63.356` UCs:
   $$\text{MAD}_i = \text{mediana}\left(|E_{i,d} - \tilde{E}_i|\right) \equiv 0, \qquad \beta_{\text{trend}, i} \equiv 0$$
   Com colunas inteiras constantes em zero e atributos cadastrais nulos, árvores de isolamento (*Isolation Forest*) realizam cortes degenerados sobre ruído residual de ponto flutuante, enquanto matrizes de covariância do PCA perdem posto (*rank deficiency*).

2. **Inviabilidade Estatística de Janelas Deslizantes Diárias (`stride = 1 dia`):** A alternativa ingênua de deslizar uma janela de $W = 30\text{ dias}$ passo a passo a cada dia ao longo dos 92 dias de Agosto a Outubro produziria $63.356 \times 92 \approx 5,83\text{ milhões}$ de vetores nos quais janelas consecutivas da mesma UC compartilham $\frac{29}{30} = 96,7\%$ das mesmas leituras físicas. Tal super-amostragem destrói a premissa de amostras independentes e identicamente distribuídas (i.i.d.) exigida pelos estimadores de densidade: no algoritmo LOF ($k = 20$), os $20$ vizinhos mais próximos de qualquer UC no espaço de atributos passariam a ser as próprias réplicas temporais dessa mesma UC nos dias adjacentes ($t-10, \dots, t+10$), mascarando anomalias persistentes (cuja densidade local aparentaria ser alta devido ao auto-agrupamento) além de inviabilizar o custo quadrático de busca espacial.

| Abordagem de Janelamento | Amostras (3 Meses) | Sobreposição Intra-UC | Dispersão (`MAD`) e `Slope` | Propriedade no LOF / IsoForest |
| :--- | :---: | :---: | :---: | :--- |
| **Snapshot Único (1 dia)** | $63.356$ | N/A (1 dia real) | Degenerados (`100% = 0,0`) | Colapso de variância; modela ruído instantâneo |
| **Deslizante Diária (`stride=1d`)** | $\approx 5.828.000$ | $96,7\%$ ($29/30$ dias) | Definidos, porém redundantes | Auto-vizinhança espúria no LOF; violação severa de i.i.d. |
| **Canônica Mensal (`30d`, `stride=1m`)** | $\approx 190.000$ | **$0,0\%$ (Desacoplada)** | **Plenos e estacionários (4 semanas)** | **Independência temporal + fidelidade ao ciclo de faturamento** |
| **Canônica Quinzenal (`15d`, `stride=15d`)** | $\approx 380.000$ | **$0,0\%$ (Desacoplada)** | **Plenos (2 ciclos semanais)** | **Maior granularidade sem vazamento nem sobreposição** |

### 2.3. Singularidades Eletrofísicas em Regime de Espera (*Standby*) e Canais Desenergizados

A inspeção das grandezas elétricas diárias evidenciou três fontes adicionais de contaminação física:
- **Saturação de desequilíbrio de corrente na madrugada:** Em unidades consumidoras residenciais bifásicas (`BI`) ou trifásicas (`TRI`), durante a madrugada o consumo reduz-se a cargas parasitas em uma única fase (e.g., $I_{L1} = 0,25\text{ A}, I_{L2} = 0,00\text{ A}$). A tomada do operador máximo estrito (`max`) ao longo dos 288 intervalos diários resultava em `CURRENT_IMBALANCE_MAX = 1,0` para a quase totalidade da população residencial normal, destruindo o poder discriminativo desse atributo para detectar derivações reais sob carga.
- **Falsos desequilíbrios de tensão de `100%`:** Em medidores bifásicos cujo terceiro canal registra $0,0\text{ V}$ (ou durante *dropouts* transitórios de leitura de canal), a inclusão de fases com tensão nula na média trifásica produzia `VOLTAGE_IMBALANCE_MAX = 100,0%`.
- ** Divergência de nomenclatura na memória de massa (`U_L1_AVG` vs. `U_L1`):** A camada intervalar armazena tensões médias de 5 minutos sob o sufixo `_AVG`, fazendo com que consultas estritas por `U_L1` retornassem zero para mediana e dispersão de tensão.

### 2.4. Viés de Sobrevivência (*Survival Bias*) e Queda Aparente de Consumo na Rede *RF Mesh*

Em redes *RF Mesh* densas, medidores situados em bordas topológicas ou zonas de obstrução sofrem perda intermitente de pacotes. Verificou-se que:
1. Dias com recepção parcial (e.g., apenas $12$ de $288$ intervalos recebidos, $\text{cov}_{i,d} = 4,2\%$) tinham sua energia ativa somada normalmente (`FA_INTERVAL_SUM` próximo de zero), simulando uma queda brusca de consumo e inflando artificialmente o `MAD` e o `TREND_FA_SLOPE` negativo — mimetizando a assinatura matemática exata de uma fraude;
2. O cálculo anterior de `AVG_INTERVAL_COVERAGE` tomava a média aritmética apenas sobre os dias **presentes** na partição da UC: um medidor silencioso durante 29 dias do mês que transmitisse 1 único dia completo recebia `AVG_INTERVAL_COVERAGE = 1,0` (*survival bias*), ingressando no conjunto de treinamento como se possuísse telemetria perfeita.

---

## 3. Especificação Formal do Pipeline de Saneamento Retroativo (`RetroactiveDataCleaner`)

Para solucionar integralmente essas distorções sem alterar ou descartar os arquivos brutos já extraídos do *Data Lake* (`output/measurements/`, `output/events/`, `output/context/`), instituiu-se uma camada determinística e vetorizada de **Curadoria Retroativa** (`src/datasets/retroactive_cleaner.py`), estruturada em cinco estágios sequenciais:

```text
   DATA LAKE BRUTO (IMUTÁVEL)                  CURADORIA RETROATIVA                     MATRIZES DE SAÍDA
 ┌────────────────────────────┐      ┌────────────────────────────────────────┐
 │ measurements/ami_interval  │─────▶│ 1. Reconciliação Canônica NIO(8/10)↔UC │
 │ measurements/instantaneous │      │ 2. Filtragem Física (U>=50V, I>=1.0A)  │      ┌──────────────────────────┐
 │ measurements/ami_registers │─────▶│ 3. Isolamento RF Mesh (cov >= 0.70)    │─────▶│ model_input/v1/          │
 │ events/alarm_events        │      │ 4. Janelas Desacopladas (30d / 15d)    │      │ training_dataset.parquet │
 │ context/CIS + GEO          │─────▶│ 5. Portão IS_CLEAN_FOR_TRAIN           │      └────────────┬─────────────┘
 └────────────────────────────┘      └───────────────────┬────────────────────┘                   │
                                                         │                                        ▼
                                                         ▼                             Treino Limpo (Fit) +
                                            ┌─────────────────────────────┐            Avaliação OOT Sem Ruído
                                            │ reports/telecom_quarantine_ │
                                            │ uc_windows.parquet (.csv)   │
                                            └─────────────────────────────┘
```

### Estágio 1: Reconciliação Canônica Temporal de Entidades (`NIO ↔ UC ↔ Alimentador`)

Define-se o operador de normalização lexicográfica $\nu(s) = \text{lstrip}_0(\text{trim}(s))$. A partir do histórico de instalação de medidores (`meter_installation_history`) e da hierarquia elétrica (`electrical_hierarchy`), constrói-se uma relação temporalmente indexada:

$$\mathcal{B} = \left\{ \big(\nu(\text{NIO}_k),\, \nu(\text{UC}_k),\, t_{\text{inst}, k},\, t_{\text{ret}, k}\big) \right\}$$

Para toda medição registrada pelo medidor $m$ na data calendário $d$, a entidade analítica $\text{UC}$ é resolvida pela correspondência exata sobre $\nu(m)$ respeitando a vigência física do equipamento $t_{\text{inst}, k} \le d \le t_{\text{ret}, k}$ (com *fallback* determinístico para o vínculo cadastral ativo quando datas históricas estiverem ausentes). Adicionalmente, partições legadas em cache de `uc_day_features` onde se detecte a patologia $\text{UC} == \text{NIO}$ são automaticamente invalidadas e recalculadas.

### Estágio 2: Condicionamento Eletrofísico de Grandezas de Fase

1. **Desequilíbrio de Tensão sob Fases Energizadas:** Em cada instante $t$, avaliam-se apenas as fases efetivamente energizadas $\Phi_U(t) = \{\phi \in \{L_1, L_2, L_3\} : U_\phi(t) \ge 50,0\text{ V}\}$. Somente quando $|\Phi_U(t)| \ge 2$ calcula-se o desvio percentual máximo em relação à média das fases ativas $\bar{U}(t) = \frac{1}{|\Phi_U(t)|}\sum_{\phi \in \Phi_U(t)} U_\phi(t)$:
   $$\delta_U(t) = \frac{\max_{\phi \in \Phi_U(t)} |U_\phi(t) - \bar{U}(t)|}{\bar{U}(t)} \times 100\%$$
   Adota-se o quantil $P_{95}(\delta_U)$ diário e intra-janela em substituição ao máximo pontual absoluto, eliminando *glitches* isolados de comutação.

2. **Desequilíbrio de Corrente Estritamente sob Regime de Carga:** Seja $\Phi_I$ o conjunto de fases ativas do medidor ($|\Phi_I| \ge 2$) e $\bar{I}(t) = \frac{1}{|\Phi_I|}\sum_{\phi \in \Phi_I} I_\phi(t)$ a corrente média instantânea. O desequilíbrio de corrente é avaliado **exclusivamente nos intervalos em regime de carga significativa** ($\bar{I}(t) \ge I_{\min}$, com $I_{\min} = 1,0\text{ A}$):
   $$\delta_I(t) = \begin{cases} \dfrac{\max_{\phi \in \Phi_I} |I_\phi(t) - \bar{I}(t)|}{\bar{I}(t)}, & \text{se } \bar{I}(t) \ge 1,0\text{ A} \\[8pt] \text{indefinido (ignorado)}, & \text{se } \bar{I}(t) < 1,0\text{ A} \end{cases}$$
   Consolidando-se pelo quantil robusto $P_{90}(\delta_I)$, neutraliza-se integralmente a saturação noturna de *standby* em consumidores residenciais normais, preservando sensibilidade total a *bypasses* e assimetrias sob carga real.

### Estágio 3: Isolamento Intra-Janela de Perda de Pacotes *RF Mesh* e Eliminação do Viés de Sobrevivência

Seja $\mathcal{T}_c = \{d \in \mathcal{D}_{\text{lake}} : t_c - W < d \le t_c\}$ o conjunto de dias calendário efetivamente extraídos no *Data Lake* para a janela encerrada no *cutoff* $t_c$, com cardinalidade $N_{\text{cal}}(t_c) = |\mathcal{T}_c|$.

1. **Cobertura Normalizada pelo Calendário (Sem *Survival Bias*):** A cobertura média da UC $i$ na janela $t_c$ é normalizada por todos os dias extraídos no calendário municipal, atribuindo cobertura zero aos dias em que o medidor silenciou completamente:
   $$\overline{\text{cov}}_{i, c} = \frac{1}{N_{\text{cal}}(t_c)} \sum_{d \in \mathcal{T}_c} \text{cov}_{i, d}, \qquad \text{onde } \text{cov}_{i,d} = \frac{n_{\text{slots}}(i, d)}{288} \in [0, 1]$$

2. **Estimação Elétrica Restrita a Dias Confiáveis:** Define-se o subconjunto de dias com telemetria íntegra da UC $i$ na janela $t_c$ por $\mathcal{R}_{i,c} = \{d \in \mathcal{T}_c : \text{cov}_{i,d} \ge \gamma_{\text{dia}}\}$, com limiar padrão $\gamma_{\text{dia}} = 0,70$ ($\ge 202$ leituras de 5 minutos no dia). Todas as estatísticas comportamentais de consumo (`HIST_FA_SUM_MEDIAN`, `HIST_FA_SUM_MAD`, `HIST_LOAD_FACTOR_MEAN` e a inclinação linear por mínimos quadrados ordinários `TREND_FA_SLOPE` sobre coordenadas reais de calendário) são computadas **exclusivamente sobre $d \in \mathcal{R}_{i,c}$**. Assim, um dia de apagão na rede *RF Mesh* jamais simula um dia de consumo elétrico zero nem gera inclinações negativas artificiais de consumo.

### Estágio 4: Consolidação Multi-Janela Desacoplada e Validação *Out-of-Time* (OOT)

Em vez de um único *snapshot* degenerado ou de janelas móveis diárias autocorrelacionadas, o `select_non_overlapping_cutoffs` particiona o eixo temporal em **janelas contíguas e mutuamente exclusivas** (sobreposição nula):
- **Modo Mensal (`stride_mode = "monthly"`, padrão recomendado, $W = 30\text{ dias}$):** Emite exatamente um *cutoff* consolidado no fechamento de cada mês calendário (Agosto, Setembro e Outubro compõem o conjunto canônico de Treinamento; Novembro e meses subsequentes compõem os conjuntos cegos *Out-of-Time* de Validação e Teste). Cada janela encapsula quatro ciclos semanais completos, estabilizando a razão entre dias úteis e finais de semana.
- **Modo Quinzenal (`stride_mode = "biweekly"`, $W = 15\text{ dias}$):** Emite dois *cutoffs* independentes por mês ($d \le 15$ e $d > 15$), dobrando a densidade longitudinal quando desejado para análise de transientes mais curtos, mantendo $0\%$ de sobreposição de leituras.
- **Particionamento OOT Intra-Trimestre Pré-Novembro (`intra_canonical_multi_window_oot`):** Enquanto os dados de Novembro ainda estão em coleta, o `DatasetSplitter` reconhece automaticamente a presença de múltiplos *cutoffs* mensais dentro de Agosto–Outubro, treinando os especialistas de **todos os alimentadores** nas janelas de Agosto e Setembro e retendo a janela fechada de Outubro para validação e teste *Out-of-Time* estratificados por alimentador.

### Estágio 5: Portão de Elegibilidade (`IS_CLEAN_FOR_TRAIN`) e Bifurcação Diagnóstica Operacional

Cada janela $\text{UC} \times t_c$ recebe o predicado booleano auditável de elegibilidade para ajuste de densidade:

$$\text{IS\_CLEAN\_FOR\_TRAIN}_{i,c} \iff \big(|\mathcal{R}_{i,c}| \ge m_{\min}(t_c)\big) \;\land\; \big(\overline{\text{cov}}_{i,c} \ge 0,60\big)$$

onde $m_{\min}(t_c) = \min\!\big(15,\, \lceil 0,50 \cdot N_{\text{cal}}(t_c) \rceil\big)$ adapta-se proporcionalmente caso um mês específico possua extração parcial no calendário.

Essa separação institui uma **bifurcação operacional rigorosa** no treinamento e na tomada de decisão da distribuidora:
1. **Ajuste Não-Supervisionado Imune a Telecom (`Fit` sobre $\mathcal{X}_{\text{clean}}$):** Tanto o Modelo Global Âncora quanto cada Especialista Local de Alimentador no `HierarchicalModelOrchestrator` ajustam suas florestas de isolamento, vizinhanças LOF, distribuições empíricas ECOD, autovetores PCA e limiares calibrados $\tau_f$ **estritamente sobre janelas com `IS_CLEAN_FOR_TRAIN == True`**.
2. **Roteamento Explícito para Quarentena de Telecomunicações (`TELECOM_QUARANTINE`):** Unidades consumidoras cujas janelas reprovem no portão de integridade não são descartadas silenciosamente nem misturadas aos alvos de inspeção antifraude; são exportadas em artefato dedicado (`output/reports/telecom_quarantine_uc_windows.parquet` / `.csv`, categorizadas em `QUARANTINE_SILENT_MESH` e `QUARANTINE_INTERMITTENT_MESH`) e rotuladas como `TELECOM_QUARANTINE` no pipeline hierárquico.

Do ponto de vista de engenharia de distribuição, essa distinção evita o erro operacional dispendioso de despachar equipes de inspeção comercial (caça-fraude) para endereços cujo único sintoma era sombra de rádio ou falha de roteador *RF Mesh*, direcionando cada anomalia à equipe competente (Telecomunicações/Automação vs. Recuperação de Energia).

---

## 4. Governança, Auditabilidade e Integração à Suíte TUI

Para assegurar reprodutibilidade científica integral ao longo do desenvolvimento do TCC:
- **Imutabilidade dos Dados Brutos:** O `RetroactiveDataCleaner` opera em modo 100% *offline* sobre as partições locais já extraídas, sem modificar nenhum registro das camadas `raw/`, `measurements/`, `events/` ou `context/`. Qualquer pesquisador ou auditor pode reexecutar a curadoria variando $\gamma_{\text{dia}}$ ou o modo de janela (`monthly` vs. `biweekly`) de forma determinística.
- **Integração Nativa na TUI (`uv run main.py`):**
  - Na **Opção 1 (`[5] Saneamento Retroativo & Reconstrução Multi-Janela`)**, o operador executa a reconstrução completa das partições diárias e janelas canônicas com barra de progresso em tempo real e relatório de auditoria (`%` de alimentadores reconciliados, `%` de janelas com `MAD > 0`, volume elegível vs. quarentena *RF Mesh*).
  - Na **Opção 2 (`Treinar & Validar Modelos de IA`)**, um **Portão Automático de Pré-Voo** inspeciona a matriz `training_dataset` antes de qualquer ajuste: caso detecte sintomas de dataset legado não saneado (taxa de `meta__feeder` nulo $> 50\%$ ou taxa de `MAD == 0` $> 95\%$), bloqueia preventivamente o treinamento defeituoso, apresenta o diagnóstico quantitativo na tela e oferece a execução imediata do saneamento retroativo.
  - Ao término de novas extrações em lote (`scripts/run_araucaria_full.py`), a consolidação multi-janela saneada passa a ser executada automaticamente como etapa padrão de fechamento do pipeline.

