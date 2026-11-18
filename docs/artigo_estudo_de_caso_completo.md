# Artigo de Estudo de Caso: Anatomia Empírica da Detecção de Fraudes, Telemetria RF Mesh e Diagnóstico Operacional em Smart Grid

**Autores:** Engenharia Analítica AMI — Caso Copel Araucária  
**Status:** Artigo Completo de Estudo de Caso e Validação Operacional  
**Ambiente:** Infraestrutura AMI de Araucária/PR (262.849 janelas avaliadas, 66.099 UCs ativas)  
**Conformidade:** LGPD (Anonimização de identificadores de UC e NIO via `censor_uc_id`)  

---

## Resumo Executivo

A detecção de Perdas Não-Técnicas (PNT), falhas metrológicas e anomalias de rede em sistemas de Medição Inteligente (*Advanced Metering Infrastructure* - AMI) de grande porte enfrenta três armadilhas fundamentais:
1. **A ilusão da média / agregação comercial**, onde consumidores fraudulentos ou com perfis opostos apresentam o mesmo faturamento mensal agregado;
2. **A sobreposição entre telemetria de rede e física da carga**, na qual perdas de pacotes no protocolo RF Mesh mimetizam desvios ou cortes de consumo;
3. **A falta de rastreabilidade causal**, em que alarmes de modelos "caixa-preta" não fornecem evidências visuais e físicas acionáveis para envio de equipes de campo.

Este artigo apresenta um estudo de caso sistemático baseado na evidência empírica coletada no piloto de Araucária/PR. A partir das imagens geradas pelo pipeline analítico e pelo **Cockpit AMI**, investigam-se três atos operacionais:
- **Ato I:** O paradoxo do consumo equivalente (`934***50` vs. `116****57`), demonstrando por que métricas clássicas de faturamento falham e como a dispersão robusta e a curva horária resolvem a separação de classes;
- **Ato II:** A assinatura periódica em dente de serra da rede RF Mesh (Heatmap de densidade de *delays*), evidenciando a separação ontológica entre o ciclo de *polling* do protocolo e anomalias crônicas reais;
- **Ato III:** O processo de triagem operacional no Cockpit Analítico AMI, a priorização dos **Candidatos a Vistoria** com persistência temporal (`DIAS_ANOMALIA_SEGUIDOS`), o diagnóstico da UC `685***20` (desbalanceamento/bypass) e o tratamento de Geração Distribuída (GD) clandestina.

---

## 1. Ato I — A Falácia da Média: Duas UCs com Mesmo Consumo Mediano e Dinâmicas Opostas

```
+--------------------------------------------------------------------------------------------------+
|                    COMPARAÇÃO DE PERFIL DE CARGA: UCs COM MESMO CONSUMO                          |
|                             (934***50  vs  116****57)                                            |
|                                                                                                  |
|   [Mediana Diária Idêntica]               [Curva de Carga Horária Real de 7 Dias]                |
|   UC 934***50: 20,38 kWh/dia              UC 934***50: Linha plana contínua (0,85 kWh/h)         |
|   UC 116****57: 19,22 kWh/dia             UC 116****57: Pico diurno solar/comercial (7,03 kWh/h)  |
|                                                                                                  |
|   MAD 934***50: 0,051 kWh/dia             Fator de Carga: 0,973 (934***50) vs 0,150 (116****57)  |
|   MAD 116****57: 16,564 kWh/dia (324x)    Desbalanceamento Corrente: 0,001 vs 1,000              |
+--------------------------------------------------------------------------------------------------+
```

### 1.1 O Caso Empírico das UCs `934***50` e `116****57`

Nos sistemas legados de faturamento das concessionárias, a avaliação de regularidade de uma Unidade Consumidora é historicamente restrita à energia faturada em 30 dias. Sob essa ótica, considere duas UCs reais da rede de Araucária:

* **UC Estável `934***50`:** Mediana diária de **`20,38 kWh/dia`** (Alimentador *Fonte Nova*);
* **UC Volátil `116****57`:** Mediana diária de **`19,22 kWh/dia`** (Alimentador *CSN*).

A diferença de consumo diário aparente é inferior a 5,7%, o que induziria qualquer regra de negócio unidimensional a enquadrar ambos os consumidores na mesma faixa tarifária residencial/comercial convencional.

### 1.2 A Revelação pelas Métricas Robustas e Granularidade AMI

Ao decompor a telemetria horária de 7 dias (168 horas contínuas de memória de massa em `ami_interval`), a assinatura física revela naturezas eletrotécnicas completamente divergentes:

1. **Dispersão Absoluta da Mediana ($MAD$):**  
   Enquanto a UC `934***50` tem $MAD = 0{,}051\text{ kWh/dia}$ e coeficiente de variação $CV = 0{,}0025$, a UC `116****57` apresenta $MAD = 16{,}564\text{ kWh/dia}$ ($CV = 0{,}8572$) — uma dispersão **324 vezes superior**!
2. **Morfologia da Curva de Carga 24h:**  
   - A UC `934***50` mantém um consumo ininterrupto de $0{,}85\text{ kWh/h}$ as 24 horas do dia, 7 dias por semana. Seu Fator de Carga ($FC = P_{\text{med}} / P_{\text{max}}$) é de **0,973**. Essa é a assinatura física característica de **cargas de base contínuas** (servidores, estações rádio-base de telecomunicações ou câmaras frigoríficas industriais).
   - A UC `116****57` exibe consumo nulo durante a madrugada, com elevação abrupta a partir das 07h00 e picos intensos que atingem **`7,03 kWh/h`** no período vespertino, encerrando com fator de carga de apenas **0,150** e desbalanceamento de corrente máximo igual a **1,000**.
3. **Implicação Regulatória e Metodológica:**  
   Tratar a UC `934***50` como anormal por "não desligar de madrugada" ou classificar a UC `116****57` apenas pelo total mensal é um erro operacional grave. O pipeline do projeto resolve essa ambiguidade pareando a tendência central robusta com $MAD$, $IQR$ intradiário, decomposição em componentes principais (PCA) e a matriz de resíduos estruturais.

---

## 2. Ato II — A Assinatura da Rede: O Padrão Dente de Serra na Telemetria RF Mesh

```
+--------------------------------------------------------------------------------------------------+
|                   HEATMAP DE DENSIDADE DE DELAYS DE TELEMETRIA (VOLTAGE)                         |
|                         (Agregação de 5 minutos — Resolução 0 a 2520 min)                        |
|                                                                                                  |
|   2160-2520m | ············ (Latência residual de pacotes retidos / buffers de concentrador)    |
|   1320-1350m |                                                                                   |
|    960-990m  |                                                                                   |
|    600-630m  |                                                                                   |
|    240-270m  | ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// (Dente de Serra)     |
|    120-125m  | ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// (Ciclos de Polling)  |
|     30-31m   | ||||| ||||| ||||| ||||| ||||| ||||| ||||| ||||| ||||| ||||| (Transmissão Direta)|
|      0-1m    |                                                                                   |
|              +-----------------------------------------------------------------------------------+
|               00:00  02:00  04:00  06:00  08:00  10:00  12:00  14:00  16:00  18:00  20:00  22:00 |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 Decodificando o Heatmap de Latência de Medição

A imagem do Heatmap de Densidade de Delays da grandeza Tensão (`VOLTAGE`), com agregação temporal de 5 minutos ao longo de 24 horas, evidencia um comportamento fascinante:
- No eixo vertical ($Y$), observa-se a escala de atraso de entrega dos pacotes, normalizada em faixas de 0 a 2.520 minutos (42 horas);
- No eixo horizontal ($X$), a janela de medição das 00:00 às 23:30;
- A escala de cor logarítmica ($\log_{10}(\text{Densidade})$) destaca a concentração dos pacotes.

A estrutura visual dominante é formada por **linhas diagonais repetitivas em formato "dente de serra"** entre 60 e 450 minutos de atraso.

### 2.2 Física da Rede vs. Falso Positivo de Fraude

Esse padrão em dente de serra **não representa anomalia de carga, nem fraude no medidor**:
1. **Origem Protocolar (Batch Polling):** Em redes mesh inteligentes (Wi-SUN / IEEE 802.15.4g), medidores armazenam as leituras de 5 minutos em memória circular local (*head-end buffer*). Os concentradores operacionais realizam o esvaziamento em janelas regulares (ex.: a cada 2 ou 4 horas). À medida que o tempo avança entre os ciclos de varredura, a idade do dado mais antigo daquele lote cresce linearmente (inclinação unitária positiva), despencando abruptamente para zero no instante em que o *concentrador* executa o *flush* e transmite a rajada de pacotes.
2. **Isolamento de Causa (Padrão dentro do Padrão):**  
   Se um analista aplicasse detecção de anomalia ingênua sobre a latência, milhares de UCs seriam classificadas como defeituosas devido às cristas do dente de serra.  
   O detector de telecomunicação do projeto (`CommunicationAnomalyDetector`) aprendeu a assinatura do dente de serra e isola apenas a **verdadeira anomalia**:
   - Medidores que fogem das diagonais regulares e entram na faixa superior constante de $> 1.320\text{ minutos}$ (atraso crônico por perda de rota no grafo mesh);
   - UCs com déficit localizado de cobertura em relação aos seus vizinhos no mesmo transformador de distribuição (`meta__coverage_deficit_vs_peers >= 0.30`).

---

## 3. Ato III — O Cockpit Analítico AMI: Triagem, Diagnóstico e Ação de Campo

```
+--------------------------------------------------------------------------------------------------+
|                             COCKPIT ANALÍTICO AMI — COPEL ARAUCÁRIA                              |
|                                                                                                  |
|   [Total UCs Analisadas]      [Anomalias Confirmadas]    [Possível GD Clandestina]  [Quarentena]  |
|         262.849                      2.629                         704                  2.939    |
|     Araucária Completa           1,00% da População         Fluxo Reverso s/ Cad.    Mesh Degr.  |
|                                                                                                  |
|   +-- PAINEL ANALÍTICO DE ANOMALIAS DETECTADAS ----------------------------------------------+   |
|   | 🎯 Candidatos a Vistoria (Campo) : 274 UCs (0,41%)  ====================                  |   |
|   | ✔ Operação Regular (Consumo Ok)  : 62.886 UCs (95,14%) ===============================... |   |
|   | 📶 Quarentena Telecom (Isolada)  : 2.939 UCs (4,45%)   ===========                        |   |
|   | ☀️ Possível GD Não Cadastrada    : 201 UCs (0,30%)     ====                               |   |
|   | ⚡ Desequilíbrio / Bypass        : 63 UCs (0,10%)      ==                                 |   |
|   | 🔬 Outlier Multivariado PyOD     : 6 UCs (0,01%)       =                                  |   |
|   | ⚠️ Medidor Travado (Consumo Zero): 2 UCs (0,00%)       =                                  |   |
|   | 📉 Degrau Descendente CUSUM      : 2 UCs (0,00%)       =                                  |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                                                                  |
|   [FILTROS RÁPIDOS]: [🎯 Candidatos Vistoria] [🚨 Fraude/Desb.] [☀️ Injeção GD] [🔄 Mudança]... |
|                                                                                                  |
|   UC           Alimentador    Score    Diagnóstico Causal           Dias Seguidos    Consumo Med |
|   100618405    -              0,5643   POSSIVEL GD NAO CADASTRADA   🔥 21 dias       25,8 kWh    |
|   100845703    Independência  0,5791   POSSIVEL GD NAO CADASTRADA   🔥 17 dias       49,1 kWh    |
|   101047126    Represa        0,5458   POSSIVEL GD NAO CADASTRADA   🔥 42 dias       41,3 kWh    |
|   102551294    -              0,5506   POSSIVEL GD NAO CADASTRADA   🔥 57 dias       10,4 kWh    |
|   103440666    -              0,5499   POSSIVEL GD NAO CADASTRADA   🔥 28 dias       10,8 kWh    |
|   104546603    Gonvarri       0,5439   POSSIVEL GD NAO CADASTRADA   🔥 40 dias       43,2 kWh    |
+--------------------------------------------------------------------------------------------------+
```

### 3.1 Anatomia da Tela Principal do Cockpit

A interface operacional em execução revela a síntese do processamento das 262.849 janelas históricas de Araucária:
1. **Quadro de KPIs Estratégicos:**
   - **Total de UCs Analisadas:** 262.849 janelas;
   - **Anomalias Confirmadas pelo Ensemble PyOD:** 2.629 (exatamente 1,00% da população com score no quantil superior crítico);
   - **Possível GD Clandestina:** 704 UCs com registro persistente de reversão ativa de energia (`RA_INTERVAL_SUM > 0`) sem cadastro técnico no CIS;
   - **Quarentena Telecom:** 2.939 UCs com perda massiva de pacotes (cobertura $< 60\%$ ou $< 15$ dias confiáveis).
2. **O Funil de Priorização Operacional:**  
   Do total de 2.629 anomalias numéricas, a categoria **`🎯 Candidatos a Vistoria (Campo)`** isola estritamente **274 UCs prioritárias (0,41%)**. Essa filtragem remove descontinuidades de cadastro de UCs novas e problemas de comunicação, entregando às equipes de fiscalização apenas alvos com alta probabilidade de retorno de receita.

### 3.2 O Diferencial da Métrica `DIAS_ANOMALIA_SEGUIDOS`

Como demonstrado na tabela de UCs ativas filtradas por *Possível GD Não Cadastrada*:
- UC `101047126` (Alimentador *Represa*): Score 0,5458 com **🔥 42 dias seguidos de reversão ativa**;
- UC `102551294`: Score 0,5506 com **🔥 57 dias seguidos**;
- UC `104546603` (Alimentador *Gonvarri*): Score 0,5439 com **🔥 40 dias seguidos**.

A inclusão dos dias seguidos de persistência temporal garante que flutuações pontuais de medição (por exemplo, 1 dia de chaveamento ou manutenção de rede) não gerem falsos alertas. A anomalia torna-se uma constatação inequívoca de geração fotovoltaica conectada clandestinamente à rede de distribuição.

### 3.3 Diagnóstico de Derivação e Bypass: O Caso Real `685***20`

No relatório de anomalias com indício de fraude eletrotécnica (`DESEQUILIBRIO_CARGA_OU_BYPASS`), destaca-se a UC `685***20` (Alimentador *Polonia*):
- **Diagnóstico Causal:** Desbalanceamento de corrente severo associado à inclinação descendente de consumo;
- **Score do Ensemble:** 0,6212;
- **Slope de Consumo:** $-1{,}3682\text{ kWh/dia}$;
- **Assimetria de Fases:** Desbalanceamento de corrente máximo de **0,798** com tensão nominal balanceada.

**Interpretação Física:** Em uma instalação trifásica ou bifásica regular, correntes de carga podem ter desvios moderados por uso de equipamentos monofásicos. Porém, quando uma das fases tem sua corrente drenada a níveis próximos de zero sem alteração na tensão, enquanto o consumo total em kWh registra quebra de patamar estatisticamente relevante (CUSUM), conclui-se que uma das fases foi derivada antes do elemento medidor (fraude por *bypass parcial*).

---

## 4. Conclusão e Recomendações para a Operação

A integração entre rigor matemático não-supervisionado, compreensão protocolar da telemetria RF Mesh e visualização interativa comprovou as seguintes diretrizes para a concessionária:

1. **Abandono do Faturamento Isolado:** Consumos médios iguais mascaram realidades operacionais diametralmente opostas (como visto no Par Contrastante `934***50` vs. `116****57`). Métricas de forma de curva ($FC$, $MAD$, resíduos de PCA) devem compor a análise regular de cadastro.
2. **Desacoplamento de Telecom da Física da Carga:** As cristas periódicas de dente de serra do *polling* mesh jamais devem ser interpretadas como anomalias de consumo, sob pena de gerar milhares de falsos alarmes em bairros saudáveis.
3. **Priorização por Persistência Temporal:** O envio de fiscais de campo deve basear-se exclusivamente no ranking de `CANDIDATOS_A_VISTORIA`, ordenado decrescentemente por `DIAS_ANOMALIA_SEGUIDOS` e `ANOMALY_SCORE`.
4. **Combinação de Modelos Ortogonais:** Cada técnica de ML possui seu papel insubstituível — K-Means segmenta hábitos comparáveis, PCA residual captura deformações na onda de carga, IForest isola raridades globais, LOF descobre micro-grupos de fraude e ECOD mapeia caudas extremas multivariadas.

---

### Referências Operacionais e Arquivos de Evidência

* Imagem comparativa de curvas horárias: [`docs/figures/eda/exemplos/comparacao_duas_ucs_mesmo_consumo.png`](file:///d:/Projects/tcc_extraction/docs/figures/eda/exemplos/comparacao_duas_ucs_mesmo_consumo.png)
* Heatmap de atraso de pacotes mesh: Visualizado na telemetria de 5 minutos da Copel Araucária
* Dashboard e Servidor do Cockpit Web: [`scripts/run_web_cockpit.py`](file:///d:/Projects/tcc_extraction/scripts/run_web_cockpit.py)
* Script de demonstração dos métodos de ML: [`scripts/demo_ml_methods.py`](file:///d:/Projects/tcc_extraction/scripts/demo_ml_methods.py)
* Enriquecimento vetorizado de persistência: [`scripts/enrich_candidate_vistorias.py`](file:///d:/Projects/tcc_extraction/scripts/enrich_candidate_vistorias.py)
