# Fundamentação Teórico-Metodológica: Adaptação de Processamento de Sinais e Qualidade da Energia para Redes AMI

> **Classificação:** Documento Teórico e Metodológico para o Artigo / Dissertação  
> **Autoria:** Projeto TCC Extração & Detecção de Anomalias Copel (Araucária)  
> **Status:** Referência Canônica de Modelagem Conceitual (Versão Consolidada e Calibrada)  
> **Pergunta Central de Pesquisa:**
> *"Como decompor sinais AMI de baixa frequência em evidências temporais, físicas, de comunicação e contextuais, respeitando o que cada fonte realmente permite inferir?"*
>
> **Literatura de Referência Primária:**
> - Bollen, M. et al. (2007) — *Classification of Underlying Causes of Power Quality Disturbances: Deterministic versus Statistical Methods*. EURASIP Journal on Advances in Signal Processing.
> - Chen, S. & Zhu, X. (2007) — *Wavelet-Based Power Quality Disturbance Feature Extraction and Classification*.
> - Xi, J. et al. (2018) — *Adaptive Kalman Filtering for Power Quality Disturbance Segmentation and Edge Detection*.
> - ANEEL — *Procedimentos de Distribuição de Energia Elétrica no Sistema Elétrico Nacional (PRODIST)*, Módulo 8 (Qualidade da Energia Elétrica) e Módulo 5 (Medição).

---

## 1. Contexto e Motivação Científica

A literatura clássica de **Qualidade da Energia Elétrica (QEE / PQ - Power Quality)** e **Detecção de Perdas Não Técnicas (PNT)** desenvolveu-se historicamente sobre duas vertentes desconectadas:

1. **A Vertente Subciclada / Alta Resolução (Processamento de Sinais Elétricos):**
   Trabalha com formas de onda amostradas em dezenas de quilohertz (kHz), aplicando Transformadas de Fourier de Tempo Curto (STFT), Wavelets Discretas (DWT), Transformada S e Decomposição em Modos Empíricos (EMD / Hilbert-Huang). O foco reside em capturar afundamentos rápidos (*voltage sags*), sobretensões transitórias (*swells*), oscilações harmônicas e ruído de chaveamento em escala de microssegundos a milissegundos.
2. **A Vertente Cadastral / Baixa Resolução (Machine Learning Comercial):**
   Trabalha com faturamento mensal consolidado (1 ponto a cada 30 dias), aplicando algoritmos tabulares genéricos (SVM, Random Forest, Redes Neurais densas) sobre médias e históricos de consumo em kWh, sem qualquer acoplamento com a física do circuito elétrico ou com a topologia do alimentador.

A implementação de **Infraestruturas de Medição Inteligente (AMI - Advanced Metering Infrastructure)**, como o parque de medidores inteligentes de Araucária (Copel), introduz uma **camada intermediária**: séries temporais multivariadas com cadência típica de **5 a 15 minutos** (96 a 288 intervalos diários), integrando:
- Grandezas acumuladas e integradas (Energia Ativa Direta $FA$, Reversa $RA$, Reativa Indutiva/Capacitiva $FR$, Demanda Máxima $MD$);
- Grandezas instantâneas fasoriais pontuais (Tensões por fase $U_{L1}, U_{L2}, U_{L3}$, Correntes $I_{L1}, I_{L2}, I_{L3}$, Ângulos e Fator de Potência);
- Alarmes e eventos assíncronos gerados na borda do medidor (Violações de tampa, perturbações de relógio, falhas de fase, tentativas de intrusão magnética).

### 1.1 O Limite da Amostragem e o Uso Adequado de Transformadas
Em um sinal com intervalo de discretização $\Delta t = 15 \text{ min}$, o limite de Nyquist impõe que a frequência máxima observável seja $f_{\max} = \frac{1}{2 \times 900 \text{ s}} \approx 0.55 \text{ mHz}$. 

Fenômenos subciclados transitórios de milissegundos **já foram integrados ou filtrados pelo firmware do medidor** antes do envio do pacote. Tentar rodar Wavelet ou Transformada S nessa escala para buscar "transitórios rápidos de tensão" é um erro metodológico. No entanto, ferramentas multiescala (como wavelets em escalas de dias/semanas) poderiam conceitualmente representar tendências de longo prazo de consumo, mas não justificam a complexidade introduzida no estágio atual do projeto.

O que se aproveita da literatura não são os algoritmos de alta frequência em si, mas sim os **princípios estruturantes da engenharia de sinais**:
1. A divisão entre segmentação temporal, extração de representações intermediárias e classificação;
2. A separação entre detecção de mudança de regime e caracterização de anomalia;
3. A dualidade entre regras determinísticas (física/regulação) e modelos estatísticos.

---

## 2. Quatro Pilares Conceituais Calibrados para Dados AMI

### Pilar 1: Complementaridade entre Change Point Detection (CPD) e Anomaly Detection
A detecção de mudança de regime (*Change Point Detection*) e a detecção de anomalias estatísticas (*Anomaly Detection*, ex.: Isolation Forest) respondem a perguntas essencialmente distintas e complementares:
- **Change Point Detection pergunta:** *"Houve uma descontinuidade ou quebra de regime na dinâmica temporal da série da UC?"*
- **Anomaly Detection pergunta:** *"O estado/regime atual desta UC é raro e discrepante quando comparado à distribuição de referência (histórico da própria UC ou de seus pares)?"*

Uma não substitui a outra:
- Uma UC pode estar sob fraude já consolidada há meses (sem mudança abrupta na janela corrente de 30 dias), sendo detectável apenas por atipicidade na distribuição estatística de consumo ou desbalanço de corrente.
- Fraudes podem ocorrer por deriva lenta de calibração, redução progressiva, manipulação intermitente (fins de semana) ou alteração sutil de sazonalidade.
- Por outro lado, um consumidor pode sofrer uma mudança legítima de regime (desocupação de imóvel, mudança de moradores) que gera um *change point* nítido sem configurar irregularidade.

Portanto, o **CPD atua como um bloco central e enriquecedor de features dinâmicas**, fornecendo variáveis estruturadas (instante do salto, magnitude $\Delta \mu$, persistência e direção) que alimentam o classificador downstream.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   COMPLEMENTARIDADE: CPD vs. DETECTOR DE ANOMALIA                      │
├──────────────────────────────────────┬─────────────────────────────────────────────────┤
│ Change Point Detection (CPD)         │ Anomaly Detection (Isolation Forest / PCA Res)  │
├──────────────────────────────────────┼─────────────────────────────────────────────────┤
│ • Responde: "Quando e como mudou?"   │ • Responde: "Quão atípico é este comportamento?"│
│ • Detecta transições e degraus       │ • Detecta estados raros e regimes crônicos      │
│ • Depende de ordenação temporal      │ • Opera no espaço multivariado de atributos     │
│ • Sensível ao momento da quebra      │ • Sensível a valores extremos consolidados      │
└──────────────────────────────────────┴─────────────────────────────────────────────────┘
```

### Pilar 2: Formulação do CPD — Sazonalidade e Distinção Offline vs. Online
A aplicação de CUSUM ou testes de comparação de distribuições (Kolmogorov-Smirnov, Wasserstein) exige rigor estatístico sobre o sinal:
1. **Tratamento da Sazonalidade:** Aplicar CUSUM sobre $FA$ em kWh bruto gera alarmes falsos inevitáveis devido aos ciclos naturais (temperatura, sazonalidade semanal, horários de pico). O algoritmo deve operar sobre os **resíduos da série desazonalizada** ou sobre uma série normalizada por horário do dia e tipo de dia (útil vs. fim de semana):
   $$e(t) = x(t) - \hat{\mu}_{\text{circadiana}}(h(t), d(t))$$
   onde $h(t) \in [0, 23]$ e $d(t) \in \{\text{útil}, \text{sábado}, \text{domingo}\}$.
2. **Distinção entre Auditoria Offline e Detecção Online:**
   - **Offline (Auditoria Histórica):** Compara janelas simétricas antes $[t - W, t]$ e depois $[t, t + W]$ do instante candidato $t$. Válido para perícias forenses e análises em lote retroativas, mas incorre em vazamento temporal se executado em tempo real.
   - **Online (Monitoramento Contínuo):** Compara a janela de referência histórica estável $[t - W_{\text{base}}, t - W_{\text{test}}]$ contra a janela recente de teste $[t - W_{\text{test}}, t]$. O CUSUM bicaudal sequencial acumula desvios unilaterais com limiares de folga $k$ e decisão $h$:
     $$S_t^+ = \max(0, S_{t-1}^+ + (e_t - k)), \quad S_t^- = \max(0, S_{t-1}^- - (e_t + k))$$

### Pilar 3: Cautela Epistemológica nas Saídas — De Rótulos Causais a Padrões de Compatibilidade
Em sistemas de baixa cadência de medição (15 minutos), telemetria isolada **nunca constitui prova material de causa**. Modelos estatísticos e regras de sinais não devem emitir veredictos como *"Fraude Confirmada"*, *"UC Desocupada"* ou *"GD Não Cadastrada"*, mas sim **padrões de compatibilidade operacional**:

```
                       Mudança ou Anomalia Detectada
                                     │
      ┌──────────────────────────────┼──────────────────────────────┐
      ▼                              ▼                              ▼
Padrão Compatível              Padrão Compatível              Padrão Compatível
com Geração                    com Redução de Carga           com Falha de Medição
(FA ↓, RA ↑ diurno)            (FA ↓, RA ≈ 0, V normal)       (U_Lk = 0 ou I_Lk = 0)
      │                              │                              │
      └──────────────────────┬───────┴──────────────────────────────┘
                             ▼
              CORROBORAÇÃO MULTIFONTE
  ┌────────────────────────────────────────────────────────┐
  │ • Cadastro Comercial (CIS): Contrato de GD ativo?      │
  │ • Histórico de Medição: Troca recente de medidor?      │
  │ • Alarmes Físicos: Violação de tampa, campo magnético? │
  │ • Vizinhança Topológica: Outras UCs do trafo caíram?   │
  └──────────────────────────┬─────────────────────────────┘
                             ▼
              DIAGNÓSTICO OPERACIONAL GRADUADO
       (Despacho de Inspeção / Manutenção / Auditoria)
```

1. $FA \downarrow$ simultâneo a $RA \uparrow$ durante o dia é um **padrão compatível com geração local**; torna-se suspeita de irregularidade apenas se o cadastro comercial da Copel não acusar homologação de microgeração naquela UC.
2. $FA \approx 0$ com tensão dentro das faixas normais é um **padrão compatível com imóvel desocupado ou desligamento total de disjuntor**; só configura anomalia se houver contradição cadastral ou persistência atípica em perfil comercial.
3. Desbalanço acentuado de corrente sem alarme físico é um **padrão compatível com by-pass monofásico ou falha interna de carga**; demanda corroboração espacial no transformador.

### Pilar 4: Rigor Regulatória — Aplicação Estrita do PRODIST Módulo 8
A modelagem de conformidade de tensão deve respeitar os critérios normativos da ANEEL, evitando simplificações que descaracterizem o padrão regulatório:
1. **Regime Permanente de Tensão:** O PRODIST M8 não define uma regra binária instantânea isolada. A conformidade é avaliada por indicadores estatísticos de duração calculados sobre amostras semanais (1.008 leituras de 10 minutos):
   - **DRP (Duração Relativa da Transgressão de Tensão Precária):** Percentual de leituras em faixa precária, com limite regulatório de $DRP \le 3\%$.
   - **DRC (Duração Relativa da Transgressão de Tensão Crítica):** Percentual de leituras em faixa crítica, com limite regulatório de $DRC \le 0.5\%$.
   - As faixas de tensão dependem estritamente da tensão nominal contratada ($V_n$) e da faixa de fornecimento (ex.: em secundário 127V/220V: adequada de 117V a 133V; precária de 110V a 116V e 134V a 135V; crítica $<110\text{V}$ ou $>135\text{V}$).
2. **Desequilíbrio de Tensão:** O PRODIST M8 e normas IEC estabelecem o indicador de Fator de Desequilíbrio de Tensão ($FD\%$) baseado no método de componentes simétricas (razão entre sequência negativa e sequência positiva: $V_2 / V_1$). Em dados AMI sem fasores de ângulo, o PVUR (*Phase Voltage Unbalance Rate*) constitui uma aproximação de engenharia aceita, mas sua aplicação deve ser documentada como proxy de triagem, não como laudo normativo conclusivo.
3. **Distinção entre Fenômenos:**
   - Transgressões de regime permanente (DRP/DRC);
   - Desequilíbrio permanente vs. intermitente;
   - Variações de Tensão de Curta Duração (VTCDs - afundamentos e elevações subsegundo), que no AMI aparecem apenas como reflexos integrados ou registros de alarmes específicos.

---

## 3. Arquitetura Alvo Simplificada e Pragmática

Em vez de colecionar modelos não supervisionados redundantes (empilhando ECOD, COPOD, LOF, Isolation Forest e PCA sem ganho marginal), a arquitetura consolida-se em um pipeline linear, interpretável e computacionalmente eficiente:

```
                          AMI + CONTEXTO (Parquet / Datalake)
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
     [Qualidade e Comunicação]                        [Normalização Temporal]
      • Cobertura (m_t)                                • Perfil circadiano (hora)
      • Lacunas e micro-gaps                           • Perfil semanal (útil/fds)
      • Data Quality Score (C_30d)                     • Série de Resíduos e(t)
                 │                                                 │
                 ▼                                                 ▼
             [S_comm]                                    [Change Point Engine]
       (Detector Específico)                              • Instante da quebra t*
                                                          • Magnitude do salto Δμ
                                                          • Persistência temporal
                                                          • Direção (queda/subida)
                                                                   │
                                                                   ▼
                                                     [Caracterização do Regime]
                                                      • Features elétricas
                                                      • Desbalanço dinâmico
                                                      • Razões temporais (dia/noite)
                                                      • Autocorrelação (Lag 1, 7)
                                                      • S_elec (Isolation Forest / PCA)
                                                                   │
                 ┌─────────────────────────────────────────────────┴─────────────────┐
                 ▼                                                                   ▼
    [Regras Físicas / PRODIST M8]                                       [Atribuição de Padrões]
     • Triagem DRP/DRC preliminar                                        • Compatibilidade com GD
     • Limites de desbalanço                                             • Compatibilidade com queda
     • Alarmes de hardware                                               • Atipicidade comportamental
                 │                                                                   │
                 └────────────────────────┬──────────────────────────────────────────┘
                                          ▼
                       [Corroboração Espacial & Topológica]
                        • Vizinhança de Transformador
                        • Vizinhança de Alimentador
                        • Histórico Cadastral de Medidor
                                          │
                                          ▼
                         [Diagnóstico Operacional Integrado]
```

### Componentes Fundamentais do MVP:
1. **$S_{\text{comm}}$:** Detector específico de integridade e padrões de perda de pacotes (gaps curtos vs. falhas prolongadas vs. apagões de rede);
2. **CPD Engine:** Mecanismo unificado de mudança de regime baseado em CUSUM sobre resíduos desazonalizados e divergência estatística entre janelas de observação;
3. **$S_{\text{elec}}$:** Detector de anomalia multivariado principal (Isolation Forest calibrado ou PCA Residual), operando sobre a matriz com *gating* estrito de qualidade ($C_{30d} \ge 0.70$);
4. **Módulo Físico:** Regras determinísticas orientadas à física da distribuição e conformidade PRODIST;
5. **Pós-processamento de Atribuição:** Cruzamento com alarmes de hardware, histórico de trocas de medidor e consenso espacial com transformadores adjacentes.

---

## 4. Priorização de Features Temporais de Baixa Frequência

A caracterização temporal é priorizada na seguinte ordem de relevância empírica e interpretabilidade:

1. **Razão Diurno / Noturno ($\text{Day/Night Ratio}$):**
   $$R_{\text{dn}} = \frac{\sum_{t \in [06\text{h}, 18\text{h}]} FA(t)}{\sum_{t \notin [06\text{h}, 18\text{h}]} FA(t) + \epsilon}$$
   Sensível a desvios que ocorrem predominantemente em horário de expediente comercial ou noturno.

2. **Razão Dias Úteis / Fim de Semana ($\text{Weekday/Weekend Ratio}$):**
   $$R_{\text{ww}} = \frac{\bar{FA}_{\text{seg-sex}}}{\bar{FA}_{\text{sab-dom}} + \epsilon}$$
   Captura quebras no hábito de funcionamento de estabelecimentos industriais e comerciais.

3. **Autocorrelações Temporais ($\rho_1$ e $\rho_7$):**
   - $\rho_1$: Persistência diária consecutiva;
   - $\rho_7$: Persistência do ciclo semanal de carga. Quedas abruptas em $\rho_7$ indicam desorganização comportamental ou intermitência de sinal.

4. **Diferenciais Pré/Pós-Mudança ($\Delta \mu$ e $\Delta \text{MAD}$):**
   $$\Delta \mu = \frac{\bar{x}_{\text{post}} - \bar{x}_{\text{pre}}}{\bar{x}_{\text{pre}} + \epsilon}, \quad \Delta \text{MAD} = \frac{\text{MAD}_{\text{post}} - \text{MAD}_{\text{pre}}}{\text{MAD}_{\text{pre}} + \epsilon}$$
   Normalizados pela escala da própria instalação, permitindo comparar UCs residenciais de 150 kWh/mês com comerciais de 5.000 kWh/mês.

5. **Métricas de Persistência:**
   - $\text{Days Since Change}$ ($D_{\text{shift}}$): Tempo decorrido desde o ponto de virada;
   - $\text{Persistence Ratio}$ ($P_{\text{regime}}$): Proporção de dias após $t^*$ em que o novo patamar se manteve estável.

---

## 5. Conclusão Metodológica para o TCC

Ao evitar analogias forçadas com processamento de sinais de alta frequência e fundamentar a modelagem na dinâmica de sistemas de distribuição de baixa frequência, o trabalho atinge três objetivos científicos:
1. **Realismo Metodológico:** As variáveis manipuladas representam fenômenos que o hardware AMI de 15 minutos efetivamente consegue capturar sem aliasing.
2. **Transparência Regulatória:** A fronteira entre leis da física, normas da ANEEL e estimativas de Machine Learning permanece explícita e auditável.
3. **Robustez Operacional:** A combinação de detecção de mudança (CPD), detecção de raridade estatística ($S_{\text{elec}}$) e corroboração espacial reduz drasticamente a taxa de alarmes falsos inerente a modelos não supervisionados convencionais.
