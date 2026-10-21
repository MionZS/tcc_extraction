# Arquitetura de Modelagem de Machine Learning: Especialistas Multivisão, Detecção de Mudança e Atribuição de Padrões

> **Classificação:** Documento Técnico de Arquitetura e Modelagem para o Artigo / Dissertação  
> **Autoria:** Projeto TCC Extração & Detecção de Anomalias Copel (Araucária)  
> **Status:** Especificação Canônica de Machine Learning e Inteligência Analítica  
> **Arquivos-Fonte no Repositório:**
> - Detectores Especialistas: [`src/models/specialized_detectors.py`](file:///d:/Projects/tcc_extraction/src/models/specialized_detectors.py)
> - Construção da Matriz de Treinamento: [`src/datasets/build_training_dataset.py`](file:///d:/Projects/tcc_extraction/src/datasets/build_training_dataset.py)
> - Grid Regular e Análise de Lacunas: [`src/features/temporal_grid.py`](file:///d:/Projects/tcc_extraction/src/features/temporal_grid.py)
> - Categorização de Eventos: [`src/events/alarm_categories.py`](file:///d:/Projects/tcc_extraction/src/events/alarm_categories.py)
> - Conformidade e Classificação PRODIST: [`src/models/voltage_classifier.py`](file:///d:/Projects/tcc_extraction/src/models/voltage_classifier.py)
> - Filtragem Espúria e Vizinhança: [`src/models/spurious_filter.py`](file:///d:/Projects/tcc_extraction/src/models/spurious_filter.py)

---

## 1. Princípios Arquiteturais e Visão Geral

A detecção de anomalias e perdas não técnicas em redes de distribuição de energia enfrenta um desafio clássico: **o trade-off entre interpretabilidade física e flexibilidade estatística**. Tratar a medição inteligente (AMI) como uma tabela única alimentada em um algoritmo caixa-preta genérico (ex.: Isolation Forest cego) produz taxas de falsos positivos inaceitáveis para as equipes de campo da concessionária, além de mascarar falhas de comunicação como se fossem irregularidades elétricas.

Para superar essa limitação, o sistema foi estruturado sobre **quatro princípios orientadores**:

1. **Ortogonalidade das Famílias de Detectores:**
   Problemas de comunicação (perda de sinal de modem GPRS), problemas mecânicos (abertura de tampa de medidor) e comportamentos elétricos físicos anômalos (queda de impedância, desbalanço de corrente) possuem naturezas e distribuições matemáticas completamente distintas. Devem ser modelados por **famílias especialistas desacopladas**.
2. **Gating Rígido de Qualidade de Dados ($C_{30d}$):**
   Modelos elétricos nunca devem receber dados imputados cegamente. Se uma UC possui menos de $70\%$ de cobertura de dados válidos nos últimos 30 dias ($C_{30d} < 0.70$), ela é direcionada prioritariamente para auditoria de telemetria ($S_{\text{comm}}$), impedindo que buracos de medição contaminem a estimativa do comportamento físico ($S_{\text{elec}}$).
3. **Complementaridade: Mudança de Regime (CPD) vs. Raridade Estatística:**
   Um evento atípico pode se apresentar como uma ruptura súbita no tempo ou como um estado cronicamente fora da curva. O pipeline combina um mecanismo de **Change Point Detection** (para descontinuidades e transições) com um **Detector de Anomalia Multivariado** (para raridade multidimensional).
4. **Saídas de Compatibilidade Operacional (Sem Veredictos Automáticos Precipitados):**
   A saída analítica não rotula sumariamente uma instalação como "Fraude", mas fornece um **perfil de compatibilidade probabilística e estruturada** (ex.: *padrão compatível com redução abrupta de carga*; *padrão compatível com geração*), cruzando com dados cadastrais e corroborando espacialmente com UCs vizinhas no mesmo transformador.

---

## 2. As Quatro Famílias de Especialistas

```
                                  DADOS BRUTOS AMI
                    (Intervalos 15 min, Fasores Instantâneos, Alarmes)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            [Triagem de Telemetria]                 [Normalização Cadastral]
                     │                                         │
        ┌────────────┴────────────┐                            │
        ▼                         ▼                            ▼
  [S_comm: Comunicação]     [Gating: C_30d >= 0.70?]    [S_tamper: Hardware]
  • Perda de pacotes        (Filtro de Integridade)     • Alarmes físicos
  • Padrões de lacunas                    │             • Trocas de medidor
  • Viés noturno/diurno                   ▼             • Contexto de idade
                             [S_elec: Elétrico Puro]
                             • Matriz física (sem NaNs)
                             • Change Point (CUSUM resíduos)
                             • IF / PCA Residual
                             • Regras PRODIST M8
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
        [Behavioral Clustering]                   [Corroboração Espacial]
        • Curvas 24h normalizadas                 • Consenso de Transformador
        • PCA + KMeans                            • Consenso de Alimentador
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          ▼
                         [Matriz de Atribuição e Diagnóstico]
```

### 2.1 Especialista de Comunicação e Disponibilidade ($S_{\text{comm}}$)
* **Objetivo:** Quantificar a integridade do canal de telemetria, modelando falhas de modem, saturação de rede mesh/celular e perda intermitente de dados.
* **Espaço de Entrada ($X_{\text{comm}}$):**
  - $\text{COVERAGE\_RELIABLE\_DAYS}$: Número de dias nos últimos 30 dias com cobertura $\ge 90\%$;
  - $\text{AVG\_INTERVAL\_COVERAGE}$: Cobertura média de intervalos de 15 minutos na janela ($0.0$ a $1.0$);
  - $\text{GAP\_COUNT}$: Quantidade total de lacunas de telemetria identificadas;
  - $\text{MAIOR\_GAP\_MIN}$: Duração da maior lacuna contínua em minutos;
  - $\text{NOCTURNAL\_MISSING\_RATIO}$: Proporção de dados perdidos durante o período noturno (00h às 06h);
  - $\text{AVG\_ALARM\_LATENCY\_SEC}$: Tempo médio decorrido entre a ocorrência do evento no medidor e a recepção no MDM.
* **Formulação Matemática do Score:**
  $$S_{\text{comm}} = 1.0 - \left( 0.40 \cdot \text{AVG\_INTERVAL\_COVERAGE} + 0.35 \cdot \frac{\text{COVERAGE\_RELIABLE\_DAYS}}{30} + 0.25 \cdot (1.0 - \text{BURST\_RATIO}) \right)$$
  Valores próximos a $1.0$ indicam telemetria severamente degradada.

### 2.2 Especialista Elétrico e Físico ($S_{\text{elec}}$)
* **Objetivo:** Isolar perturbações físicas legítimas, perdas não técnicas e distúrbios de fornecimento sem contaminação por dados faltantes.
* **Gating de Entrada:** Executado **estritamente** para UCs com $\text{Data Quality} = C_{30d} \ge 0.70$. Valores faltantes em micro-gaps ($\le 15\text{ min}$) são interpolados linearmente; gaps superiores a 15 minutos são preservados e forçam o isolamento da observação.
* **Sub-blocos Componentes:**
  1. **Bloco de Mudança de Regime (Change Point Engine):**
     - Opera sobre os **resíduos desazonalizados** da série diária:
       $$e_t = FA_t - \bar{FA}_{\text{dow}(t)}$$
     - CUSUM bicaudal computa a evidência acumulada de quebra de média com parâmetros calibrados de tolerância ($k = 0.5\sigma$) e limiar de alarme ($h = 4.0\sigma$):
       $$S_t^+ = \max(0, S_{t-1}^+ + (e_t - k)), \quad S_t^- = \max(0, S_{t-1}^- - (e_t + k))$$
     - Extrai: $\Delta \mu_{\text{pre/post}}$ (salto percentual), $D_{\text{shift}}$ (dias decorridos desde $t^*$) e persistência temporal da nova média.
  2. **Bloco Estatístico e Dinâmico:**
     - $\text{HIST\_FA\_SUM\_MEDIAN}$ e $\text{HIST\_FA\_SUM\_MAD}$ (nível e dispersão robusta);
     - $\text{TREND\_FA\_SLOPE}$ (inclinação linear da tendência por regressão robusta de Theil-Sen);
     - $\text{DAY\_NIGHT\_RATIO}$ e $\text{WEEKDAY\_WEEKEND\_RATIO}$;
     - Autocorrelações de curto e médio prazo ($\rho_1$ e $\rho_7$).
  3. **Bloco Físico e Regulatório:**
     - $\text{VOLTAGE\_IMBALANCE\_MAX}$ (PVUR calculado fasor a fasor sobre fases ativas);
     - $\text{CURRENT\_IMBALANCE\_MAX}$ (assimetria máxima dinâmica de corrente entre condutores de fase);
     - $\text{HIST\_RA\_REVERSAL\_DAYS}$ e $\text{HIST\_RA\_REVERSAL\_RATIO\_MAX}$ (ocorrência e intensidade de fluxo reverso de potência ativa);
     - Indicadores de conformidade PRODIST M8 (triagem de violações em regime permanente).
  4. **Motor Multivariado Principal:**
     - **Isolation Forest calibrado** ($n=200$ árvores, contaminação esperada $2\%$) ou **PCA Residual** (reconstrução sobre os 5 componentes principais mais representativos, medindo o erro quadrático de projeção $Q = \|x - \hat{x}\|^2$).

### 2.3 Especialista de Violação e Histórico de Medição ($S_{\text{tamper}}$)
* **Objetivo:** Detectar evidências físicas de intervenção mecânica, fraude de relógio interno ou adulteração de fiação combinadas com alteração contratual.
* **Espaço de Entrada ($X_{\text{tamper}}$):**
  - $\text{ALARM\_COUNT\_WINDOW}$: Total de alarmes de violação física (abertura de tampa, tamper magnético, inversão de TC) na janela de 30 dias;
  - $\text{METER\_AGE\_DAYS}$: Idade do medidor instalado em relação à data de corte (evitando vazamento de instalações futuras);
  - $\text{METER\_CHANGED\_30D}$ e $\text{METER\_CHANGED\_90D}$: Flags booleanas indicando substituição física recente de medidor;
  - $\Delta \mu_{\text{pre/post}}$ (tamanho da queda de consumo pós-evento).
* **Lógica de Decisão:**
  Um alarme isolado de abertura de tampa pode ser uma manutenção de rotina da concessionária. No entanto, a ocorrência de $\text{TAMPER\_COVER\_OPEN}$ acompanhada de $\Delta \mu < -40\%$ nos dias subsequentes à data do evento eleva drasticamente o score de prioridade de fiscalização.

### 2.4 Agrupador Comportamental de Carga (*Behavioral Clusterer*)
* **Objetivo:** Identificar o perfil diário típico de consumo da UC e quantificar desvios em relação aos padrões de consumo de seus pares da mesma classe comercial/tarifária.
* **Formulação:**
  1. A matriz diária de 96 pontos normalizados (soma unitária diária $\sum_{i=1}^{96} p_i = 1.0$) descreve a **forma da curva**, desacoplada da magnitude de kWh:
     $$p_i = \frac{FA_i}{\sum_{k=1}^{96} FA_k + \epsilon}$$
  2. Aplica-se **PCA** para redução de 96 dimensões para $k=6$ variáveis latentes (responsáveis por $>85\%$ da variância da forma da curva);
  3. Algoritmo **K-Means** (com $K \in [3, 8]$ otimizado por pontuação de Silhueta) particiona as UCs em arquétipos: *comercial diurno*, *residencial com pico noturno*, *industrial contínuo*, *perfil plano/baixo consumo*.
  4. O score de atipicidade mede a distância euclidiana da curva da UC ao centroide do seu cluster atribuído.

---

## 3. Matriz de Atribuição e Diagnóstico Operacional

Em vez de produzir uma decisão binária "normal vs. anômalo", o pipeline calcula um vetor de compatibilidade com classes físicas conhecidas:

| Classe Operacional de Diagnóstico | Condições Elétricas e Temporais Observadas | Corroboração Cadastral / Alarmes Requerida | Ação Operacional Recomendada |
| :--- | :--- | :--- | :--- |
| **Padrão Compatível com Geração Local** | $\Delta \mu < 0$ no consumo diurno ($FA$), $RA$ consistente nos horários de maior irradiância (10h às 15h). | Cruzamento CIS: UC possui homologação de micro/minigeração distribuída ativa? | Se não homologada: vistoria de conexão à revelia (risco de segurança e ilha). Se homologada: operação normal. |
| **Padrão Compatível com Redução Súbita de Carga** | Quebra de regime detectada pelo CPD ($t^*$), degrau $\Delta \mu < -50\%$, $RA \approx 0$, tensões equilibradas e normais. | Alarme de tamper ausente, sem registros de substituição de medidor. | Triagem comercial: vistoria para verificar desocupação de imóvel, mudança de atividade ou desvio físico. |
| **Padrão Compatível com Falha de Fase / Medição** | Queda abrupta de consumo, fasor de tensão de uma fase em zero ($U_{Lk} \approx 0$) ou corrente zerada ($I_{Lk} = 0$), desbalanço extremo de fases. | Alarmes de perda de fase (`PHASE_FAILURE`) ou corroboração com UCs vizinhas no mesmo transformador. | Se UCs vizinhas normais: defeito no medidor/TP da UC (manutenção de medição). Se vizinhas também afetadas: falta física de fase na rede secundária. |
| **Padrão Compatível com Intervenção Mecânica / By-pass** | CPD detecta degrau persistente de queda ($\Delta \mu < -40\%$), acompanhado de desbalanço acentuado de corrente entre fases ativas sem queda de tensão. | Presença de alarmes da categoria `TAMPER` (`COVER_OPEN`, `MAGNETIC_TAMPER`) em janela próxima ao instante $t^*$. | Prioridade máxima de fiscalização de campo para recuperação de receita (perdas não técnicas). |
| **Padrão Indeterminado / Transiente** | Perturbação estatística detectada, mas sem persistência temporal ($D_{\text{shift}} < 3\text{ dias}$) e sem quebra sustentada de resíduos. | Ausência de alarmes físicos e conformidade de tensão mantida. | Manutenção em lista de observação temporal passiva (sem despacho de equipe). |

---

## 4. Corroboração Espacial e Topológica (Filtro de Falsos Positivos)

O principal mecanismo para blindar o sistema contra falsos positivos sistêmicos é o **Filtro de Vizinhança Elétrica** ([`src/models/spurious_filter.py`](file:///d:/Projects/tcc_extraction/src/models/spurious_filter.py)):

Uma perturbação observada em uma única UC não pode ser analisada isoladamente de sua localização na topologia da rede:
1. **Consenso de Transformador de Distribuição:**
   Para cada UC sinalizada com desvio, inspecionam-se todas as outras UCs conectadas ao mesmo transformador (`POSTO_OPERACIONAL`).
   - Se $> 70\%$ das UCs do mesmo posto exibem afundamento de tensão ou queda de consumo no mesmo dia $\to$ Trata-se de um **evento sistêmico da rede** (ex.: manutenção programada, queima de fusível de ramal, perturbação de média tensão a montante). O score individual da UC é desqualificado como suspeita de irregularidade comercial.
   - Se apenas a UC avaliada exibe anomalia enquanto seus pares no transformador operam em regime estável $\to$ A evidência de **anomalia intrínseca da instalação** é fortalecida.
2. **Consenso de Alimentador:**
   Variações de nível de tensão são normalizadas pela média do alimentador (`ALIMENTADOR`), neutralizando manobras de regulação de tensão na subestação (mudança de tap de transformador comutador de carga - LTC).

---

## 5. Estratégia de Validação e Prevenção de Vazamento Temporal (*Anti-Leakage*)

Para garantir que as métricas de validação reflitam o desempenho real em produção, o particionamento de dados segue regras estritas de integridade científica:

1. **Particionamento por Unidade Consumidora (GroupKFold por `UC`):**
   Registros temporais da mesma UC **nunca** podem ser alocados simultaneamente em conjuntos de treino e teste. Se uma mesma UC figurar em ambos os conjuntos em datas distintas, o modelo memorizará o nível absoluto de consumo do consumidor, resultando em acurácia inflada artificialmente.
2. **Validação Temporal Out-of-Time:**
   - Conjunto de Treinamento: Dias $D_1$ a $D_{60}$ (Agosto e Setembro de 2026);
   - Conjunto de Teste: Dias $D_{61}$ a $D_{81}$ (Outubro de 2026).
3. **Causalidade Estrita de Features:**
   Ao calcular as características de uma UC na data de corte $D$, apenas eventos, medições e alarmes com timestamp $\le D$ às 23:59:59 são considerados. Registros de trocas de medidor ou cadastros ocorridos em $D+1$ são rigorosamente blindados.
