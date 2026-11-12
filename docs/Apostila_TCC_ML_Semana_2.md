# Apostila TCC: Machine Learning e Detecção de Anomalias em Redes Inteligentes (AMI)
## Módulo 2 — Qualidade de Dados, Detecção de Quebra de Regime (CPD/CUSUM) e Agrupamento (Clustering)

---

### 1. Mecanismos de Ausência de Dados e Tratamento de Gaps em AMI

Em redes mesh de rádio frequência (RF) ou celulares (GPRS/LTE/NB-IoT), a perda de pacotes e a retenção de dados são governadas por três mecanismos estatísticos:
1. **MCAR (Missing Completely at Random)**:
   - A probabilidade de perda não depende nem dos valores observados nem dos não-observados.
   - Exemplo: Queda aleatória de pacote por colisão eletromagnética esporádica no canal de RF.
2. **MAR (Missing at Random)**:
   - A perda depende de grandezas observáveis do sistema.
   - Exemplo: Medidores situados no fim de ramal ou com múltiplos saltos mesh (*hop count* > 4) apresentam maior probabilidade de gap.
3. **MNAR (Missing Not at Random)**:
   - A perda está correlacionada com o próprio fenômeno elétrico que deveria ser medido.
   - Exemplo: Desligamento intencional do medidor para ligação direta (tampering/fraude) ou blecaute por sobretensão severa.

#### Métricas Críticas de Qualidade Telemetria:
- **Taxa de Completude ($C$)**:
  $$C = \frac{N_{\text{registros\_recebidos}}}{N_{\text{teorico\_periodo}}} \in [0, 1]$$
- **Duração Máxima do Gap ($G_{\max}$)**: Tempo contínuo em horas ou dias em que a telemetria permaneceu silenciada.
- **Número de Janelas Interrompidas ($N_{\text{gaps}}$)**.

---

### 2. Detecção de Mudança de Regime (Change Point Detection - CPD)

A distinção entre um "dia atípico" (evento transitório) e um "degrau de consumo permanente" (fraude, troca de morador, avaria de TC/TP) exige métodos formais de CPD.

#### 2.1 CUSUM Sequencial Bicaudal (Cumulative Sum Control Chart)
O CUSUM monitora desvios acumulados da série de resíduos $e_t = y_t - \hat{m}_t$ em relação ao valor de tolerância $k$:
$$S_t^+ = \max(0, S_{t-1}^+ + e_t - k)$$
$$S_t^- = \max(0, S_{t-1}^- - e_t - k)$$
com condições iniciais $S_0^+ = S_0^- = 0$.
- **Alarme disparado**: Quando $S_t^+ > h$ (aumento persistente) ou $S_t^- > h$ (queda persistente), onde $h$ é o limiar de decisão estatística calibrado para evitar falsos alarmes.
- **Ponto de Inflexão Estimado ($t^*$)**: Data do último ponto em que $S_t$ iniciou a subida monotônica a partir de zero.

#### 2.2 Detecção Offline Global (PELT - Pruned Exact Linear Time)
Minimização global do custo de particionamento com penalização $\beta$:
$$\min_{m, \tau} \sum_{i=1}^{m+1} \mathcal{C}(y_{\tau_{i-1} : \tau_i}) + \beta \, f(m)$$
Onde $\mathcal{C}$ representa o custo de verossimilhança residual (ex.: variância interna do segmento).

---

### 3. Decomposição Espectral e Redução de Dimensionalidade (PCA / SVD)

A curva horária de 24 horas de uma UC reflete sua rotina operativa. Ao invés de trabalhar com 24 dimensões correlacionadas, aplicamos **Análise de Componentes Principais (PCA)**:

1. **Normalização por Perfil de Carga (Shape)**:
   $$p_{d, h} = \frac{E_{d, h}}{\sum_{k=1}^{24} E_{d, k} + \epsilon}, \quad \sum_{h=1}^{24} p_{d, h} = 1$$
   Essa formulação desacopla o **formato da curva** da **magnitude (volume de energia)**.
2. **Decomposição da Matriz de Covariância**:
   $$X = U \Sigma V^T$$
   - **PC1**: Captura o balanço diurno vs. noturno (explica tipicamente 50–65% da variância).
   - **PC2**: Captura o deslocamento do horário de ponta (vespertino vs. matutino, ~15–25% da variância).
   - **PC3**: Modulação de finais de semana.

---

### 4. Agrupamento Comportamental de Consumidores (Clustering)

#### 4.1 K-Means com Distância Euclidiana em Componentes Principais
Agrupamento nos scores de $[PC1, PC2, \text{Fator de Carga}]$ para categorizar arquétipos operacionais:
- **Cluster 1: Residencial Noturno** (duplo pico, baixa carga diurna).
- **Cluster 2: Comercial Diurno** (consumo concentrado das 08h às 18h).
- **Cluster 3: Industrial / Comercial 24h** (curva plana contínua, alto fator de carga).
- **Cluster 4: Residencial Baixa Renda / Iluminação Externa** (perfil achatado, picos unicamente após 19h).

#### 4.2 Métricas de Avaliação de Cluster:
- **Silhouette Score**: Coesão intra-cluster vs. separação inter-cluster:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- **Índice Davies-Bouldin ($DB$)**: Razão de dispersão intra-cluster sobre distâncias entre centroides (quanto menor, mais compacto e isolado).

#### 4.3 DBSCAN e HDBSCAN para Detecção Baseada em Densidade
Ao invés de forçar cada ponto em um grupo, algoritmos baseados em densidade isolam consumidores cujas características residem em regiões de baixa densidade no espaço de features, rotulando-os nativamente como **ruído (-1)**, servindo como pré-filtro não-supervisionado de anomalias raras.

