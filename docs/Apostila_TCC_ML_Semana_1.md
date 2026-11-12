# Apostila TCC: Machine Learning e Detecção de Anomalias em Redes Inteligentes (AMI)
## Módulo 1 — Estatística Descritiva, Séries Temporais e Engenharia de Features para Smart Grids

---

### 1. Visão Geral e Contexto de Smart Grids (AMI)
A infraestrutura de medição avançada (**AMI - Advanced Metering Infrastructure**) substitui a leitura eletromecânica mensal por telemetria bidirecional em tempo real. No contexto de concessão (ex.: COPEL Araucária), cada medidor inteligente reporta grandezas elétricas e operacionais:
- **Energia Ativa Direta (FA)**: kWh consumido pela Unidade Consumidora (UC).
- **Energia Ativa Reversa (RA)**: kWh injetado na rede (relevante para Geração Distribuída / Solar FV).
- **Tensões por Fase ($V_A, V_B, V_C$)** e **Correntes ($I_A, I_B, I_C$)**.
- **Fator de Potência ($FP$) e Reversão de Corrente**.
- **Metadados de Comunicação**: Latência, taxa de entrega de pacotes, indicador de salto mesh, resets do medidor.

O grande desafio do processamento de dados AMI reside no volume, na assimetria das distribuições de consumo e na presença contínua de ruídos, perdas de pacotes e comportamentos não-estacionários.

---

### 2. Estatística Descritiva Robusta vs. Paramétrica

#### 2.1 Média ($\mu$) vs. Mediana ($\tilde{x}$)
Em curvas de carga reais, o consumo não segue uma distribuição gaussiana (Normal). A presença de picos súbitos (partida de motores, fornos, solda) ou períodos prolongados de consumo nulo faz com que a média aritmética seja facilmente corrompida:
$$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$
A **mediana** $\tilde{x} = \text{Q2} = P_{50}$ representa o valor central ordenado, possuindo um ponto de ruptura (*breakdown point*) de 50%, o que a torna imune a outliers pontuais extremos.

#### 2.2 Desvio Padrão ($\sigma$) vs. Desvio Absoluto Mediano (MAD)
O desvio padrão quadrático $\sigma = \sqrt{\frac{1}{N-1}\sum (x_i - \mu)^2}$ amplifica os erros ao quadrado. Em contrapartida, a estatística de dispersão robusta ideal é o **MAD (Median Absolute Deviation)**:
$$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$
Para equipará-lo ao desvio padrão de uma normal assintótica, aplica-se o fator multiplicativo $1.4826$:
$$\hat{\sigma}_{\text{robust}} = 1.4826 \times \text{MAD}$$

#### 2.3 Intervalo Interquartil (IQR) e Diagrama de Tukey
O IQR delimita os 50% centrais dos dados ordenados:
$$\text{IQR} = Q_3 - Q_1$$
Critério de corte de Tukey para detecção de valores atípicos:
- Outlier moderado: $x > Q_3 + 1.5 \times \text{IQR}$ ou $x < Q_1 - 1.5 \times \text{IQR}$
- Outlier severo: $x > Q_3 + 3.0 \times \text{IQR}$ ou $x < Q_1 - 3.0 \times \text{IQR}$

---

### 3. Anatomia de Séries Temporais em Redes de Distribuição

Uma série temporal de telemedição de uma UC $y(t)$ amostrada com cadência regular (ex.: 1 hora ou 15 minutos) pode ser decomposta aditivamente ou multiplicativamente:
$$y(t) = T(t) + S(t) + R(t)$$
onde:
1. **$T(t)$ (Tendência)**: Mudança estrutural do patamar de consumo ao longo dos meses (crescimento orgânico, sazonalidade anual de temperatura, evasão ou fraude paulatina).
2. **$S(t)$ (Sazonalidade)**:
   - *Intradia (24 horas)*: Curva em duplo pico típica (pico matutino e pico noturno residencial entre 18h e 21h).
   - *Semanal (7 dias)*: Contraste entre dias úteis (segunda a sexta) e finais de semana (sábado e domingo).
3. **$R(t)$ (Resíduo / Estocástico)**: Variações aleatórias, ruído de medição e anomalias transitórias.

#### Fator de Carga ($LF$)
Razão fundamental do perfil da unidade consumidora:
$$LF = \frac{E_{\text{total}}}{P_{\text{max}} \times \Delta t} = \frac{\bar{P}}{P_{\text{max}}}$$
Um fator de carga próximo a $1.0$ reflete consumo plano contínuo (cargas industriais/servidores). Um fator de carga baixo ($< 0.20$) reflete operação intermitente e picos curtos.

---

### 4. Engenharia de Features para Detecção de Perdas e Anomalias

Para alimentar modelos de Machine Learning (Isolation Forest, LOF, XGBoost), a série temporal bruta de 30 a 90 dias deve ser condensada em um vetor tabular de características comportamentais:

| Família | Feature | Formulação / Definição | Interpretação no TCC |
| :--- | :--- | :--- | :--- |
| **Magnitude** | `fa_sum_median` | $\text{median}_d(\sum_{h=1}^{24} E_{d,h})$ | Patamar central diário de consumo. |
| **Variabilidade** | `fa_sum_mad` | $\text{MAD}_d(\sum_{h=1}^{24} E_{d,h})$ | Estabilidade operacional do cliente. |
| **Assimetria** | `skewness` | $\frac{\frac{1}{N}\sum(x-\mu)^3}{\sigma^3}$ | Cauda alongada de consumo extraordinário. |
| **Tendência** | `trend_slope` | $\frac{\sum(t_i - \bar{t})(y_i - \bar{y})}{\sum(t_i - \bar{t})^2}$ | Queda progressiva (possível início de furto/derivação). |
| **Ciclicidade** | `peak_hour_cyclic` | $\sin(\frac{2\pi \cdot h_{\text{pico}}}{24}), \cos(\frac{2\pi \cdot h_{\text{pico}}}{24})$ | Momento horário de maior utilização sem descontinuidade 23h-0h. |
| **Partição Temporal** | `day_night_ratio` | $\frac{\sum_{h=06}^{18} E_h}{\sum_{h=19}^{05} E_h}$ | Razão diurno/noturno (detecta atividade comercial irregular). |
| **Operação Semanal** | `weekend_weekday_ratio`| $\frac{\bar{E}_{\text{fim\_de\_semana}}}{\bar{E}_{\text{dia\_util}}}$ | Distingue residências de comércios e escolas. |
| **Integridade** | `completion_rate` | $\frac{N_{\text{leituras\_recebidas}}}{N_{\text{esperadas}}}$ | Qualidade de telemetria da memória de massa. |
| **Fluxo Inverso** | `has_reverse_flow` | $\sum \text{RA} > 0 \land \text{sem\_cadastro\_gd}$ | Suspeita de Geração Distribuída não homologada. |
| **Desequilíbrio** | `current_unbalance` | $\frac{\max(I_A, I_B, I_C) - \min(I_A, I_B, I_C)}{\text{mean}(I_A, I_B, I_C)}$ | Desvio de fase em ramal trifásico / bifásico. |

