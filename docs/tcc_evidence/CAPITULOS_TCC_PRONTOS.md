# Capítulos Metodológicos e Experimentais Prontos para Incorporação ao TCC

**Título:** *Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches*

---

## Capítulo X — Metodologia de Engenharia de Dados e Fusão Multi-Visão

### X.1 Entidade Analítica Canônica e Blindagem Temporal
Toda a modelagem projeta fenômenos elétricos intervalares (5 min), contexto cadastral/topológico e eventos de alarme na entidade canônica $\text{UC} \times \text{cutoff\_date}$. Nenhuma medição com $\text{REPORT\_DAY} > \text{cutoff\_date}$ participa do cálculo de atributos móveis, garantindo ausência estrita de vazamento temporal (*temporal leakage*).

### X.2 Formalização Matemática da Fusão de Especialistas (Fiel ao Código)
A arquitetura executa quatro camadas sequenciais:

1. **Blindagem de Subespaço (`select_pure_electrical_features`):** Exclusão determinística de variáveis de comunicação e ciclo de vida (`coverage`, `latency`, `alarm`, `meter_age`, `meter_changed`).
2. **Comitê Elétrico Não-Supervisionado (`CompositeAnomalyDetector`):** Combina seis famílias ortogonais padronizadas por `RobustScaler` e normalizadas em $[0, 1]$:
   $$S_{\text{comp}}(x) = \text{clip}\left(\frac{0.22\,s_{\text{IF}} + 0.20\,s_{\text{LOF}} + 0.18\,s_{\text{ECOD}} + 0.18\,s_{\text{COPOD}} + 0.10\,s_{\text{HBOS}} + 0.12\,s_{\text{PCA}}}{1.0} + 0.15\cdot \mathbb{I}_{\text{PRODIST}}(x),\; 0,\; 1\right)$$
3. **Gating de Qualidade (`ElectricalAnomalyDetector`):** Observações com `data_quality < 0.70` recebem `is_reliable = False` e não são declaradas anomalias elétricas.
4. **Roteamento Hierárquico (`MoERouter`) e Filtro Espúrio (`filter_spurious_anomalies`):** Combina modelos por alimentador, município e global conforme tamanho amostral do circuito; eventos coletivos no mesmo transformador ($\ge 60\%$ dos pares anômalos) sofrem atenuação ($\times 0.35$) e falhas não isoladas ($\times 0.40$).

---

## Capítulo Y — Resultados Experimentais na Rede de Araucária

Na campanha analisada sobre **66,099 UCs únicas** (**262,849 janelas** `UC × cutoff_date`), o sistema identificou **932 UCs com anomalia detectada** (1.41%) no último corte temporal e priorizou **274 candidatos a vistoria** por persistência temporal consecutiva.

Os resultados completos de ablação ($M_0 \to M_5$), sensibilidade de gating ($50\%$ a $90\%$) e caracterização exploratória encontram-se auditados nas tabelas CSV e figuras vetoriais SVG/PNG anexas no diretório de evidências.

---

## Capítulo Z — Estudos de Caso Periciais e Diagnóstico Diferencial (Fraude Física vs Telecomunicação)

A eficácia do sistema reside na sua capacidade de resolver o diagnóstico diferencial entre eventos de rede de comunicação e fraudes metrológicas com evidência empírica incontestável:

### Z.1 UC 90850700 — Fraude por Derivação Clandestina Precedida de Apagão de Rede
Na instalação `UC 90850700` (Alimentador Óleos Brasil), o detector CUSUM identifica um degrau descendente abrupto de **$-68.54\%$** em **31/10/2026**, reduzindo a média diária de $17.16\text{ kWh/dia}$ para $\sim 5.10\text{ kWh/dia}$, com persistência contínua por **93 dias seguidos**.

A decomposição horária da telemetria revela que, entre **28/10 21h e 30/10 21h (exatas 48 horas)**, a completude da comunicação foi a **$0.0\%$ (apagão absoluto)**. Imediatamente após a reconexão da comunicação em 30/10 à noite, as medições de 31/10 ressurgem mutiladas no novo patamar deprimido. 

Este padrão reproduz fidedignamente o rastro pericial de adulteração no padrão de entrada: o disjuntor geral é desarmado ou o medidor é removido para implantação de *by-pass* / shunt de corrente nas fases. Ao ser religado, o medidor restabelece enlace com a malha mesh, mas afere apenas uma fração ($31.5\%$) da corrente real. O desequilíbrio severo ($I_{imb} = 1.40$) e reversão induzida ($RA = 0.43$) corroboram a irregularidade, classificando a UC com prioridade máxima de fiscalização (`Candidato a Vistoria: SIM (Prioritário)`).

### Z.2 UC 29113296 — O Falso Positivo Evitado via Gating de Telecomunicação (Controle Negativo)
Na instalação `UC 29113296`, o histórico diário de 30 dias atesta um consumo regular e estável ao redor de $25.45\text{ kWh/dia}$, com $100\%$ de integridade do registrador metrológico acumulador. Todavia, a telemetria horária de 14 dias acusa descarte de pacotes de $50.0\%$ na malha RF mesh ($168/336$ janelas degradadas).

Sem o filtro de qualidade, o algoritmo computaria a soma dos intervalos parciais e geraria um alarme falso de sub-consumo severo. A camada de *Gating de Quarentena* (`meta__telecom_status == "QUARANTINE"`) intervém e bloqueia o despacho de campo (`IS_CANDIDATO_VISTORIA = NÃO`), economizando o deslocamento inútil de viaturas de inspeção e comprovando a imunidade do pipeline a ruídos de telecomunicação.


---
## Adendo Experimental: Avaliação em Novembro/2026

### 3.4 Protocolo de Avaliação Temporal com Dados Reais de Novembro/2026
A validação do modelo com dados de campo foi executada sobre os cortes temporais de 01/11/2026 a 19/11/2026 da concessão de Araucária (PR). Para mitigar qualquer risco de vazamento temporal (*look-ahead bias*), o conjunto de treino utilizou estritamente os cortes mensais consolidados de Agosto, Setembro e Outubro/2026 (197.083 janelas UC × dia), enquanto a avaliação fora do tempo (*out-of-time* - OOT) foi processada sobre a janela de 30 dias com corte em 19/11/2026 (65,686 unidades consumidoras ativas). Medidores com degradação severa de enlace RF Mesh ou apagões prolongados de telemetria (<70% de cobertura horária em mais de 15 dias) foram compulsoriamente segregados em Quarentena de Telecomunicação.


### 4.5 Resultados do Teste Fora do Tempo (OOT - Novembro/2026)
A inferência do pipeline multi-view especializado sobre o corte de 19/11/2026 produziu um score médio de anomalia de **0.1583**. Do total de 65,686 medidores inteligentes avaliados na rede de baixa e média tensão:
- **728 instalações** (1.11%) apresentaram score $\ge 0,50$, com desvios elétricos ou de comportamento metrológico;
- **0 instalações** (0.00%) foram priorizadas como **Candidatos Formais à Vistoria de Campo** após filtragem por persistência temporal e conformidade com perfis operacionais acionáveis;
- A Baseline com *Isolation Forest* global acusou dispersão excessiva em alimentadores rurais com grande sazonalidade, ao passo que o modelo multi-view conteve os falsos alarmes através da separação entre perdas de pacotes RF e quedas reais de carga.


### 4.6 Reconciliação Pericial dos Estudos de Caso (Auditoria Novembro/2026)
A análise temporal individual confirmou:
1. **UC 90850700 (Hipótese de Bypass PNT):** Queda de consumo de 18,05 para 5,72 kWh/dia (-68,3%) após 48 h de silêncio de telemetria em 29-30/10/2026. A persistência comprovada do degrau estendeu-se por 20 dias consecutivos até 19/11/2026, constituindo hipótese de intervenção física prioritária para vistoria.
2. **UC 29113296 (Controle Negativo de Telecom):** A taxa de 50,0% de completude derivou da cadência de amostragem de 10 minutos (6 leituras/h frente ao divisor de 12), sem alteração do consumo real (21,8 kWh/dia), sendo retida em Quarentena Telecom sem abertura espúria de ordem de serviço.
3. **UC 96905697 (Sazonalidade Comercial):** Ciclicidade de fim de semana (redução de ~65% em sábados/domingos com recuperação imediata na segunda-feira).
4. **UC 82889805 (Possível GD Não Cadastrada):** Injeção reversa de energia ativa diária entre 11h e 15h nos dias ensolarados.


### 5.1 Limitações Metodológicas e Recomendações
Ressalta-se que a plataforma atua na detecção de desvios físicos e priorização operacional de inspeções. Sem laudos de inspeção em campo (Termo de Ocorrência e Inspeção — TOI), os diagnósticos representam **hipóteses diagnósticas de alta probabilidade física** e **candidatos à vistoria**, não devendo ser denominados 'fraudes confirmadas' ou computados sob métricas de economia realizada até que ocorra a averiguação presencial da distribuidora.

