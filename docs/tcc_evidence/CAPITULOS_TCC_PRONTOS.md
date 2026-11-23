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

Na campanha de avaliação fora do tempo (*out-of-time* - OOT em 19/11/2026) sobre **65,686 UCs únicas** (**62,857 elegíveis** e **2,829 em quarentena técnica**), o comitê especializado multi-visão identificou **6,333 candidatos operacionais à vistoria** sob critérios rigorosos de persistência e materialidade física. Paralelamente, o monitoramento diário da carteira (`DAILY_RULES`, 01/11 a 19/11/2026, totalizando **1,248,501 janelas** `UC × cutoff`) registrou uma média de **5,387.7 candidatos/dia**.

Os resultados completos de ablação sequencial ($M_0 \to M_5$, com $M_4 = 4.390 \to M_5 = 4.390$ para UCs contextualizadas topologicamente), sensibilidade de gating ($50\%$ a $90\%$, com 70% retendo 62.857 UCs elegíveis e interceptando 915 candidatos em quarentena) e caracterização exploratória encontram-se auditados nas tabelas e figuras anexas sob os manifestos de execução versionados em `experiments/runs/`.

---

## Capítulo Z — Estudos de Caso Periciais e Diagnóstico Diferencial (Hipóteses Físicas vs Telecomunicação)

A eficácia do sistema reside na sua capacidade de resolver o diagnóstico diferencial entre eventos de rede de comunicação e anomalias físicas com evidência empírica auditável:

### Z.1 UC 90850700 — Degrau Descendente com Injeção Reversa (Hipótese de GD Não Cadastrada / Adulteração Física)
Na instalação `UC 90850700` (Alimentador Óleos Brasil), o detector CUSUM identifica um degrau descendente abrupto de **$-68.3\%$** em **31/10/2026**, reduzindo a média diária de consumo ativo de $18.05\text{ kWh/dia}$ para $\sim 5.72\text{ kWh/dia}$, com persistência contínua por **20 dias seguidos** até o corte de 19/11/2026 (sendo 12 dias com injeção reversa $RA$ ativa de 14 a 31 kWh/dia).

A decomposição temporal revela que entre 28/10 21h e 30/10 21h (exatas 48 horas) ocorreu silêncio de telemetria. Em 31/10, o registro cadastral do CIS foi atualizado administrativamente, mas no mesmo medidor físico (mesmo NIO). Ao reaparecer, a instalação passa a apresentar forte injeção de energia ativa reversa nas horas de sol e retração severa de consumo forward.

Com **score individual de anomalia de 0.9601** (frente à média populacional de 0.2944), o sistema corrigiu o roteamento anterior de ciclo de vida (que incorretamente a rotulava como `INICIACAO_PERFIL_UC_NOVA`), reclassificando-a sob a hipótese diagnóstica de `POSSIVEL_GD_NAO_CADASTRADA` e priorizando-a para vistoria de campo com máxima urgência (`IS_CANDIDATO_VISTORIA = SIM`), sem emitir conclusão causal antecipada de fraude antes do laudo presencial (TOI).

### Z.2 UC 29113296 — O Falso Positivo Evitado via Gating de Telecomunicação (Controle Negativo de Telecom)
Na instalação `UC 29113296`, o histórico diário atesta consumo elétrico regular e estável ao redor de $21.8\text{ a }28.0\text{ kWh/dia}$, com $100\%$ de integridade do registrador metrológico acumulador. Todavia, a telemetria intervalar horária acusa completude média de exatamente $50.0\%$.

A completude de 50,0% é compatível com uma cadência de amostragem de 10 minutos (6 leituras/h frente à grade nominal de 5 minutos / 12 leituras/h), embora o arquivo diário reporte `CADENCE_MINUTES: None`. Não se afirma causa física de reconfiguração de firmware sem auditoria direta nos timestamps intervalares brutos.

Sem o filtro de qualidade, o algoritmo tradicional somaria os intervalos faltantes e geraria um falso alarme de queda de consumo. A camada de *Gating de Quarentena* (`meta__telecom_status == "QUARANTINE"`) intervém e bloqueia o despacho de campo para vistoria de perdas comerciais (`IS_CANDIDATO_VISTORIA = NÃO`), direcionando a UC, se necessário, para a fila técnica de manutenção de telecomunicações (`DESTINO_OPERACIONAL = "MANUTENCAO_TECNICA_TELECOM_INDIVIDUAL"`).

---
## Adendo Experimental: Avaliação em Novembro/2026

### 3.4 Protocolo de Avaliação Temporal com Dados Reais de Novembro/2026
A validação do modelo com dados de campo foi executada sobre os cortes temporais de 01/11/2026 a 19/11/2026 da concessão de Araucária (PR). Para mitigar qualquer risco de vazamento temporal (*look-ahead bias*), o conjunto de treino utilizou estritamente os cortes mensais consolidados de Agosto, Setembro e Outubro/2026 (189.341 janelas limpas), enquanto a avaliação fora do tempo (*out-of-time* - OOT) foi processada sobre a janela com corte em 19/11/2026 (65.686 unidades consumidoras ativas). Medidores com degradação severa de enlace RF Mesh ou apagões prolongados de telemetria (<70% de cobertura horária média) foram compulsoriamente segregados em Quarentena de Telecomunicação (2.829 UCs).

### 4.5 Resultados do Teste Fora do Tempo (OOT - Novembro/2026)
A inferência do pipeline multi-view especializado sobre o corte congelado de 19/11/2026 produziu um score médio de anomalia de **0.2944** (P95 = 0.9115). Do total de 65.686 medidores inteligentes avaliados na rede de baixa e média tensão:
- **62.857 instalações** (95.69%) foram declaradas tecnicamente elegíveis para avaliação de perdas;
- **2.829 instalações** (4.31%) foram segregadas em Quarentena Técnica de Telecomunicação;
- **6.333 instalações** (9.64% da base total) foram priorizadas como **Candidatos Operacionais à Vistoria de Campo** com base em anomalias elétricas persistentes e hipóteses físicas delimitadas (PNT, possível GD não cadastrada e medidor travado);
- A Baseline com *Isolation Forest* global (limiar congelado de treino de 0.643) indicou **3.122 anomalias**, demonstrando que o ensemble multi-view e o pós-processamento de regras capturam um volume operacional consistente, ao mesmo tempo em que fornecem categorização física acionável.

### 4.6 Reconciliação Pericial dos Estudos de Caso (Auditoria Novembro/2026)
A análise temporal individual confirmou:
1. **UC 90850700 (Hipótese de GD Não Cadastrada / PNT):** Queda de consumo FA de 18,05 para 5,72 kWh/dia (-68,3%) e 12 dias de injeção reversa RA ativa (14 a 31 kWh/dia) após silêncio de telemetria em 29-30/10/2026. A persistência comprovada do degrau estendeu-se por 20 dias consecutivos até 19/11/2026 (score elétrico individual 0.9601), constituindo hipótese física prioritária para vistoria.
2. **UC 29113296 (Controle Negativo de Telecom):** A taxa de 50,0% de completude é compatível com cadência de amostragem de 10 minutos (6 leituras/h frente ao divisor de 12), sem alteração do consumo real estável (21,8 a 28,0 kWh/dia e 100% integridade no registrador), sendo retida em Quarentena Telecom sem despacho espúrio para perdas comerciais.
3. **UC 96905697 (Sazonalidade Comercial):** Ciclicidade de fim de semana (redução de ~65% em sábados/domingos com recuperação imediata na segunda-feira).
4. **UC 82889805 (Possível GD Não Cadastrada):** Injeção reversa de energia ativa diária entre 11h e 15h nos dias ensolarados (fluxo reverso respondendo por expressiva parcela do balanço).

### 5.1 Limitações Metodológicas e Recomendações
Ressalta-se que a plataforma atua na detecção de desvios físicos e priorização operacional de inspeções. Sem laudos de inspeção em campo (Termo de Ocorrência e Inspeção — TOI), os diagnósticos representam **hipóteses diagnósticas de alta probabilidade física** e **candidatos operacionais à vistoria**, sendo vedada qualquer alegação de 'fraude confirmada', 'precisão de campo' ou 'redução real de falsos positivos' até que ocorra a averiguação presencial da distribuidora.

