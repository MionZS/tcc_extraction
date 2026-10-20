# Relatório Técnico-Científico: Classificador de Conformidade de Tensão PRODIST Módulo 8 e Detecção Não Supervisionada

> **Localização dos Módulos:**  
> - Classificador Determinístico & Modelo Supervisionado: [`src/models/voltage_classifier.py`](file:///d:/Projects/tcc_extraction/src/models/voltage_classifier.py)  
> - Pipeline de Treinamento e Avaliação: [`scripts/train_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/scripts/train_voltage_classifier.py)  
> - Motor Não Supervisionado Multitécnica (Legado/Geral): [`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py)  
> - Testes Automatizados: [`tests/test_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/tests/test_voltage_classifier.py)  
> **Normas Técnicas:** ANEEL PRODIST Módulo 8 (Qualidade da Energia Elétrica) e IEEE Std 1159 (Power Quality Monitoring)  
> **Status:** Operacional e em Calibração de Limiares  

---

## 1. Resumo Executivo (Estrutura CAR)

- **Contexto (C):**  
  Em redes elétricas de distribuição monitoradas por infraestrutura de medição inteligente (AMI), anomalias de tensão possuem forte causalidade física e regulatória (limiares de transgressão contratual definidos pelo PRODIST Módulo 8 da ANEEL). A abordagem não supervisionada inicialmente proposta (Isolation Forest, LOF, PCA) tratava desvios de tensão como outliers estatísticos genéricos. A engenharia de operação apontou que esse método agrupava falsos positivos estatísticos (ex.: ruídos de telemetria e afundamentos transitórios) e não distinguia transgressões regulatórias críticas de perdas de pacote de comunicação.

- **Ação (A):**  
  Instituiu-se formalmente o **Classificador de Distúrbios e Desbalanceamento PRODIST**, combinando:
  1. Um **módulo determinístico** estrito baseado nas regras e janelas do PRODIST M8 / IEEE 1159 com tratamento físico de dados faltantes (nulos estruturais vs. faltas de fase vs. desconexão de telemetria cruzada com transformadores e alarmes).
  2. Um **modelo supervisionado** (`HistGradientBoostingClassifier`) treinado sobre os estados físicos, projetado para refinar as fronteiras difusas (*fuzzy boundaries*) de subtensão e intermitência de desbalanceamento por meio de calibração de limiares probabilísticos.

- **Resultado (R):**  
  Em teste empírico com ~50.000 registros reais de telemetria de alimentadores da Região Metropolitana de Curitiba (Araucária):
  - **Desconexão de Telemetria:** 100% de precisão e recall (11.839 casos perfeitamente isolados de apagões físicos reais via razão de pares de transformador $\le 0.30$ e alarmes *last gasp*).
  - **Faltas Monofásicas e Bifásicas:** 100% de separação física sem contaminação cruzada.
  - **Detecção Não Supervisionada vs. Determinística:** O modelo não supervisionado categorizava 95% do dataset como "normal", 4,1% como ruído de telemetria, 0,4% como perturbação da rede e apenas 0,5% como anomalia pura.
  - **Fronteira Subtensão:** Identificados 140 falsos positivos de fronteira na regra de subtensão estrita ($0.895 \text{ p.u.} \le V < 0.900 \text{ p.u.}$), com F1-Score inicial de 66,5%, motivando o ajuste dinâmico do limiar probabilístico do modelo supervisionado para aproximação do baseline normativo.

---

## 2. Motivação e Justificativa de Engenharia

### 2.1. A Sugestão da Engenharia
Durante os testes de campo com o motor não supervisionado (Isolation Forest + LOF em [`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py)), a equipe de engenharia elétrica observou que:
1. **O fenômeno de tensão é essencialmente determinístico:** Diferente da curva de carga e corrente elétrica (que varia com o comportamento do consumidor), as faixas adequadas de tensão são contratuais:
   - Tensão Adequada: $0.93 \text{ p.u.} \le V \le 1.05 \text{ p.u.}$ (em 127V: $117\text{V} \le V \le 133\text{V}$).
   - Tensão Precária: $0.87 \text{ p.u.} \le V < 0.93 \text{ p.u.}$ ou $1.05 \text{ p.u.} < V \le 1.06 \text{ p.u.}$.
   - Tensão Crítica: $V < 0.87 \text{ p.u.}$ ou $V > 1.06 \text{ p.u.}$.
2. **Dados Nulos não são Nulos Matemáticos:** Uma unidade consumidora monofásica não possui medição física nas fases $L2$ e $L3$. O modelo não supervisionado interpretava esses valores ausentes como alta anomalia multidimensional (outliers de alta dimensão), gerando alarmes espúrios.
3. **Telemetria vs. Queda de Energia:** A perda total de sinal de um medidor não significa necessariamente que a UC está sem energia elétrica. Sem cruzar com o transformador de distribuição e alarmes, o sistema não diferenciava um chip GPRS desconectado de um blackout de alimentador.

### 2.2. A Decisão Arquitetural: Criação, Nome e Localização

| Dimensão | Especificação Oficial |
|---|---|
| **Nome Oficial** | *PRODIST M8 Voltage Disturbance & Imbalance Classifier* |
| **Identificador Interno** | `RuleBasedVoltageClassifier` e `MLVoltageDisturbanceClassifier` |
| **Localização do Código** | [`src/models/voltage_classifier.py`](file:///d:/Projects/tcc_extraction/src/models/voltage_classifier.py) |
| **Pipeline de Treino/Execução** | [`scripts/train_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/scripts/train_voltage_classifier.py) |
| **Escopo de Aplicação** | Classificação em lote e streaming de pacotes AMI para detecção de conformidade e desbalanceamento crônico |

---

## 3. Comparativo Estruturado: Determinístico vs. Não Supervisionado

| Aspecto | Abordagem Determinística (`voltage_classifier.py`) | Abordagem Não Supervisionada (`composite.py`) |
|---|---|---|
| **Premissa Fundamental** | Conformidade com normas físicas e regulatórias (PRODIST M8 / IEEE 1159). | Detecção de densidade e isolamento geométrico no espaço de atributos. |
| **Tratamento de Fases** | Mapeamento explícito de cadastro (`IS_ACTIVE_L1..L3`). Descarta fases inexistentes. | Imputação estatística (média/mediana/zero), distorcendo UCs monofásicas. |
| **Separação de Apagão** | Cruza vizinhança (`NEIGHBOR_COVERAGE_RATIO <= 0.3`) e alarmes (*last gasp*). | Marca qualquer ausência prolongada como anomalia indistinta. |
| **Cálculo de Desbalanceamento** | PVUR estrito sobre fases ativas: $\max(\|V_k - \bar{V}\|)/\bar{V}$ para circuitos polifásicos. | Não calcula assimetria física; usa distâncias Euclidianas globais. |
| **Interpretabilidade** | 100% explicável: rótulos discretos associados diretamente aos artigos da norma. | Pontuação contínua de anomalia $[0.0, 1.0]$ sem causa-raiz explícita. |
| **Fronteiras de Decisão** | Rígidas (ex.: $V < 0.90 \text{ p.u.}$ é Subtensão instantânea). | Suaves e dependentes do percentil de contaminação definido na sessão. |

---

## 4. O Modelo Supervisionado e o Ajuste de Limiares (*Threshold Tuning*)

### 4.1. Por que usar Machine Learning se as regras são determinísticas?
A regra determinística pura avalia a medição como uma "foto" estática. No entanto, na operação de distribuição, medições pontuais em torno de $0.899 \text{ p.u.}$ podem representar:
- Uma carga pontual de partida de motor (transitório normal).
- Degradação contínua da regulação do transformador (subtensão sistêmica).
- Ruído de quantização do conversor analógico-digital do medidor.

O modelo supervisionado [`MLVoltageDisturbanceClassifier`](file:///d:/Projects/tcc_extraction/src/models/voltage_classifier.py#L380-L450) (`HistGradientBoostingClassifier`) foi introduzido para **aprender os padrões multivariados complexos** ao mesmo tempo em que permite calibração de limiares probabilísticos.

### 4.2. Estratégia de Ajuste de Limiares (Threshold Tuning)
Atualmente, estamos calibrando as probabilidades de predição $P(\text{classe} = k)$ para ajustar a sensibilidade do classificador:
1. **Limiar Conservador para Ações de Campo:** Transgressões com probabilidade $P \ge 0.85$ e desbalanceamento sustentado são despachadas como ordens de inspeção de conexão/transformador.
2. **Zona de Atenção Operacional:** Casos com $0.50 \le P < 0.85$ entram em lista de monitoramento de continuidade, evitando o envio prematuro de equipes para perturbações transitórias.

---

## 5. Próximo Passo Delimitado: Desbalanceamentos Intermitentes vs. Permanentes

O reconhecimento de que uma medição está fora dos parâmetros de conformidade já está consolidado e delimitado no código. A evolução técnica orientada pela engenharia é determinar o **caráter temporal** do desbalanceamento de fases:

```
                            ┌───────────────────────────────────────────┐
                            │   Evento: VOLTAGE_IMBALANCE_RATIO >= 3%   │
                            └─────────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                Janela Deslizante de 7 a 30 dias
                                (Agrupado por UC e Alimentador)
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
      Taxa de Ocorrência < 15%                                           Taxa de Ocorrência >= 15%
   (Concentrada em horários de pico)                                  (Uniforme ao longo do dia e semana)
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │ Desbalanceamento Dinâmico │                                     │ Desbalanceamento Crônico  │
   │       / Intermitente      │                                     │        / Permanente       │
   ├───────────────────────────┤                                     ├───────────────────────────┤
   │ Causa: Alocação desigual  │                                     │ Causa: Neutro rompido,    │
   │ de cargas monofásicas     │                                     │ derivação clandestina,    │
   │ no horário nobre (chuveiro│                                     │ ou problema interno no    │
   │ elétrico / ar condicionado│                                     │ transformador da rede     │
   └───────────────────────────┘                                     └───────────────────────────┘
```

### 5.1. Implementação Planejada
- **Métrica:** Razão de persistência $R_{\text{imb}} = \frac{\sum \text{horas}(V_{\text{imb}} \ge 3\%)}{\text{Total de horas ativas na janela}}$.
- **Classificação Extendida:**
  - `DESBALANCEAMENTO_INTERMITENTE`: $R_{\text{imb}} < 0.15$ (ação: rebalanceamento de ramais no circuito).
  - `DESBALANCEAMENTO_PERMANENTE`: $R_{\text{imb}} \ge 0.15$ (ação: manutenção emergencial de neutro e transformador).

---

## 6. Conclusão e Diretrizes de Uso

1. **Uso Recomendado:** O classificador PRODIST M8 deve ser a referência primária para qualquer análise de qualidade do fornecimento, auditoria de transformadores e triagem de falhas físicas no Datalake de Araucária.
2. **Papel do Modelo Não Supervisionado:** Permanece ativo apenas no segundo estágio da arquitetura para detecção de anomalias desconhecidas de consumo e fraudes volumétricas complexas (quando a tensão está normal, mas o consumo ativo/reativo apresenta padrão atípico).
3. **Execução:** O treino e a matriz de confusão atualizada podem ser gerados a qualquer momento pelo comando:
   ```bash
   uv --system-certs run python scripts/train_voltage_classifier.py
   ```
   ou pela tarefa do VS Code **`ml: train voltage classifier`**.
