# Classificação de Distúrbios de Tensão e Desbalanceamento em Smart Meters

> **Módulo:** `src/models/voltage_classifier.py`  
> **Testes:** `tests/test_voltage_classifier.py`  
> **Normas de Referência:** ANEEL PRODIST Módulo 8 (Qualidade da Energia Elétrica) e IEEE Std 1159 (Monitoring Electric Power Quality)  
> **Status:** Implementado (v1.0)  
> **Data:** Outubro de 2026

---

## 1. Visão Geral e Motivação

Nos sistemas de medição inteligente (AMI), as grandezas de tensão são os sinais elétricos mais ricos e diretos para diagnosticar o estado operacional da rede secundária e das Unidades Consumidoras (UCs). Diferente da corrente e da energia ativa — que sofrem forte influência do comportamento volátil de consumo do cliente —, **a tensão reflete a física do circuito**, a integridade dos ramais de ligação e eventos de contingência na rede.

Este módulo implementa uma arquitetura híbrida (regras determinísticas da norma + modelo de *Machine Learning* supervisionado com tratamento nativo de dados ausentes) para classificar:
1. **Níveis de Tensão:** Faixas de conformidade regulatória (Adequada, Subtensão/Afundamento, Sobretensão/Elevação).
2. **Tipos de Faltas Elétricas:** Quedas de fase individuais (Falta Monofásica, Falta Bifásica) e desligamentos completos da rede (Interrupção Total).
3. **Desbalanceamento de Fases:** Desequilíbrios críticos inter-fases que sobrecarregam neutros e degradam equipamentos.
4. **Anomalias de Telemetria:** Diferenciação entre apagão físico na rede vs. falha isolada de comunicação/chip de telemetria.

---

## 2. O Papel do Desbalanceamento

O **desbalanceamento de tensão** desempenha um duplo papel nesta arquitetura:

1. **Como Feature Diferenciadora:**
   O desbalanceamento instantâneo relativo ($V_{\text{imb}}$) atua como um separador imediato entre eventos simétricos (ex.: subtensão generalizada em todo o alimentador) e assimetrias severas (ex.: perda de fase, ramal partido, neutro rompido ou cargas monofásicas desproporcionais).
   
   A fórmula adotada segue o padrão internacional da IEEE Std 1159 e do PRODIST Módulo 8:
   $$\bar{V}(t) = \frac{1}{K} \sum_{k \in \text{fases ativas}} U_{Lk}(t)$$
   $$V_{\text{imb}}(t) = \frac{\max_k |U_{Lk}(t) - \bar{V}(t)|}{\bar{V}(t)}$$

2. **Como Classe Alvo (*Target*):**
   Quando as tensões das fases permanecem em níveis operacionais normais ($V \ge 0.90 \text{ p.u.}$), mas a assimetria entre fases ultrapassa os limites regulatórios ($V_{\text{imb}} \ge 2\%$), o evento é categorizado como `DESBALANCEAMENTO_CRITICO`, apontando degradação da qualidade ou conexão irregular sem interrupção de fornecimento.

---

## 3. O Desafio Central: O Que Fazer com Dados Faltantes?

Em redes reais de distribuição, medições de tensão ausentes ou zeradas ocorrem frequentemente por motivos completamente distintos. Tratar todo valor nulo simplesmente preenchendo com zero ou com a média distorceria gravemente o diagnóstico elétrico. 

A arquitetura segmenta os dados faltantes em **três categorias físicas bem delimitadas**:

```
                              ┌────────────────────────────────────────┐
                              │               ┌────────────────────────────────────┴────────────────────────────────────┐
             │                                                                         │
    ┌────────▼─────────┐                                                      ┌────────▼─────────┐
    │ Null Estrutural  │                                                      │  Medição 0.0V    │
    │ (Fase não existe)│                                                      │ (Falta de Fase)  │
    └────────┬─────────┘                                                      └────────┬─────────┘
             │                                                                         │
    `IS_ACTIVE_L1..L3`                                                         `IS_ZERO_L1..L3`
    (Contexto da UC:                                                           (Tensão <= 5.0V com
     Monofásica/Bifásica)                                                       medidor energizado)
```

### 3.1. Null Estrutural (Fase Não Existente no Cadastro)
- **Causa:** Uma UC **monofásica** possui fisicamente apenas a fase $L1$ conectada; os canais $L2$ e $L3$ constarão como nulos no MDM. Uma UC **bifásica** não possui a fase $L3$.
- **Tratamento:** A função `extract_voltage_classifier_features()` cruza a medição com o cadastro estático da UC (`FASE_CIRCUITO` em `uc_context`) ou infere a ativação a partir das leituras físicas presentes por linha. Cria os flags booleanos:
  - `IS_ACTIVE_L1`, `IS_ACTIVE_L2`, `IS_ACTIVE_L3`
  - Fases estruturalmente inativas são descartadas do cálculo de desbalanceamento e não contam como "falta de energia".

### 3.2. Medição Zero ($V \le 5.0\text{V}$) com Medidor Energizado
- **Causa:** O medidor está energizado por outra fase (ou por bateria/capacitor) e transmitindo dados normalmente, mas uma ou duas fases registram valor próximo de zero ($\le 5\text{V}$).
- **Tratamento:** Criação dos indicadores `IS_ZERO_L1`, `IS_ZERO_L2`, `IS_ZERO_L3`.
  - Se 1 de 2 ou 3 fases ativas estiver zerada $\rightarrow$ `FALTA_MONOFASICA` (ex.: queima de fusível em ramal).
  - Se 2 de 3 fases ativas estiverem zeradas $\rightarrow$ `FALTA_BIFASICA`.
  - Se todas as fases ativas estiverem zeradas $\rightarrow$ `INTERRUPCAO_TOTAL` (queda física de energia na UC).

### 3.3. Segregação de Domínio: Telemetria Ausente vs. Qualidade da Tensão PRODIST M8
- **Separação Arquitetural:** Pacotes de telemetria sem medições de tensão (ausência de dados de telecomunicação) pertencem ao domínio de **Saúde da Infraestrutura AMI e Redes de Telecomunicação ($S_{comm}$ no Multi-View)**.
- O classificador PRODIST Módulo 8 foca com precisão matemática nas **perturbações físicas da onda de tensão**. Registros sem medição de tensão são descartados antes do treinamento físico, evitando contaminação da qualidade de energia por artefatos de comunicação.

---

## 4. Normalização em p.u. (Per-Unit)

Para permitir que o modelo aprenda padrões universais independentemente da classe de tensão nominal (redes de 127V fase-neutro, 220V fase-fase, ou primário em 13.8kV), todas as tensões são convertidas para o sistema *per-unit* ($V_{\text{pu}}$):

$$U_{Lk,\text{pu}} = \frac{U_{Lk}}{\text{TENSAO\_BASE}}$$

Onde $\text{TENSAO\_BASE}$ é obtida diretamente do cadastro da UC (`uc_context.parquet`), com *fallback* padrão para 127.0 V se ausente.

---

## 5. Classes Alvo (Enum `VoltageDisturbanceType`)

| Classe | Descrição | Condição Típica ($V_{\text{pu}} = V / V_{\text{base}}$) |
| :--- | :--- | :--- |
| **`NORMAL`** | Fornecimento adequado e equilibrado | $0.93 \le V_{\text{pu}} \le 1.05$ em todas as fases ativas e $V_{\text{imb}} < 2\%$ |
| **`SUBTENSAO`** | Afundamento de tensão (*voltage sag*) | $0.05 < V_{\text{pu}} < 0.90$ em uma ou mais fases ativas |
| **`SOBRETENSAO`** | Elevação de tensão (*voltage swell*) | $V_{\text{pu}} > 1.05$ em uma ou mais fases |
| **`FALTA_MONOFASICA`** | Queda total em apenas 1 fase | 1 fase ativa com $V \le 5\text{V}$, demais fases operacionais |
| **`FALTA_BIFASICA`** | Queda total em 2 fases | 2 fases ativas com $V \le 5\text{V}$, 1 fase operacional |
| **`INTERRUPCAO_TOTAL`** | Apagão ou desligamento geral da UC/rede | Todas as fases ativas em $0\text{V}$ OU alarme *Last Gasp* confirmado |
| **`DESBALANCEAMENTO_CRITICO`** | Assimetria severa inter-fases | $V_{\text{imb}} \ge 3\%$ (PRODIST Seção 8.2) com tensões $\ge 0.90 \text{ p.u.}$ |

---

## 6. Estratégia de Treinamento e Teste (Baseline Canônico de 81 Dias)

### 6.1. Padrão de Treinamento Histórico
- O período de **81 dias** do Data Lake (01/08/2026 a 20/10/2026) é cadastrado como o **Padrão de Treinamento Oficial (Baseline de Treinamento)**.
- **100% dos dados desta janela** são utilizados para ajustar o classificador (`test_size = 0.0`), garantindo máxima cobertura estatística de todos os alimentadores e eventos de perturbação raros.

### 6.2. Estratégia de Teste: Validação Temporal Contínua (Out-of-Time)
- O teste e avaliação de generalização são executados **fora do tempo** (*out-of-time*) sobre novos dados coletados em períodos futuros/subsequentes, refletindo perfeitamente a operação real em produção.

---

## 6. Modelos: Baseline Heurístico vs. Machine Learning

### 6.1. Baseline Determinístico (`RuleBasedVoltageClassifier`)
- Implementa diretamente as equações de limiares físicos e regulatórios do PRODIST Módulo 8.
- Serve como rotulador *cold-start* (gera rótulos de treino a partir de medições brutas, sem depender de marcações manuais).
- Totalmente transparente, auditável e determinístico.

### 6.2. Modelo de Machine Learning (`MLVoltageDisturbanceClassifier`)
- Baseado em **`HistGradientBoostingClassifier`** da Scikit-Learn:
  - **Suporte Nativo a NaNs:** Lida diretamente com valores ausentes sem necessidade de imputações artificiais agressivas que mascarem a ausência da fase.
  - **Classificação Multiclasse Não-Linear:** Aprende fronteiras suaves para estados limítrofes ("quase desbalanceados" ou transientes).
  - **Probabilidades por Classe e Confiança:** Retorna não apenas a classe prevista (`PREDICTED_DISTURBANCE_CLASS`), mas o vetor completo de probabilidades (`PROB_NORMAL`, `PROB_SUBTENSAO`, etc.) e o nível de certeza da previsão (`DISTURBANCE_CONFIDENCE`).

---

## 7. Catálogo de Features Extraídas

As seguintes colunas são geradas por `extract_voltage_classifier_features()`:

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `U_L1_PU`, `U_L2_PU`, `U_L3_PU` | Float64 | Tensão de cada fase normalizada em per-unit |
| `V_MIN_PU`, `V_MAX_PU`, `V_AVG_PU` | Float64 | Tensão mínima, máxima e média entre as fases ativas |
| `VOLTAGE_IMBALANCE_RATIO` | Float64 | Razão máxima de desbalanceamento em relação à média ($V_{\text{imb}}$) |
| `ACTIVE_VALID_PHASES_COUNT` | Int32 | Número de fases ativas cadastradas com medição válida |
| `IS_ACTIVE_L1`, `IS_ACTIVE_L2`, `IS_ACTIVE_L3` | Boolean | Se a fase existe estruturalmente na instalação da UC |
| `IS_NULL_L1`, `IS_NULL_L2`, `IS_NULL_L3` | Boolean | Se a medição da fase veio ausente (nula) |
| `IS_ZERO_L1`, `IS_ZERO_L2`, `IS_ZERO_L3` | Boolean | Se a fase registrou tensão zerada ($\le 5\text{V}$) |
| `NEIGHBOR_COVERAGE_RATIO` | Float64 | Cobertura média de transmissão dos vizinhos no mesmo transformador |
| `HAS_POWER_ALARM` | Boolean | Se houve alarme de falta de energia no medidor |

---

## 8. Guia Rápido de Uso

```python
from __future__ import annotations
import polars as pl
from src.models.voltage_classifier import (
    extract_voltage_classifier_features,
    RuleBasedVoltageClassifier,
    MLVoltageDisturbanceClassifier,
)

# 1. Carregar medições instantâneas e contextos
instantaneous_df = pl.read_parquet("output/measurements/ami_instantaneous/data.parquet")
context_df = pl.read_parquet("output/context/uc_context.parquet")
neighborhood_df = pl.read_parquet("output/features/neighborhood_features.parquet")

# 2. Extrair features e indicadores de dados faltantes
features_df = extract_voltage_classifier_features(
    instantaneous_df=instantaneous_df,
    context_df=context_df,
    neighborhood_df=neighborhood_df,
)

# 3. Gerar rótulos da norma com a Baseline Heurística
labeled_df = RuleBasedVoltageClassifier.classify(features_df)

# 4. Treinar o classificador supervisionado
clf = MLVoltageDisturbanceClassifier(random_state=42)
clf.fit(labeled_df, target_column="BASELINE_VOLTAGE_LABEL")

# 5. Fazer predições com confiança e probabilidades
result = clf.predict(features_df)
predictions_df = result.predictions_df

# Exemplo de seleção de saídas analíticas
print(predictions_df.select([
    "UC",
    "PREDICTED_DISTURBANCE_CLASS",
    "DISTURBANCE_CONFIDENCE",
    "PROB_INTERRUPCAO_TOTAL",
    "PROB_DESBALANCEAMENTO_CRITICO",
]))
```
