# Catálogo de Features — ARAUCARIA Smart Meter TCC

> **Documento**: `docs/feature_catalog.md`  
> **Versão**: 1.0  
> **Atualizado**: 2025-07-16  
> **Rascunho**: Sim

## 1. Visão Geral

Este catálogo documenta todas as features extraídas dos dados normalizados de medidores inteligentes (smart meters) para a região de Araucária. As features são calculadas pela função `src.features.meter_day.build_meter_day_features()` e organizadas em grupos conceituais.

### Convenções

| Símbolo | Significado |
|---------|-------------|
| 📊 | Feature derivada de dados intervalares (5 min) |
| ⚡ | Feature derivada de dados instantâneos |
| 📋 | Feature derivada de snapshot de registradores |
| 📐 | Feature derivada de dados cadastrais/metadados |
| 🏷️ | Feature derivada de histórico de labels |

## 2. Features de Cobertura e Qualidade

```python
# Grupo: coverage_features
# Fonte: src.features.quality
```

| Feature | Tipo | Descrição | Fórmula / Definição |
|---------|------|-----------|---------------------|
| `COVERAGE_RATIO` | 📊 Float [0,1] | Fração de pontos esperados presentes | `actual_points / expected_points` |
| `NULL_RATIO` | 📊 Float [0,1] | Fração de valores nulos nas colunas de energia | `null_count / total_count` |
| `QUALITY_SCORE` | 📊 Float [0,1] | Score composto de qualidade | `0.4 × coverage + 0.3 × completeness + 0.3 × physical_compliance` |
| `IS_ESTIMATED_RATIO` | 📊 Float [0,1] | Fração de pontos marcados como estimados | `sum(IS_ESTIMATED) / total_points` |

### Critérios de Qualidade

O `QUALITY_SCORE` combina três dimensões:
- **Cobertura** (40%): `COVERAGE_RATIO` — presença de dados esperados
- **Completude** (30%): `1 - NULL_RATIO` — ausência de valores nulos
- **Conformidade Física** (30%): fração de pontos dentro de faixas físicas esperadas (ex.: U_L [0,500] V, I_L [0,1000] A, FA_INTERVAL [0,100] kWh)

## 3. Features de Energia

```python
# Grupo: energy_features
# Fonte: src.features.electrical
```

| Feature | Tipo | Descrição | Fórmula / Definição |
|---------|------|-----------|---------------------|
| `FA_ENERGY_DAY` | 📊 Float (kWh) | Energia ativa total no dia | `sum(FA_INTERVAL)` |
| `RA_ENERGY_DAY` | 📊 Float (kWh) | Energia reativa total no dia | `sum(RA_INTERVAL)` |
| `ENERGY_IMBALANCE_RATIO` | 📊 Float | Razão reativo/ativo | `RA_ENERGY_DAY / FA_ENERGY_DAY` (evita divisão por zero) |
| `FA_MAX_INTERVAL` | 📊 Float (kWh) | Máximo FA_INTERVAL no dia | `max(FA_INTERVAL)` |
| `RA_MAX_INTERVAL` | 📊 Float (kWh) | Máximo RA_INTERVAL no dia | `max(RA_INTERVAL)` |

## 4. Features de Carga

```python
# Grupo: load_features
# Fonte: src.features.electrical
```

| Feature | Tipo | Descrição | Fórmula / Definição |
|---------|------|-----------|---------------------|
| `LOAD_FACTOR` | 📊 Float [0,1] | Fator de carga médio/demanda máxima | `avg(FA_INTERVAL) / max(FA_INTERVAL)` |
| `PEAK_HOUR` | 📊 Int [0..23] | Hora de pico de demanda | Hora com maior `FA_INTERVAL` |
| `ENERGY_RAMP_MAX` | 📊 Float (kWh) | Máxima rampa de energia entre pontos consecutivos | `max(abs(diff(FA_INTERVAL)))` |
| `LOAD_VARIABILITY` | 📊 Float | Coeficiente de variação da carga | `std(FA_INTERVAL) / avg(FA_INTERVAL)` |
| `AUTOCORRELATION_LAG1` | 📊 Float [-1,1] | Autocorrelação de FA_INTERVAL com lag 1 | Correlação de Pearson entre série e série deslocada por 1 período |

### Fator de Carga

O `LOAD_FACTOR` mede a eficiência do uso da capacidade:
- **1.0**: Carga perfeitamente constante
- **< 0.5**: Alta variação com pico bem acima da média
- **Próximo de 0**: Período com consumo quase zero na maior parte do tempo

## 5. Features de Tensão e Corrente

```python
# Grupo: voltage_current_features
# Fonte: src.features.electrical
```

| Feature | Tipo | Descrição | Fórmula / Definição |
|---------|------|-----------|---------------------|
| `U_MEAN` | ⚡ Float (V) | Média das tensões trifásicas | `mean(U_L1, U_L2, U_L3)` |
| `VOLTAGE_IMBALANCE` | ⚡ Float (V) | Máximo desvio da média das fases | `max(abs(U_Lx - U_MEAN))` |
| `I_MEAN` | ⚡ Float (A) | Média das correntes trifásicas | `mean(I_L1, I_L2, I_L3)` |
| `CURRENT_IMBALANCE` | ⚡ Float (A) | Máximo desvio da média das correntes | `max(abs(I_Lx - I_MEAN))` |
| `U_MIN` | ⚡ Float (V) | Mínimo entre U_L1, U_L2, U_L3 | `min(U_L1, U_L2, U_L3)` |
| `U_MAX` | ⚡ Float (V) | Máximo entre U_L1, U_L2, U_L3 | `max(U_L1, U_L2, U_L3)` |
| `I_MAX` | ⚡ Float (A) | Máximo entre I_L1, I_L2, I_L3 | `max(I_L1, I_L2, I_L3)` |

### Desbalanceamento

O `VOLTAGE_IMBALANCE` (e análogo para corrente) mede o quão desbalanceadas estão as fases:

```
U_MEAN = mean(U_L1, U_L2, U_L3)
VOLTAGE_IMBALANCE = max(|U_L1 - U_MEAN|, |U_L2 - U_MEAN|, |U_L3 - U_MEAN|)
```

Valores altos podem indicar problemas de qualidade de energia ou carga desbalanceada.

## 6. Features de Registradores

```python
# Grupo: register_features
# Fonte: src.features.electrical
```

| Feature | Tipo | Descrição | Fórmula / Definição |
|---------|------|-----------|---------------------|
| `FA_ACCUMULATED` | 📋 Float (kWh) | Energia ativa acumulada no dia (fim - início) | `last(FA_ENERGY_D) - first(FA_ENERGY_D)` |
| `RA_ACCUMULATED` | 📋 Float (kWh) | Energia reativa acumulada no dia | `last(RA_ENERGY_D) - first(RA_ENERGY_D)` |
| `MD_MAX` | 📋 Float (kW) | Demanda máxima no dia | `max(FA_MD_D)` |
| `RA_REVERSAL_RATIO` | 📋 [0,1] | Fração de leituras com reversão de RA negativa | `negative_RA_readings / total_RA_readings` |

### Reversão de RA

A `RA_REVERSAL_RATIO` indica a fração de leituras do registrador de energia reativa (`RA_ENERGY_D`) em que o valor é negativo — o que sugere fluxo reverso de reativo (injeção de reativo na rede).

## 7. Features de Metadados

```python
# Grupo: metadata_features
# Fonte: src.features.meter_day
```

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `METER_TYPE` | 📐 String | Tipo do medidor (ex.: Eletrônico, Eletromecânico) |
| `METER_CLASS` | 📐 String | Classe do medidor |
| `INSTALLATION_YEAR` | 📐 Int | Ano de instalação |
| `MUNICIPALITY` | 📐 String | Município (ARAUCARIA) |
| `GEO_LAT` | 📐 Float | Latitude da UC |
| `GEO_LON` | 📐 Float | Longitude da UC |
| `TRANSFORMER_ID` | 📐 String | Identificador do transformador |
| `FEEDER_ID` | 📐 String | Identificador do alimentador |

## 8. Nomenclatura de Colunas

### Prefixos

| Prefixo | Significado |
|---------|-------------|
| `FA_` | (F) Forward Active — Energia Ativa |
| `RA_` | (R) Reverse Active — Energia Reativa |
| `U_` | Voltage — Tensão |
| `I_` | Current — Corrente |
| `MD_` | Maximum Demand — Demanda Máxima |
| `Q_` | Apparent/Reactive Power — Potência Reativa |

### Sufixos

| Sufixo | Significado |
|--------|-------------|
| `_L1`, `_L2`, `_L3` | Fase 1, 2, 3 |
| `_C` | Intervalar (5 min) |
| `_D` | Acumulador diário (registrador) |
| `_INTERVAL` | Valor por intervalo de 5 min |
| `_INSTANT` | Instantâneo (tensão/corrente no momento) |
| `_DAY` | Agregação diária |

## 9. Versionamento

Features são versionadas por tag semântica (ex.: `v1`, `v2`). O versionamento permite:

- Rastrear quais features foram usadas em cada modelo treinado
- Garantir reprodutibilidade: um modelo treinado com features `v1` sempre usará as mesmas definições
- Evoluir features sem quebrar modelos em produção

O versionamento é gerenciado pelo parâmetro `feature_set_version` em `build_meter_day_features()` e registrado no manifesto de execução.

## 10. Glossário

| Termo | Definição |
|-------|-----------|
| **FA** | Forward Active — Energia Ativa (consumo real) |
| **RA** | Reverse Active — Energia Reativa |
| **MD** | Maximum Demand — Demanda Máxima |
| **NIO** | Número de Identificação do Medidor |
| **UC** | Unidade Consumidora |
| **Cadência** | Intervalo entre medições (ex.: 5 min → 288 pts/dia) |
| **Grain** | Granularidade única de uma tabela (ex.: NIO + TIMESTAMP_UTC) |
