# Relatório Técnico de Resultados Experimentais — TCC AMI Copel Araucária

**Título Científico:** *Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches*  
**Timestamp da Execução:** `2026-11-20T14:50:17.122044` | **Modo:** `full` | **Seed:** `42`

---

## 1. Reconciliação de Granularidade e KPIs do Cockpit

| Indicador | Granularidade | Valor Observado |
|:---|:---|---:|
| UCs Únicas Analisadas | UC (`id__uc_id` último cutoff) | **66,099** |
| Janelas Avaliadas | `UC × cutoff_date` | **262,849** |
| UCs c/ Anomalia Detectada | UC (último cutoff) | **932** (1.41%) |
| Ocorrências de Anomalia em Janelas | `UC × cutoff_date` | **2,629** (1.0%) |
| Candidatos a Vistoria (Campo) | UC (`IS_CANDIDATO_VISTORIA == True`) | **274** |
| UCs c/ Possível GD Não Cadastrada | UC (último cutoff) | **201** (704 ocorrências) |
| UCs em Quarentena Telecom | UC (último cutoff) | **2,939** (10,422 ocorrências) |

## 2. Auditoria Quantitativa do Data Lake

- **UCs distintas no histórico de medidores:** 69895
- **NIOs distintos no histórico:** 72661
- **UCs com múltiplos medidores históricos:** 1288
- **NIOs reutilizados em múltiplas UCs:** 46
- **Alimentadores distintos na hierarquia:** 23
- **Transformadores / Postos distintos:** 0
- **Cobertura média / mediana na população:** 0.9216 / 0.991

## 3. Benchmark Controlado de Engenharia de Dados (Polars Eager vs Lazy/Sink)

- **Linhas avaliadas (`classified_ucs_latest.parquet`):** 262849
- **Eager Read+Write Zstd:** 0.1487 s (1767748 linhas/s)
- **Lazy Scan + `sink_parquet` Zstd:** 0.1232 s (2133447 linhas/s)

## 4. Experimentos de Machine Learning e Estabilidade de Normalização

- **Coorte avaliada:** 66099 UCs (treino limpo: 62766 UCs)
- **Features elétricas puras (`select_pure_electrical_features`):** 16
- **Tempo de ajuste (`CompositeAnomalyDetector` 6 detectores):** 5868.55 ms | **Pico RAM:** 57.163 MB
- **Teste de Estabilidade (Remoção de 1% extremos do lote):**
  - Deslocamento máximo do Min-Max dependente do lote: `0.3253`
  - Deslocamento com referência congelada P01–P99: `0.0`

## 5. Sensibilidade do Gating (`50%` a `90%`) e Ablação (`M0 → M5`)

- Resultados tabulares completos em `gating/gating_sensitivity.csv` e `ablation/ablation_funnel_metrics.csv`.
- **Incidência de cobertura degradada (<70%) entre anomalias brutas:** 74.03%
- **Incidência de cobertura degradada (<70%) nos candidatos finais M5:** 11.68%

## 6. EDA, Comportamento, Estudos de Caso, Operação e Economia

- **Figuras EDA reais geradas:** 12 (ver `eda/EDA_REPORT.md`)
- **Dossiês de Raio-X anonimizados:** 9 (ver `case_studies/CASE_STUDIES_XRAY.md`)
- **Estudo de Caso Pericial (Fraude vs Telecom):** Detalhamento comparativo aprofundado entre UC 90850700 (Apagão 48h + Degrau CUSUM -68.5%) e UC 29113296 (Perda de pacotes 50% + Quarentena Telecom preventiva) disponível em [ESTUDO_DE_CASO_FRAUDE_VS_TELECOM.md](ESTUDO_DE_CASO_FRAUDE_VS_TELECOM.md).
- **Cenários de capacidade e sensibilidade econômica:** 27 cenários parametrizados em `economics/economic_sensitivity_scenarios.csv` (*cenários de sensibilidade, não resultados realizados*).


---
## Resultados Reais de Novembro/2026 (OOT)

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



---
## Resultados Reais de Novembro/2026 (OOT)

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

