# Experimento de Perturbações Controladas sobre Séries AMI Reais (P1 — Seções 13 a 16)

- **Execução:** `20261126_173223_p1_controlled_perturbations` | **Seed:** `42` | **Amostra estratificada:** `100` UCs reais em `5` circuitos (`433.068` registros intervalares de 5/10 min)
- **Integridade do Data Lake:** Leitura estritamente *read-only*; perturbações aplicadas **antes** da extração de features sobre cópias isoladas em memória (`data_lake_untouched = True`).
- **Grupo de referência (não perturbado):** cobertura média `99,3%`, taxa `S_elec >= τ` = `32,0%`, taxa CPD queda = `6,0%`, mudança de cluster 24h = `13,0%`. *(Não se presume que séries reais de referência sejam comprovadamente isentas de irregularidade.)*
- **Figuras:** `fig_p1_10` a `fig_p1_14` em `docs/tcc_evidence/p1_perturbations/figures/`.

## 1. Matriz Completa de Resultados por Cenário (23 Cenários Controlados)

| Cenário | Componente | Δ Cob. | Δ S_comm | Δ S_elec | Quarent. (%) | Anom s/Gate | Anom c/Gate | CPD <=3d (%) | Erro CPD (d) | Beh 24h >P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lacuna aleatória 10% | S_comm / Gating | -0,099 | 0,245 | -0,008 | 0,0% | 31,0% | 31,0% | 0,0% | 14,5 | 5,0% |
| Lacuna aleatória 20% | S_comm / Gating | -0,198 | 0,427 | -0,018 | 0,0% | 26,0% | 26,0% | 0,0% | 14,5 | 5,0% |
| Lacuna aleatória 30% | S_comm / Gating | -0,297 | 0,798 | -0,010 | 61,0% | 24,0% | 9,0% | 0,0% | 13,0 | 5,0% |
| Lacuna aleatória 50% | S_comm / Gating | -0,496 | 0,825 | -0,063 | 100,0% | 19,0% | 0,0% | 0,0% | N/A | 8,0% |
| Bloco consecutivo 20% (6d) | S_comm / Gating | -0,000 | 0,405 | -0,002 | 0,0% | 28,0% | 28,0% | 0,0% | 17,0 | 9,0% |
| Bloco consecutivo 30% (9d) | S_comm / Gating | -0,001 | 0,492 | 0,001 | 0,0% | 27,0% | 27,0% | 1,0% | 12,5 | 18,0% |
| Bloco consecutivo 50% (15d) | S_comm / Gating | -0,000 | 0,782 | 0,029 | 0,0% | 31,0% | 31,0% | 1,0% | 6,0 | 35,0% |
| Queda sustentada -10% | CPD / S_elec | 0,000 | 0,000 | -0,003 | 0,0% | 28,0% | 28,0% | 4,0% | 3,0 | 5,0% |
| Queda sustentada -20% | CPD / S_elec | 0,000 | 0,000 | 0,012 | 0,0% | 30,0% | 30,0% | 11,0% | 1,0 | 5,0% |
| Queda sustentada -30% | CPD / S_elec | 0,000 | 0,000 | 0,039 | 0,0% | 36,0% | 36,0% | 44,0% | 0,0 | 5,0% |
| Queda sustentada -50% | CPD / S_elec | 0,000 | 0,000 | 0,106 | 0,0% | 43,0% | 43,0% | 82,0% | 0,0 | 5,0% |
| Mudança -40% (3 dias) | CPD / Persistência | 0,000 | 0,000 | 0,017 | 0,0% | 29,0% | 29,0% | 22,0% | 2,0 | 5,0% |
| Mudança -40% (7 dias) | CPD / Persistência | 0,000 | 0,000 | 0,053 | 0,0% | 35,0% | 35,0% | 48,0% | 0,0 | 6,0% |
| Mudança -40% (14 dias) | CPD / Persistência | 0,000 | 0,000 | 0,069 | 0,0% | 41,0% | 41,0% | 70,0% | 0,0 | 5,0% |
| Evento isolado 1d (0,1x) | Contraste S_elec vs CPD | 0,000 | 0,000 | 0,002 | 0,0% | 32,0% | 32,0% | 1,0% | 7,0 | 6,0% |
| Evento isolado 1d (3,5x) | Contraste S_elec vs CPD | 0,000 | 0,000 | 0,002 | 0,0% | 32,0% | 32,0% | 2,0% | 3,0 | 6,0% |
| Deslocamento rotina +1h | S_behavior (24h) | 0,000 | 0,000 | 0,000 | 0,0% | 32,0% | 32,0% | 1,0% | 7,5 | 22,0% |
| Deslocamento rotina +2h | S_behavior (24h) | 0,000 | 0,000 | 0,000 | 0,0% | 32,0% | 32,0% | 1,0% | 7,5 | 39,0% |
| Deslocamento rotina +4h | S_behavior (24h) | 0,000 | 0,000 | 0,000 | 0,0% | 32,0% | 32,0% | 1,0% | 7,5 | 52,0% |
| Achatamento forma (kWh fixo) | PCA / Forma Diária | 0,000 | 0,000 | 0,033 | 0,0% | 35,0% | 35,0% | 1,0% | 7,5 | 21,0% |
| Injeção reversa solar (GD) | Regra GD / S_elec | 0,000 | 0,000 | 0,491 | 0,0% | 100,0% | 100,0% | 21,0% | 0,0 | 34,0% |
| Carga zero c/ tensão normal | Medidor Travado / S_elec | 0,000 | 0,000 | 0,280 | 0,0% | 72,0% | 72,0% | 96,0% | 0,0 | 100,0% |
| Desequilíbrio fase (-75% L1) | Desbal. Corrente / S_elec | 0,000 | 0,000 | 0,058 | 0,0% | 38,0% | 38,0% | 62,0% | 0,0 | 5,0% |

## 2. Perturbação Isolada vs. Coletiva no Circuito (`Circuito F-04`)

- **Evento isolado (-50% em 1 UC do posto):** sinalizada = `1/1`, mantida como `GENUINE_UC_ANOMALY` = `1/1`, classificada como evento de rede = `0/1` (escore médio `0,336` → `0,336`).
- **Evento coletivo (-50% simultâneo em 80% do circuito = `16` UCs):** sinalizadas brutas = `16/16`, reclassificadas como `GRID_DISTURBANCE` = **`16/16`**, escore atenuado de `0,525` para `0,184`.

## 3. Limitações e Achados Desfavoráveis Registrados Honestamente

- **Quedas moderadas (-10% a -30%) no `S_elec` transversal:** Sem série temporal pré-evento, quedas moderadas alteram pouco a posição transversal da UC na população (`Δ S_elec` próximo de zero); o especialista longitudinal **CPD (CUSUM)** é indispensável para localizar essas quedas.
- **Lacuna de comunicação aleatória de 10% a 20%:** Reduz a cobertura média para ~87%–78% (acima do gate de 70%), não acionando quarentena mas elevando gradualmente `S_comm`; a partir de 30%–50% de perda, o gate intercepta a grande maioria das séries degradadas.
- **Interpretação estrita:** Métricas controladas demonstram capacidade de observação algorítmica de perturbações conhecidas, **não** desempenho de identificação de fraude em campo.
