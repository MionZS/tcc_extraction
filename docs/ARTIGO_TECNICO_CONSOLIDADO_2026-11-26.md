# Artigo Técnico Consolidado (Referência Final — 26/11/2026)

> **Nota de autoria:** Este documento consolida, em formato de artigo técnico completo, **todos os números finais reconciliados** da campanha de referência (`v1`, 19/11/2026), das auditorias P0, da avaliação multivisão P1 (Base A e perturbações controladas Base B) e da validação temporal multi-corte (`19/11 → 25/11/2026`, variante `v2` saneada). Serve como fonte técnica única e auditada para revisão autoral pessoal do manuscrito final do TCC em `docs/TCC_AMI_FINAL_BANCA_eekBR_FONTES/`.

---

## Título
**Detecção, Localização e Caracterização de Anomalias em Redes de Distribuição com Medição Inteligente (AMI): Arquitetura Multivisão, Auditoria de Sensibilidade e Validação Temporal**

## Resumo
A expansão da infraestrutura de medição avançada (AMI) permite observar consumo ativo, fluxo reverso, tensões, correntes e disponibilidade de comunicação em intervalos de 5 a 15 minutos, superando a limitação mensal do faturamento convencional. Contudo, a aplicação direta de algoritmos não supervisionados sobre telemetria real enfrenta obstáculos críticos: contaminação de escores elétricos por falhas de comunicação RF Mesh, descontinuidades de vínculo entre unidade consumidora e medidor (UC–NIO), heterogeneidade de cadências, sensibilidade à normalização em lote e ausência de rótulos de inspeção em campo. Este trabalho desenvolve e avalia experimentalmente uma plataforma analítica multivisão composta por reconstrução temporal UC–NIO, gate de qualidade de telemetria, comitê elétrico de seis detectores não supervisionados (`IForest`, `LOF`, `ECOD`, `COPOD`, `HBOS`, `PCA`) com normalização congelada (`frozen_quantile`), detector longitudinal de mudança de regime (`CUSUM/CPD`), perfilamento comportamental intradiário (24h), escore de eventos físicos (`S_tamper`) e roteamento hierárquico por circuito (`MoE` + contraste espacial de transformador). Sobre uma base real de 69.907 UCs cadastrais (262.769 janelas em quatro cortes; 65.686 UCs avaliadas fora do período de ajuste em 19/11/2026 com treino congelado até 31/10/2026), a arquitetura selecionou 6.037 candidatas (`v1`) e 4.311 sob restrição topológica. Análises de sensibilidade com modelo congelado demonstraram estabilidade superior a 97,2% no Top-2.000 frente a variações de ±20% nos pesos e correlação estritamente positiva entre todos os detectores (`IF × ECOD = +0,9626`). Em ensaio de injeção controlada de perturbações sobre 100 séries AMI reais (433.068 medições, 23 cenários), a arquitetura multivisão elevou o F1 controlado de 68,25% (Baseline IF) para 76,00%, localizou quedas sustentadas de -50% com erro mediano de 0,0 dia e reclassificou 16/16 eventos coletivos simultâneos como distúrbios de rede. Por fim, uma validação temporal multi-corte (+6 dias reais, 19/11 → 25/11/2026) com a variante saneada `v2` (regra PRODIST estrita, gate endurecido com `>= 18` dias confiáveis e 1,7 milhão de alarmes reais particionados) zerou o vazamento elétrico em blecautes consecutivos de 15 dias (31% → 0%) e atingiu 91,51% de persistência temporal da carteira. Não houve vistorias de campo (TOI); conclusões restringem-se à observabilidade computacional retrospectiva e à governança de priorização sob capacidade limitada.

---

## 1. Introdução e Escopo Científico
As perdas não técnicas (PNT) e as irregularidades operacionais em redes de baixa tensão impõem impacto técnico e econômico relevante às distribuidoras. Embora a telemetria AMI ofereça resolução sub-horária, a literatura e a prática operacional revelam que modelos monolíticos aplicados diretamente sobre séries brutas confundem falhas de comunicação com quedas de carga, sofrem deriva de escala entre lotes e não distinguem anomalias individuais de distúrbios coletivos da rede.

Este trabalho adota um **escopo estritamente retrospectivo e computacional** sobre telemetria AMI real (01/08 a 25/11/2026). **Não houve execução de vistorias de campo, retorno de OS/TOI ou confirmação pericial de fraude.** Em vez de presumir superioridade de rendimento em campo sem rótulos, o estudo responde a três perguntas verificáveis:
1. As escolhas metodológicas (gating, pesos do ensemble, normalização congelada e bônus de regras físicas) são numericamente consistentes e estáveis sob orçamento fixo (`K = 2.000/mês`)?
2. Cada especialista proposto (`S_comm`, `S_elec`, `S_behavior`, `CPD`, `S_tamper`, `MoE`/topologia) cumpre sua função observável em perturbações controladas sobre séries AMI reais?
3. As carteiras priorizadas exibem persistência temporal real em cortes futuros independentes (`19/11 → 25/11`)?

---

## 2. Arquitetura e Metodologia

### 2.1 Reconstrução Temporal UC–NIO e Separação de Subespaços
O histórico de instalação (`meter_installation_history`) é reconciliado por vigência temporal com as medições intervalares (5, 10 ou 15 minutos), calculando a cobertura sobre o denominador nativo da cadência predominante (`1440 / cadência`). As variáveis são estritamente segregadas em quatro camadas (`id__*`, `meta__*`, `x__*`, `y__*`), e o comitê elétrico recebe exclusivamente **16 features elétricas puras** (`select_pure_electrical_features`), excluindo qualquer métrica de cobertura, latência de alarme ou ciclo de vida do medidor.

### 2.2 Comitê Elétrico Congelado e Especialistas Multivisão
O ajuste ocorre estritamente nos cortes limpos pré-novembro (`31/08`, `30/09`, `31/10/2026`; `189.341` janelas). Os escores brutos dos seis detectores (`IForest=0,22`, `LOF=0,20`, `ECOD=0,18`, `COPOD=0,18`, `HBOS=0,10`, `PCA=0,12`) — definidos *a priori* por diversidade de famílias estatísticas e validados por análise de sensibilidade — são comprimidos no caso do LOF via $\log(1 + \max(-s_{\text{LOF}} - 1, 0))$, padronizados internamente no caso da PCA (`StandardScaler`) e normalizados em `[0, 1]` pelos quantis congelados `P01–P99` do treino (`frozen_quantile`), fixando o limiar no percentil 95 do treino (`τ = 0,5131`).

A arquitetura integrada adiciona cinco especialistas ortogonais:
- **`S_comm` + Gate de Qualidade:** `IForest + ECOD` sobre variáveis de comunicação; gate `cov >= 0,70` na `v1` e endurecido para `cov >= 0,70 & reliable_days >= 18` na `v2`.
- **`S_behavior`:** agrupamento comportamental (`PCA + KMeans`) em espaço elétrico e em curvas intradiárias normalizadas de 24 horas.
- **`CPD` (`CUSUM`):** localização de degraus sustentados sobre séries diárias de 30 dias com controle de escala robusta (MAD).
- **`S_tamper`:** combinação heurística de alarmes físicos de intrusão/tampa/corrente reversa (`output/events/alarm_events/`), inclinação negativa de consumo e troca recente de medidor.
- **`MoE` Hierárquico e Filtro Espacial (`SpuriousFilter`):** 16 especialistas por alimentador treinados sem vazamento (`<= 31/10/2026`) ponderados com o modelo global conforme tamanho amostral e penetração solar, acoplados ao contraste de transformador (`grid_event_threshold = 60%`).

---

## 3. Resultados Experimentais

### 3.1 Reconciliação de Populações e Auditoria de Correlações (P0)
A Tabela 1 reconcilia integralmente todas as contagens da campanha de referência (`19/11/2026`, execução `20261125_134559_972_oot_multiview`). A soma das nove categorias diagnósticas fecha exatamente em **65.686 UCs**. A diferença entre o somatório das três categorias elétricas/GD/metrológicas (**6.053 UCs**) e a fila de candidatas (**6.037 UCs**) corresponde exatamente a **16 UCs** em `POSSIVEL_GD_NAO_CADASTRADA` que atenderam à regra física de reversão (`ra_ratio >= 0,08` e `>= 2` dias) com telemetria confiável, mas obtiveram `s_elec` entre `0,2799` e `0,5091` (`< 0,5131`). Na comparação com `DAILY_RULES` em 19/11 (`4.978` candidatas), há **2.069** UCs em interseção, **3.968** exclusivas do OOT e **2.909** exclusivas das regras diárias.

A auditoria direta sobre os escores congelados confirmou alinhamento exato (`max_abs_diff = 0,000000`), matrizes simétricas e **ausência total de correlações negativas**: `IF × ECOD = +0,9626` (Spearman nos 62.857 elegíveis; `+0,9639` no total; Pearson `+0,9445`). `LOF` (`ρ = +0,4609` nos elegíveis; `+0,4763` no total) e `PCA` (`ρ = +0,4135` nos elegíveis; `+0,4396` no total) demonstram diversidade genuína frente ao bloco `IF/ECOD/COPOD/HBOS` (`docs/tcc_evidence/p0_audit/figures/fig_p0_02_correlacao_detectores_revisada.pdf`).

### 3.2 Sensibilidade Congelada de Gating, Pesos e Bônus `+0,15` (P0)
- **Gating congelado (sem *refit* no OOT):** mantendo o modelo e `τ = 0,5131` fixos, o total bruto de sinalizações permanece constante em **6.916 UCs** nos cinco limiares (`50%` a `90%`). Em `80%`, há **61.530** elegíveis (93,67%) e **4.156** em quarentena (sem inversão), com `5.821` retidas e `1.095` interceptadas (`Jaccard Top-2.000 = 0,9550` vs. 70%).
- **Sensibilidade dos pesos (23 configurações):** sob orçamento fixo `K = 2.000` (Protocolo A), perturbações individuais de **±20%** em qualquer peso produzem **Jaccard Top-2.000 entre 0,9724 e 0,9940** (apenas 6 a 28 substituições; Spearman `>= 0,9994`). Pesos uniformes (`1/6`) preservam `93,61%` do Top-2.000 (66 trocas). Em contrapartida, remover o `LOF` provoca **180 trocas** (`Jaccard = 0,8349`) e usar `Isolation Forest` isolado provoca **395 trocas** (`Jaccard = 0,6701`), evidenciando a contribuição real do comitê. Sob limiar fixo (Protocolo B), a remoção de detectores desloca a escala e exige recalibração no treino.
- **Achado de auditoria no bônus `+0,15`:** a auditoria revelou que o padrão `"IMBALANCE_MAX"` em `composite.py` incluiu inadvertidamente `x__hist_current_imbalance_max > 0,20`, acionando o bônus em **29.398 UCs elegíveis (46,77%)**, frente a apenas **89 elegíveis (0,14%)** sob violações estritas de tensão (`> 20%`) ou reversão (`> 85%`). Sem o bônus (`0,0`), o volume em `τ = 0,5131` cai de `6.039` para `3.438` (125 substituições no Top-2.000).

### 3.3 Avaliação Integrada Multivisão e Perturbações Controladas (P1)
Na Base A (`65.686` UCs OOT), o `MoE` sem vazamento (16 alimentadores) substituiu **282 vagas** no Top-2.000 (`Jaccard = 0,7528`, deslocamento médio de `5,14 p.p.`), enquanto o `CPD` avaliou `63.719` UCs em `8,96 s`, detectando `2.758` quedas `>= 30%`. O gate de qualidade zerou o vazamento de UCs em quarentena no Top-2.000 (de `138` no Baseline `V0` e `257` no `V1` sem gate para **`0`** em `V2–V5`).

Na Base B (**100 séries AMI reais**, 433.068 medições de 5/10 min, 23 cenários de injeção controlada em cópias isoladas):
- **Lacunas aleatórias (50%):** 100% interceptadas pelo gate (`0%` de vazamento elétrico vs. `19%` sem gate).
- **Quedas sustentadas (-10% a -50%):** o ensemble transversal `S_elec` isolado mostrou baixa sensibilidade a quedas moderadas (-10% a -30%), enquanto o **CPD** localizou quedas de -30%, -40% (7–14d) e -50% com **erro mediano de 0,0 dia** (82% dentro de ±3 dias em -50%) e taxa residual de apenas 1–2% em eventos isolados de 1 dia.
- **Deslocamento de rotina (+4h com `ΔkWh/dia = 0,0`):** detectado por `S_behavior` 24h em **52,0%** das UCs (`> P95` ref.), enquanto `S_elec` diário agregado apresentou variação nula (`0,0000`).
- **Injeção solar reversa (GD) e carga zero com tensão normal:** **100,0%** das injeções GD classificadas corretamente; **96,0%** das cargas zeradas localizadas pelo CPD com erro `0,0 dia`.
- **Evento coletivo no circuito (80% simultâneo em -50%):** **16/16** UCs reclassificadas como `GRID_DISTURBANCE` (escore atenuado de `0,5253` para `0,1838`), preservando `1/1` quando a queda foi isolada em uma única UC.
- **Ablação `V0–V5` sob orçamento constante (`K = 400` na Base B):** o F1 controlado subiu de **68,25%** (`V0`) e **69,25%** (`V2`) para **76,00%** (`V4/V5`).

### 3.4 Tabela Central Comparativa (Baseline × Ensemble × Multivisão)

| Indicador | Baseline (`V0`) | Ensemble (`V2`) | Multivisão (`V5`) |
|---|---:|---:|---:|
| UCs avaliadas / elegíveis (OOT 19/11) | 65.686 / 65.686 | 65.686 / 62.857 | 65.686 / 62.857 |
| Quarentena vazada no Top-2.000 (Base A) | 138 | **0** | **0** |
| Perturbações elétricas — F1 controlado (Base B, K=400) | 68,25% | 69,25% | **76,00%** |
| Vazamento de lacunas de comunicação (Base B, K=400) | 31,50% | 22,00% | **18,00%** |
| Queda -50%: localização CPD `<= 3d` (erro mediano) | N/A | N/A | **82,0% (0,0 dia)** |
| Deslocamento rotina +4h (`ΔkWh = 0`): `S_behavior 24h` | 32,0% | 32,0% | **52,0%** |
| Eventos coletivos (80% circuito) filtrados como rede | N/A | N/A | **16 / 16 (100%)** |
| Tempo computacional OOT (65.686 UCs) | 0,15 s | 11,04 s | 36,12 s |

### 3.5 Validação Temporal Multi-Corte (`19/11 → 25/11`) e Variante `v2` Saneada
Com a incorporação de 6 dias adicionais reais (`20/11` a `25/11/2026`) e dos 41 dias particionados de `alarm_events` (1,7 milhão de eventos na janela de 30d), avaliou-se a variante saneada **`v2`** (regra PRODIST estrita de tensão/reversão + gate `cov >= 70% & reliable_days >= 18` + `S_tamper` sobre alarmes reais), mantendo o treino congelado em `<= 31/10/2026`:

| Execução | Elegíveis | Flag Regra (+0,15) | `S_tamper >= 0,70` | Candidatos | Persistência 6d (`19/11 → 25/11`) |
|---|---:|---:|---:|---:|---:|
| `v1` (19/11 — Ref.) | 62.857 | 30.694 (46,73%) | 0 | 6.037 | 88,75% cands / 89,85% Top-2k (`ρ = 0,9688`) |
| `v1` (25/11 — +6d) | 62.274 | 30.649 (46,69%) | 0 | 6.059 | — |
| `v2` (19/11 — Saneada) | 62.686 | **124 (0,19%)** | **1.377** | **4.298** | **91,51% cands** / 88,00% Top-2k (`ρ = 0,9484`) |
| `v2` (25/11 — Saneada +6d) | 62.185 | **122 (0,19%)** | **1.375** | **4.300** | — |

Na verificação controlada sobre 100 UCs reais, o gate endurecido da `v2` reduziu o vazamento elétrico no bloco consecutivo de 15 dias ausentes de **31,0% (`v1`) para 0,0% (`v2`, 100% em quarentena)**. Entre `v1` e `v2` no corte de 19/11, a sobreposição no Top-2.000 atingiu **92,4%** (`1.848/2.000`), demonstrando que o saneamento removeu 1.739 casos limítrofes da cauda sem descaracterizar o núcleo prioritário. Os três estudos de caso (`908***00`, `291***96`, `685***20`) mantiveram coerência integral até `25/11/2026` (incluindo confirmação de degrau `-73,0%` em `31/10` na `908***00` e 2 alarmes reais de tamper na `291***96`).

---

## 4. Limitações, Governança Operacional e Conclusões
Os resultados demonstram que a plataforma produz carteiras estáveis, rastreáveis e auditáveis sob capacidade operacional restrita (`K = 2.000/mês`), separando filas de Telecom (`2.739` UCs), Ciclo de Vida (`136`), Metrologia (`515`), Possível GD (`1.451`) e Análise Técnica de PNT (`4.087`). As limitações centrais permanecem explícitas: ausência de confirmação de campo (TOI), natureza heurística dos pesos iniciais (embora robusta sob orçamento fixo) e dependência de um piloto prospectivo aleatorizado por blocos de circuito para mensuração de *yield*, *lift* e retorno econômico efetivo.
