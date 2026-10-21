# Diário de Bordo (Continuação): Desafios de Escala, Mudança de Regime e Armadilhas Metodológicas

> **Documento de Continuação:** Complemento analítico a [`docs/article/dificuldades.md`](file:///d:/Projects/tcc_extraction/docs/article/dificuldades.md)  
> **Escopo:** Lições de engenharia de dados em larga escala, armadilhas de inferência estatística e maturidade arquitetural  
> **Status:** Documentado e Integrado ao Histórico Técnico do TCC

---

## 6. O Dilema da Janela Estática de 30 Dias e a Sazonalidade no Change Point Detection

* **O Desafio Conceitual**:  
  A abordagem inicial da modelagem calculava métricas descritivas (média, mediana, MAD, desvio padrão) sobre uma janela móvel fixa de 30 dias ($W=30$).
* **A Armadilha Identificada**:  
  1. **Mascaramento de Eventos Recentes:** Se uma intervenção fraudulenta ou avaria física iniciou-se no 22º dia da janela, uma média de 30 dias é amortecida pelos 21 dias normais anteriores, diluindo o sinal no ruído.
  2. **Normalização de Fraudes Crônicas:** Se a fraude já operava antes do início da janela de observação, a janela estática assume o consumo rebaixado como sendo o comportamento "normal" do consumidor, tornando o modelo cego a desvios consolidados.
  3. **Falsos Alarmes Sazonais no CUSUM:** A primeira tentativa ingênua de introduzir o algoritmo CUSUM (*Cumulative Sum*) sobre os dados brutos de energia ativa ($FA$ em kWh) disparava falsos pontos de quebra em dias de alta temperatura ou nas transições entre dias úteis e fins de semana.
* **A Solução Técnica Adotada**:  
  - O CUSUM foi reformulado para operar sobre a **série de resíduos desazonalizados**:
    $$e_t = FA_t - \bar{FA}_{\text{circadiano}}(h_t, d_t)$$
    onde o baseline é condicionado à hora do dia e ao perfil do dia da semana (útil vs. fim de semana).
  - Distinguiu-se formalmente o modo de **Auditoria Forense Offline** (que compara janelas simétricas antes/depois do ponto candidato $t^*$) do modo de **Monitoramento Online em Produção** (que compara estritamente o histórico consolidado contra o bloco recente de teste, prevenindo vazamento de dados futuros).

---

## 7. Gargalo de Escala do Oracle MDM: Efeito Comboio e Esgotamento da PGA

* **O Desafio de Engenharia de Dados**:  
  A extração diária de telemetria para os mais de 72.000 medidores inteligentes de Araucária exige consultar tabelas colossais no Oracle (milhões de intervalos de 15 minutos e fasores por dia de referência).
* **A Falha Observada em Teste de Carga**:  
  No benchmark de lotes heterogêneos executado em outubro de 2026, testaram-se simultaneamente 5 workers com tamanhos de lote variando de 8.000 a 30.000 medidores:
  - Workers com lotes de 8.000 e 12.000 medidores completaram suas consultas em **7 a 9 minutos** (~48 a 52 ms/medidor);
  - O worker atribuído ao lote de 28.436 medidores (limite superior do teste) degradou severamente, exigindo **44.8 minutos** (94.5 ms/medidor) — sendo **23 minutos apenas na etapa de `fetch` de rede**!
  - **O Efeito Comboio (Worker Starvation):** 4 dos 5 workers ficaram completamente ociosos por mais de 35 minutos enquanto o processo aguardava a finalização do lote gigante.
* **A Causa Raiz no Banco de Dados**:  
  1. **Esgotamento da PGA (*Program Global Area*):** Ao processar uma junção que excede 4,5 milhões de registros pontuais com strings JSON longas em uma única sessão, a memória de trabalho do Oracle estoura a cota alocada e faz *spill* para tabelas temporárias em disco (`ORA_TEMP`), derrubando o throughput de execução.
  2. **Estrangulamento do Buffer de Rede `SQL*Net`:** O driver DBAPI/cx_Oracle/oracledb utilizava o parâmetro padrão de `arraysize = 100`, gerando milhares de round-trips de rede para trafegar dezenas de milhares de registros volumosos.
* **A Solução Aplicada**:  
  - Calibração do cursor para `cursor.arraysize = 1000` e `yield_per = 1000`, reduzindo os round-trips de rede em até 10 vezes;
  - Limite superior de particionamento estabelecido em **12.000 a 14.000 medidores por lote**, mantendo a execução inteiramente na memória RAM da PGA do Oracle;
  - Balanceamento rigoroso dos 5 workers em fatias iguais (~14.487 medidores cada), garantindo que todos os workers iniciem e concluam simultaneamente em aproximadamente **10 a 12 minutos por dia completo**, eliminando o efeito comboio.

---

## 8. Ambiguidade de Avaliação Booleana em DataFrames Polars

* **O Desafio em Tempo de Execução**:  
  Durante a etapa final de consolidação das features diárias do pipeline completo (`run_araucaria_full.py`), a execução foi interrompida pelo seguinte erro:
  ```text
  ValueError: the truth value of a DataFrame is ambiguous
  Hint: to check if a DataFrame contains any values, use `is_empty()`.
  ```
* **A Causa Raiz**:  
  No arquivo [`src/features/meter_day.py`](file:///d:/Projects/tcc_extraction/src/features/meter_day.py), havia uma condicional de fallback escrita como:
  ```python
  if not meter_to_uc_map and metadata_df is not None and not metadata_df.is_empty():
  ```
  Quando o dicionário `meter_to_uc_map` era fornecido vazio (`{}`), a semântica de encadeamento lógico `not A and B` provocou uma avaliação contextual indesejada pelo interpretador Python, forçando a coerção booleana implícita do objeto `metadata_df` (que por ser um DataFrame Polars não possui método `__bool__` implementado, de modo a impedir confusões entre tabelas vazias e avaliações elemento a elemento).
* **A Correção de Engenharia**:  
  Isolamento estrito da checagem de dicionário antes de qualquer avaliação sobre DataFrames:
  ```python
  if (meter_to_uc_map is None or len(meter_to_uc_map) == 0) and metadata_df is not None and not metadata_df.is_empty():
  ```

---

## 9. A Tentação da Caixa-Preta e a Prudência com Inferências Causais

* **O Desafio Metodológico**:  
  Sistemas de detecção de anomalias frequentemente tentam emitir laudos automatizados do tipo:
  - Queda de consumo diurno + energia reversa $\to$ *"Fraude por Geração Solar Clandestina"*;
  - Consumo próximo de zero $\to$ *"UC Desocupada"*;
  - Corrente de fase desequilibrada $\to$ *"By-pass Monofásico Confirmado"*.
* **A Lição Científica Aprendida**:  
  Em redes elétricas reais e sob a cadência de 15 minutos, **a telemetria isolada é insuficiente para provar a intenção ou a causa exata**. 
  - Uma queda a zero pode ser um disjuntor geral desligado pelo consumidor durante uma viagem;
  - Um fluxo reverso pode decorrer de um sistema de microgeração que já está em processo de homologação técnica com parecer de acesso emitido, mas cujo cadastro comercial ainda não foi finalizado no ERP;
  - Um desequilíbrio acentuado de corrente pode derivar de uma carga monofásica interna de alta potência (ex.: máquina de solda, carregador veicular não homologado) sem qualquer manipulação do medidor.
* **A Resolução Arquitetural**:  
  O pipeline foi reconfigurado para emitir **padrões de compatibilidade operacional graduados** (ex.: *padrão compatível com geração*; *padrão compatível com redução abrupta de carga*). A qualificação como suspeita de perda não técnica só ocorre após o cruzamento com o histórico de ordens de serviço (`meter_installation_history`), alarmes de hardware (`alarm_categories`) e consenso de vizinhança elétrica no transformador (`spurious_filter`).
