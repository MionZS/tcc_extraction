# Mapeamento de Vulnerabilidades Sutis e Falhas de Design Arquitetural

> **Documento:** `docs/vulnerabilities.md`  
> **Classificação:** Relatório de Auditoria Técnica e Arquitetural  
> **Escopo:** Pipeline de Extração, Engenharia de Features, Modelagem de ML e Governança de Dados (Projeto TCC Araucária)  
> **Data:** Setembro de 2026  

---

## Sumário Executivo

Este documento reúne uma análise aprofundada, crítica e detalhada das vulnerabilidades estruturais, metodológicas, de segurança e de integridade matemática identificadas no ecossistema da base de dados e nos scripts de engenharia analítica. 

Ao contrário de bugs sintáticos (que interrompem a execução com traceback), as **vulnerabilidades sutis** operam silenciosamente: o pipeline conclui sem erros, mas os dados gerados contêm distorções físicas, vazamentos temporais (*data leakage*), descompassos cronológicos ou gargalos exponenciais de escalabilidade.

---

## 1. Integridade Temporal e Séries Temporais (Falhas Críticas Silenciosas)

### 1.1 O Descompasso Temporal por `ABSENT ON NULL` (Time-Shift Assíncrono)
* **Localização:** `queries/mdm_coluna.sql` (linhas 213–253) e `src/datasets/normalize.py` (linhas 95–104 e 284–308).
* **Mecanismo da Falha:**
  1. No SQL, os 288 intervalos diários (a cada 5 minutos) são agrupados via `JSON_OBJECTAGG(KEY hhmi VALUE ... ABSENT ON NULL)`. Quando uma medição é nula, a chave correspondente ao horário (ex: `"08:05"`) é sumariamente omitida do JSON gerado no Oracle.
  2. No módulo Python `normalize.py`, a função `_parse_json_slot` extrai os valores através de `sorted_items = sorted(obj.items(), key=lambda kv: kv[0])` e retorna apenas a lista unidimensional dos valores (`[_coerce_float(v) for _, v in sorted_items]`), **descartando a chave cronológica `hhmi`**.
  3. No loop de reconstrução dos slots (`normalize_mdm_day`), o código itera em `range(slot_count)` associando o valor na posição `i` ao timestamp `_slot_index_to_time(i, report_day)`.
* **Impacto Real:**  
  Se um medidor deixar de transmitir às 08:00 e 08:05, a medição real das 08:10 assume o índice 96 (08:00), a das 08:15 assume o índice 97 (08:05), e assim por diante. **Toda a série temporal subsequente sofre um deslocamento temporal (lag artificial) para trás**.
  
### 1.2 Dessincronização Física Inter-Grandezas
* **Localização:** `src/datasets/normalize.py` (`normalize_mdm_day`).
* **Mecanismo da Falha:**
  Como cada coluna física (`FA_INTERVAL`, `U_L1_AVG`, `I_L1_AVG`, etc.) é serializada em seu próprio JSON independente com `ABSENT ON NULL`, a omissão de slots ocorre de forma desemparelhada. Se `U_L1_AVG` faltar às 10:00 e `FA_INTERVAL` faltar às 14:00, os vetores resultantes terão tamanhos e alinhamentos horários divergentes.
* **Impacto Real:**  
  O pipeline correlacionará a corrente de um determinado horário com a tensão de um horário totalmente diferente na mesma linha. Métricas derivadas de física elétrica (como fator de potência, impedância aparente e correlação carga-tensão) tornam-se completamente fictícias.

### 1.3 Distorção Artificial de Cadência por Telemetria Degradada
* **Localização:** `src/datasets/normalize.py` (`_infer_cadence_from_slot_count`, linhas 192–206).
* **Mecanismo da Falha:**
  O algoritmo deduz a cadência através de `raw_cadence = 1440 / slot_count` e ajusta para a cadência padrão mais próxima `[5, 10, 15, 30, 60]`.
* **Impacto Real:**  
  Se um medidor de 5 minutos (esperado: 288 slots/dia) sofrer instabilidade na rede celular e entregar apenas 96 slots em um dia de contingência, o sistema classifica esse medidor como sendo de 15 minutos (`1440 / 96 = 15`). O script redistribui os 96 pontos espaçados uniformemente ao longo das 24 horas, esticando artificialmente a curva de carga diária e ocultando a janela de apagão.

---

## 2. Modelagem de Machine Learning e Vazamento Temporal (*Data Leakage*)

### 2.1 Vazamento de Futuro no Histórico de Instalação de Medidores
* **Localização:** `src/features/uc_window.py` (linhas 119–128).
* **Mecanismo da Falha:**
  Para calcular `METER_AGE_DAYS`, o código extrai `install_dates = uc_meters["DATA_INSTALACAO"]` e toma `latest_install = max(install_dates)`. Não há restrição `WHERE DATA_INSTALACAO <= cutoff_date`.
* **Impacto Real:**  
  Ao construir matrizes históricas de treinamento para retroanálise (ex: gerando amostras com data de corte em 01/07/2026), se a UC teve uma troca de medidor em 15/09/2026, a data futura de setembro é selecionada. A diferença `(cutoff_date - latest_install).days` resulta negativa, sendo convertida em `0` pelo `max(0, ...)`. Consequentemente, a feature booleana `METER_CHANGED_30D` é marcada como `True` retroativamente no passado. O modelo preditivo aprende correlações com eventos que ainda não haviam ocorrido na data de corte.

### 2.2 Cegueira em Janelas de Transição de Medidores
* **Localização:** `src/features/uc_window.py` (linhas 135–142).
* **Mecanismo da Falha:**
  A agregação de alarmes na janela móvel de 30 dias obtém o identificador do medidor via `nio_val = uc_daily["NIO"][0]`.
* **Impacto Real:**  
  Se um cliente teve o medidor substituído no meio da janela de 30 dias, a tabela diária contém registros com o NIO antigo nos primeiros 15 dias e o novo nos últimos 15 dias. Ao fixar estritamente o primeiro NIO (`[0]`), todos os alarmes gerados pelo segundo medidor na mesma janela são ignorados no cômputo da UC.

### 2.3 Comparação de Tipos Heterogêneos em Filtros Polars
* **Localização:** `src/features/uc_window.py` (linhas 137–141).
* **Mecanismo da Falha:**
  A coluna `ORIGIN_TIMESTAMP` da tabela de alarmes é tipada como `Datetime`, enquanto as variáveis de contorno `window_start` e `cutoff_date` são objetos nativos `datetime.date`.
* **Impacto Real:**  
  Dependendo da versão do Polars e do backend de execução, essa comparação pode disparar `ComputeError` ou forçar um casting implícito com overhead e risco de truncamento incorreto de limites horários (excluir ou incluir indevidamente eventos ocorridos durante as horas do último dia).

---

## 3. Conformidade com Engenharia Elétrica e Qualidade de Energia

### 3.1 Mascaramento do Desbalanceamento Dinâmico de Fases
* **Localização:** `src/features/electrical.py` (`compute_voltage_imbalance`, linhas 44–53).
* **Mecanismo da Falha:**
  O cálculo faz a média das tensões de cada fase ao longo do dia inteiro e, em seguida, compara a disparidade entre essas três médias estáticas:
  $$\text{Imbalance} = \frac{|\bar{V}_A - \bar{V}_B|}{\frac{\bar{V}_A + \bar{V}_B}{2}}$$
* **Impacto Real:**  
  Normas técnicas (PRODIST Módulo 8 / IEEE Std 1159 / IEC 61000-4-30) definem desbalanceamento instantâneo para cada intervalo amostral. Se a Fase A sofre sobrecarga no período da manhã e a Fase B sofre sobrecarga idêntica no período da noite, as médias diárias $\bar{V}_A$ e $\bar{V}_B$ serão iguais, resultando em desbalanceamento calculado igual a `0.0`. O algoritmo é cego para desbalanceamentos dinâmicos severos que danificam transformadores e motores na rede.

### 3.2 Omissão Sistemática de Clientes Monofásicos e Bifásicos
* **Localização:** `src/features/electrical.py` (linha 39: `source.select(u_cols).drop_nulls()`).
* **Mecanismo da Falha:**
  A lista `u_cols` inspeciona `["U_L1", "U_L2", "U_L3"]`. Em instalações monofásicas (comuns em áreas residenciais de baixa tensão), as colunas `U_L2` e `U_L3` contêm exclusivamente valores `NULL`.
* **Impacto Real:**  
  A chamada `drop_nulls()` descarta 100% das linhas do DataFrame para clientes monofásicos e bifásicos. O bloco `if temp.is_empty(): return 0.0` é acionado silenciosamente. Em vez de registrar um valor especial (ou `NaN`) indicando incompatibilidade de topologia, o sistema atribui `0.0` perfeito, distorcendo distribuições estatísticas e confundindo os algoritmos de clustering.

---

## 4. Eficiência Computacional e Arquitetura de Software

### 4.1 Anti-Pattern de Iteração $O(N \times M)$ no Polars
* **Localização:** `src/features/meter_day.py` (linhas 57–63) e `src/features/uc_window.py` (linhas 55–60).
* **Mecanismo da Falha:**
  ```python
  ucs = window_daily["UC"].unique().sort().to_list()
  for uc in ucs:
      uc_daily = window_daily.filter(pl.col("UC") == uc)
  ```
* **Impacto Real:**  
  O Polars é um motor colunar em Rust projetado para agregações vetorizadas paralelas. Executar um loop `for` em Python que realiza `filter()` sequencial para cada UC sobre uma tabela com centenas de milhares de linhas cria milhares de DataFrames temporários na memória. Para a totalidade de Araucária (dezenas de milhares de UCs), o tempo de execução degrada exponencialmente, transformando uma operação de segundos em horas. A abordagem idiomática requer `window_daily.group_by("UC").agg(...)`.

### 4.2 Overhead de Serialização Redundante (Oracle CLOB $\rightarrow$ Python JSON)
* **Localização:** `queries/mdm_coluna.sql` $\leftrightarrow$ `src/datasets/normalize.py`.
* **Mecanismo da Falha:**
  O banco de dados gasta tempo de CPU transformando dados relacionais em strings JSON dentro de tipos CLOB (`JSON_OBJECTAGG`). Em seguida, o Python aloca memória para ler o CLOB e executa `json.loads()` em 25 colunas por medidor por dia. Logo após o parse, o código quebra o JSON novamente em linhas tabulares no Polars.
* **Impacto Real:**  
  Trata-se de um ciclo de serialização e desserialização inútil. Além de consumir memória massiva em ambos os lados, degrada a largura de banda da rede entre o banco Oracle e a máquina de processamento.

---

## 5. Modelagem Relacional e Integridade Cadastral

### 5.1 Risco de Explosão Cartesiana (Fan-out de Joins)
* **Localização:** `scripts/run_feeder_pipeline.py` (linhas 209–212) e `queries/geo_feeder_direct.sql`.
* **Mecanismo da Falha:**
  O cruzamento entre a base de clientes (CIS) e a base de topologia física (GEO) é feito por `cis_df.join(geo_df, on="UC", how="inner")`.
* **Impacto Real:**  
  Na estrutura cadastral das distribuidoras, a chave `UC` não possui garantia universal de unicidade em fotos descontextualizadas de vigência (uma UC pode possuir contratos históricos, ramais inativos ou conexões múltiplas no GEO). Se uma UC possuir 2 registros no CIS e 2 registros no GEO, o join sem desduplicação prévia gera 4 linhas resultantes. Isso duplica artificialmente o peso dessa UC nas matrizes de treino e na lista de medidores a extrair.

### 5.2 Ambiguidade de Catálogo no MDM (`a_data_catalogue`)
* **Localização:** `queries/mdm_coluna.sql` (linhas 23–30).
* **Mecanismo da Falha:**
  A tabela `catalogue` faz um join direto por `meter_asset_no`.
* **Impacto Real:**  
  Se um mesmo número de medidor (`meter_asset_no`) possui múltiplos `data_id` na tabela `AMI.a_data_catalogue` (devido a reconfigurações lógicas de equipamento ou substituições de modem), o cross join com a grade de tempo (`time_grid`) multiplica as linhas. O `MAX()` posterior mescla registros de naturezas contratuais ou operacionais distintas.

---

## 6. Governança de Dependências e Segurança de Credenciais

### 6.1 Dependências Fantasma (*Ghost Dependencies*)
* **Localização:** `pyproject.toml` (linhas 7–13) versus `scripts/train_anomaly_model.py`.
* **Mecanismo da Falha:**
  O script principal de treinamento de Machine Learning importa:
  ```python
  from sklearn.cluster import KMeans
  from sklearn.ensemble import IsolationForest
  import joblib
  ```
  No entanto, o arquivo de configuração do ambiente (`pyproject.toml`) declara exclusivamente:
  ```toml
  dependencies = [
      "numpy>=2.4.6",
      "oracledb>=3.4.2",
      "polars>=1.38.1",
      "pytest>=9.1.0",
      "sqlalchemy>=2.0.48",
  ]
  ```
* **Impacto Real:**  
  O pipeline falha fatalmente com `ModuleNotFoundError: No module named 'sklearn'` em qualquer instalação limpa (`uv sync` ou novo ambiente virtual), impedindo a automação contínua e a reprodutibilidade dos experimentos por terceiros.

### 6.2 Exposição Crítica de Credenciais em Texto Claro
* **Localização:** `config.json`.
* **Mecanismo da Falha:**
  O arquivo central de conexões contém usuários e senhas explícitas para cinco bancos corporativos e servidores de produção (`HEXPRD19`, `cisdprd`, `dbgeoprd`, `medprd`, `SANPLAT`).
* **Impacto Real:**  
  Embora conste no `.gitignore` local, a dependência direta de um arquivo local estático aumenta o risco de vazamento acidental em backups, compartilhamentos de pasta de rede ou compressões `.zip`. Além disso, o código em `src/db.py` não prevê fallback para variáveis de ambiente seguras (`os.environ`).

---

## 7. Matriz de Priorização para Remediação

| Prioridade | Vulnerabilidade | Categoria | Risco Principal |
|:---|:---|:---|:---|
| 🚨 **P0 - Bloqueante** | Deslocamento temporal em slots nulos (`ABSENT ON NULL`) | Integridade Temporal | Treinamento sobre séries temporais artificialmente deformadas. |
| 🚨 **P0 - Bloqueante** | Dessincronização entre colunas físicas ($U$ vs $I$ vs $P$) | Física Elétrica | Cálculo de correlações entre grandezas de horários distintos. |
| 🚨 **P0 - Bloqueante** | Dependências ausentes (`scikit-learn`, `joblib`) no `pyproject.toml` | Reprodutibilidade | Falha de execução imediata em ambientes novos. |
| ⚠️ **P1 - Alta** | Vazamento temporal (*Data Leakage*) em `METER_AGE_DAYS` | Modelagem ML | Modelo otimista que aprende com trocas de medidor futuras. |
| ⚠️ **P1 - Alta** | Eliminação de monofásicos/bifásicos em `compute_voltage_imbalance` | Integridade Amostral | Zeros artificiais atribuídos à grande massa de clientes residenciais. |
| ⚠️ **P1 - Alta** | Loops $O(N)$ em Python puro com `DataFrame.filter()` no Polars | Escalabilidade | Pipeline trava ou demora horas ao processar a cidade toda. |
| 🟡 **P2 - Média** | Desbalanceamento médio vs. desbalanceamento dinâmico instantâneo | Domínio Técnico | Mascaramento de distúrbios elétricos severos em horários alternados. |
| 🟡 **P2 - Média** | Credenciais em texto claro e ausência de suporte a `.env` | Segurança | Exposição acidental de acessos corporativos de produção. |

