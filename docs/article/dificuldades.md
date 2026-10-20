# Diário de Bordo: Dificuldades, Desafios Técnicos e Lições Aprendidas no TCC

Este documento registra os principais obstáculos de engenharia de software, modelagem de dados, infraestrutura e governança enfrentados ao longo da construção do pipeline de extração e classificação de perturbações de tensão da Copel (Araucária).

---

## 1. O Problema da Relação UC × NIO ao Longo do Tempo (Séries Temporais em Redes AMI)

* **O Desafio**: 
  O sistema de medição inteligente (MDM/ORCA) armazena telemetria referenciada estritamente pelo número de identificação física do medidor (`NIO`). No entanto, o problema elétrico, regulatório (PRODIST) e comercial reside na **Unidade Consumidora (`UC`)**.
  Ao longo do ciclo de vida de uma instalação, ocorrem substituições de medidores (manutenção, defeito, transição de medidores convencionais para inteligentes).
* **A Falha Inicial**:
  O pipeline extraía um snapshot estático do cadastro (`CIS`) no dia da execução e usava apenas os medidores ativos no presente para buscar o histórico de 70 a 90 dias atrás. Como consequência, UCs que tiveram troca de medidor apresentavam apagões artificiais de telemetria nos dias anteriores à troca.
* **A Solução e Complexidade**:
  Foi necessário migrar de uma modelagem baseada em chaves estáticas para um modelo de **vigência temporal inclusiva**:
  $$\text{Medidor Elegível em } D \iff (\text{DATA\_INSTALACAO} \le D) \land (\text{DATA\_REMOCAO is NULL} \lor \text{DATA\_REMOCAO} \ge D)$$
  No dia da substituição ($D$), ambos os medidores (antigo e novo) precisam ser consultados e consolidados, evitando saltos espúrios e falsas anomalias.

---

## 2. Inconsistências de Esquema e Divergências de Interfaces entre Camadas

* **O Desafio**:
  O projeto evoluiu de uma modelagem plana baseada em tabelas soltas (`meter_day_metadata`) para uma arquitetura semântica multicamadas (`uc_context`, `meter_installation_history`, `electrical_hierarchy`, `ami_interval`, `ami_instantaneous`, `ami_registers`).
* **Erros Encontrados**:
  1. **Divergência de Schemas nos Testes**: O arquivo `test_dataset_grains.py` tentava importar `METER_DAY_METADATA_SCHEMA` (tabela depreciada), impedindo a coleta da suíte de testes (`ImportError`).
  2. **Quebra de Contrato em Funções de Configuração**: A função interna `_read_db_config` em `src/db.py` não expunha o alias público `load_database_config`, quebrando o teste de auditoria de vulnerabilidades `test_vulnerabilities_remediation.py`.
  3. **Campos Faltantes no Cruzamento CIS × GEO**: A query do CIS não retornava a coluna `NUM_POSTO_OPERACIONAL`. A tentativa ingênua de usar `pl.col(cis_posto_col)` causava `ColumnNotFoundError` dentro do Polars antes mesmo da avaliação condicional.

---

## 3. Gargalos de Desempenho e o Custo do "Unpivot" de Telemetria

* **O Desafio**:
  A base de Araucária conta com mais de 72.000 medidores. O MDM devolve curvas diárias com 288 intervalos de 5 minutos compactados em JSONs textuais nas colunas do Oracle.
* **A Barreira Encontrada**:
  A função `normalize_mdm_dataframe` realizava `iter_rows(named=True)` linha por linha em Python, decodificando dezenas de campos JSON com `json.loads` e instanciando pequenos DataFrames Polars por medidor. O uso de `ThreadPoolExecutor` não acelerava o processamento por esbarrar no GIL (Global Interpreter Lock) do CPython.
* **Solução Adotada**:
  Processamento em lotes, reutilização persistente de conexões com o Oracle MDM (`SYS.ODCIVARCHAR2LIST`) e transição para rotinas colunares Lazy (`sink_parquet` / `sink_csv`).

---

## 4. Retenção Rígida do Sistema Fonte (MDM ORCA)

* **O Desafio**:
  O banco de dados MDM da concessionária adota expurgo mensal programado (retenção de aproximadamente 4 a 5 meses).
* **Risco Identificado**:
  Se uma sincronização histórica retroagir a meses já expurgados, a consulta gera lotes com 0 registros ou falhas silenciosas. O Data Lake precisa atuar como cofre histórico perene com particionamento local idempotente (`report_year=*/report_month=*/report_day=*`), salvando em formatos abertos (Parquet e CSV) para permitir auditoria posterior.

---

## 5. Over-Engineering vs. Pragmatismo Científico

* **O Desafio**:
  Ao longo da pesquisa, implementou-se uma arquitetura sofisticada de aprendizado de máquina (Mixture of Experts - MoE, classificadores hierárquicos, composite models).
* **Lição Aprendida**:
  O setor elétrico brasileiro é regulado por normas estritas e determinísticas (PRODIST Módulo 8 da ANEEL). Um modelo de inteligência artificial não pode atuar como uma caixa preta desvinculada dos limites físicos de regime permanente (0.90 p.u. a 1.05 p.u.).
  O papel do modelo tornou-se auxiliar e preditivo: explicar tendências de desbalanceamento contínuo, persistência de perturbações e correlações espaciais no transformador, mantendo o limiar regulatório como referência basilar de integridade.
