# Decisão de Arquitetura de Dataset para o TCC

**Projeto:** análise de dados de medidores inteligentes em Araucária  
**Status:** proposta técnica v1.0  
**Data:** 17 de julho de 2026  
**Escopo:** extração, armazenamento analítico, engenharia de atributos, rotulagem e preparação para scikit-learn

## 1. Resumo executivo

Recomenda-se substituir o CSV diário com objetos JSON aninhados como formato central por uma arquitetura em camadas, com tabelas Parquet normalizadas e granularidade explícita. O CSV atual pode continuar existindo como exportação de compatibilidade ou evidência bruta, mas não deve ser a interface principal entre extração e modelagem.

A decisão principal é separar cinco responsabilidades:

1. **metadados temporais do medidor e da UC**;
2. **medições intervalares**;
3. **medições instantâneas**;
4. **registradores acumulados/demanda**;
5. **atributos de modelagem e rótulos**.

A arquitetura originalmente proposta deve ser corrigida. A inspeção do arquivo de 23/06/2026 mostrou cadências heterogêneas: séries intervalares de aproximadamente 5, 10 e 15 minutos; séries instantâneas de 15 e 60 minutos; registradores predominantemente em snapshots de 6 horas. Portanto, nomes como `meter_10min.parquet` e `meter_hourly.parquet` codificariam uma hipótese falsa. A frequência deve ser uma coluna e uma regra do contrato de dados, não parte rígida do nome da tabela.

## 2. Decisão arquitetural

```text
raw / landing imutável (opcional)
        │
        ├── meter_day_metadata.parquet
        │   grão: NIO × report_day
        │
        ├── meter_interval.parquet
        │   grão: NIO × timestamp
        │   cadência observada/configurada: 5, 10 ou 15 min
        │
        ├── meter_instantaneous.parquet
        │   grão: NIO × timestamp
        │   cadência observada/configurada: 15 ou 60 min
        │
        ├── meter_register_snapshot.parquet
        │   grão: NIO × timestamp
        │   registradores acumulados/demanda; tipicamente 6 h
        │
        ├── meter_day_features.parquet
        │   grão: NIO × report_day × feature_set_version
        │   interface tabular para scikit-learn
        │
        └── labels.parquet
            grão: entidade × janela de referência × label_type × label_version
            rótulo, origem, confiança e auditoria
```

![Arquitetura recomendada](dataset_architecture.png)

## 3. Por que não manter JSON dentro do CSV

O problema não é apenas o caractere separador. Um parser CSV correto consegue preservar JSON entre aspas. O problema arquitetural é combinar dois formatos de serialização e duas granularidades na mesma linha:

- a linha externa representa um medidor-dia;
- cada JSON interno representa dezenas ou centenas de medidor-timestamp;
- cobertura, duplicatas e timestamps ficam escondidos em texto;
- filtros temporais exigem desserializar todas as células;
- o schema não expressa tipos, unidades nem frequência;
- um erro em uma célula pode inviabilizar uma linha inteira;
- consultas de colunas e intervalos não aproveitam leitura seletiva;
- o dado fica inadequado para validação, joins temporais e auditoria.

Parquet é colunar e foi projetado para armazenamento e recuperação analítica eficientes, com compressão, encoding e metadados de coluna [1]. Polars consegue aplicar projection pushdown e predicate pushdown em scans lazy, evitando ler colunas e linhas desnecessárias [2][3].

## 4. Evidência do arquivo fornecido

Auditoria do arquivo `araucaria_daily_report_20260623_pipe.csv`:

- 6.877 linhas;
- 77 colunas;
- 6.371 medidores com alguma telemetria;
- 506 sem telemetria;
- JSON sintaticamente válido nas colunas avaliadas;
- `FA_INTERVAL` apresenta padrões compatíveis com 5, 10 e 15 minutos, além de dias incompletos;
- `U_L1` apresenta grupos principais com 24 pontos/dia e grupos próximos de 96 pontos/dia;
- `FA_TOTAL` possui quatro snapshots em 5.970 medidores, compatíveis com 00:00, 06:00, 12:00 e 18:00.

Consequência: `expected_points=144` não pode ser aplicado globalmente. Cobertura diária deve usar frequência configurada ou inferida por medidor/métrica:

```text
expected_points = duration_minutes / cadence_minutes
coverage = observed_valid_points / expected_points
```

A cadência deve ser validada contra uma fonte autoritativa. Inferência por timestamp serve como controle de qualidade, não como verdade definitiva.

## 5. Contratos de dados

### 5.1 `meter_day_metadata`

**Chave candidata:** `(report_day, nio)`  
**Finalidade:** fotografia temporal da instalação e do contexto elétrico no instante de referência.

Campos principais:

```text
report_day
nio
uc_key              # preferencialmente pseudonimizada
meter_serial_key    # preferencialmente pseudonimizada
meter_type
meter_subtype
phase_type
service_status
installation_date
removal_date
consumer_class
subgroup
installed_kva
feeder_id
substation_id
nominal_voltage
municipality
latitude_bucket / longitude_bucket  # conforme necessidade e LGPD
source_updated_at
schema_version
run_id
```

Justificativa para uma linha por medidor-dia: metadados mudam no tempo. Usar apenas o cadastro atual para explicar medições históricas cria erro temporal, sobretudo em troca de medidor, mudança de alimentador ou alteração de classe. Para um TCC, snapshot diário é mais simples e auditável que uma dimensão lentamente mutável. Em escala maior, pode-se migrar para histórico com `valid_from` e `valid_to`.

### 5.2 `meter_interval`

**Chave candidata:** `(nio, timestamp_utc, measurement_source)`  
**Grão:** uma linha por medidor-timestamp.  
**Formato:** longo no tempo e largo por grandeza física estável.

```text
nio
report_day
timestamp_utc
timestamp_local
cadence_minutes
fa_interval
ra_interval
i_l1_avg
i_l2_avg
i_l3_avg
u_l1_avg
u_l2_avg
u_l3_avg
r_q1_interval
r_q2_interval
r_q3_interval
r_q4_interval
quality_code
is_estimated
source_updated_at
run_id
```

Não se recomenda uma tabela EAV (`metric`, `value`) para essas grandezas estáveis. Colunas físicas tipadas preservam unidade, aceleram agregações multivariadas e reduzem joins. O formato permanece flexível porque novos conjuntos de colunas podem ser introduzidos por versão de schema.

### 5.3 `meter_instantaneous`

**Chave candidata:** `(nio, timestamp_utc, measurement_source)`.

```text
nio
report_day
timestamp_utc
timestamp_local
cadence_minutes
u_l1
u_l2
u_l3
i_instant_l1
i_instant_l2
i_instant_l3
quality_code
source_updated_at
run_id
```

Separação justificada por semântica e frequência distintas. Misturar valores médios intervalares e snapshots instantâneos na mesma tabela produz muitos nulos, dificulta regras de cobertura e favorece comparação física incorreta.

### 5.4 `meter_register_snapshot`

**Chave candidata:** `(nio, timestamp_utc, register_group)`.

```text
nio
report_day
timestamp_utc
fa_total
fa_t1_total
fa_t2_total
fa_t3_total
fa_t4_total
ra_total
ra_t1_total
ra_t2_total
ra_t3_total
ra_t4_total
fa_md
fa_md_t1
fa_md_t2
fa_md_t3
fa_md_t4
quality_code
source_updated_at
run_id
```

Registradores acumulados não devem ser tratados como energia do intervalo. Features úteis são derivadas de diferenças temporalmente válidas, com tratamento de rollover, reset e troca de medidor.

### 5.5 `meter_day_features`

**Chave:** `(nio, report_day, feature_set_version)`.

Essa tabela é derivada, descartável e reproduzível. Não é fonte de verdade. Deve conter apenas atributos disponíveis até o instante de previsão.

Grupos recomendados:

- cobertura e qualidade;
- estatísticas robustas: mediana, IQR, p05, p95;
- energia e demanda;
- fator de carga;
- reversão de energia;
- desequilíbrio de tensão e corrente;
- variações e rampas;
- horário de pico codificado ciclicamente;
- autocorrelação e variabilidade temporal;
- contexto: fase, classe, tipo de medidor e potência instalada.

Representações compactas de séries por atributos interpretáveis são abordagem estabelecida para classificação e agrupamento temporal. `tsfresh` sistematiza centenas de atributos e seleção estatística [8]; `catch22` propõe 22 características pouco redundantes e computacionalmente eficientes [9]. Para o TCC, iniciar com atributos físicos e estatísticos definidos pelo domínio é preferível; `catch22` ou subconjunto de `tsfresh` entra como experimento comparativo, não como substituição da engenharia elétrica.

### 5.6 `labels`

Rótulos devem ser independentes das features e suportar mais de uma pergunta de pesquisa.

```text
entity_type          # NIO, UC, alimentador
entity_id
reference_start
reference_end
prediction_horizon
label_type           # falha_comunicacao, anomalia_eletrica, classe_perfil...
label_value
label_source         # campo, alarme, OS, regra, especialista
confidence
reviewer_id
created_at
label_version
notes
```

Não limitar `labels.parquet` a rótulos manuais. Validação manual deve ser distinguida de regra automática ou sistema operacional. Isso permite medir qualidade do rótulo e repetir experimentos.

## 6. Unidade de observação para scikit-learn

Para modelos tabulares clássicos:

```text
uma amostra = um NIO em um dia de referência
```

```text
X(nio, dia) = metadados válidos no dia
            + features calculadas apenas até o cutoff

y(nio, dia) = evento na janela futura definida
```

Exemplo de falha de comunicação:

```text
features: 01/07/2026 00:00 até 07/07/2026 23:59
cutoff:   07/07/2026 23:59
label:    houve falha entre 08/07/2026 e 09/07/2026?
```

Usar dados do mesmo dia para prever `HAS_MDM_DATA` do mesmo dia é vazamento direto. O scikit-learn define data leakage como uso, durante construção do modelo, de informação indisponível no momento real de predição; recomenda separar treino/teste antes de qualquer `fit` e encapsular transformações em `Pipeline` [4].

## 7. Divisão de treino, validação e teste

A estratégia depende da afirmação científica:

| Objetivo | Divisão correta | Grupo/barreira |
|---|---|---|
| Prever futuro dos mesmos medidores | temporal | treino antigo, teste recente |
| Generalizar para medidores nunca vistos | por `nio` | `GroupShuffleSplit`/`GroupKFold` |
| Generalizar para alimentadores | por `feeder_id` | alimentadores inteiros fora do treino |
| Classes desbalanceadas com repetição por medidor | estratificada + grupos | `StratifiedGroupKFold` |

`TimeSeriesSplit` existe para dados ordenados no tempo e evita treinar no futuro para testar no passado [5]. `GroupShuffleSplit` separa grupos em vez de linhas individuais [6]. `StratifiedGroupKFold` tenta preservar proporções de classe sem repetir grupos entre folds [7].

Para o TCC, recomenda-se um teste final temporal intocado. Cross-validation por grupos pode ser usada dentro do período de treino, desde que a unidade de generalização seja declarada.

## 8. Particionamento físico

Estrutura recomendada:

```text
data/
├── raw/
│   └── source=mdm/report_day=2026-06-23/run_id=.../
├── normalized/
│   ├── meter_day_metadata/report_year=2026/report_month=06/report_day=23/
│   ├── meter_interval/report_year=2026/report_month=06/report_day=23/
│   ├── meter_instantaneous/report_year=2026/report_month=06/report_day=23/
│   └── meter_register_snapshot/report_year=2026/report_month=06/report_day=23/
├── features/
│   └── feature_set_version=v1/report_year=2026/report_month=06/
├── labels/
│   └── label_version=v1/
└── manifests/
    └── run_id=...json
```

Particionar primeiro por data. Particionar também por alimentador somente se cada partição continuar suficientemente grande; milhares de arquivos pequenos degradam a leitura. `feeder_id` pode permanecer como coluna e ser filtrado por predicate pushdown.

Compressão inicial recomendada: Zstandard (`zstd`). O Parquet suporta compressão por páginas/blocos [10], e `LazyFrame.sink_parquet` permite escrita streaming para resultados maiores que a RAM [3].

## 9. Manifesto e qualidade

Cada execução deve gerar manifesto imutável:

```text
run_id
pipeline_git_commit
config_hash
query_hash
schema_version
source_window_start
source_window_end
extracted_at
row_count por tabela
min/max timestamp
unique_nio
null_count crítico
duplicate_count
late_arrival_count
file paths
file sizes
checksums
status
```

Checks mínimos:

1. unicidade de chave por tabela;
2. cobertura por medidor, métrica e cadência;
3. timestamps fora do dia;
4. valores fisicamente impossíveis;
5. tensão/corrente incompatível com fase;
6. resets de registrador;
7. troca de medidor/UC;
8. consistência entre `HAS_MDM_DATA` e presença real;
9. atraso e reprocessamento;
10. schema drift.

Não preencher pontos ausentes na camada normalizada. Imputação é decisão de modelagem e deve ocorrer após divisão treino/teste, dentro de `Pipeline` [4]. Flags de ausência e cobertura devem ser preservadas como atributos.

## 10. Alterações propostas no repositório

Evolução incremental, sem reescrita total:

```text
tcc_extraction/
├── docs/
│   ├── dataset_architecture_decision.md
│   ├── data_contracts.md
│   └── feature_catalog.md
├── queries/
│   ├── meter_day_metadata.sql
│   ├── meter_interval.sql
│   ├── meter_instantaneous.sql
│   ├── meter_register_snapshot.sql
│   └── label_sources/
├── src/
│   ├── datasets/
│   │   ├── schemas.py
│   │   ├── normalize.py
│   │   ├── partitioning.py
│   │   └── writers.py
│   ├── features/
│   │   ├── meter_day.py
│   │   ├── electrical.py
│   │   └── quality.py
│   ├── labels/
│   │   ├── build.py
│   │   └── contracts.py
│   ├── checks/
│   │   ├── cadence.py
│   │   ├── uniqueness.py
│   │   ├── physical_ranges.py
│   │   └── leakage.py
│   └── run_manifest.py
└── tests/
    ├── test_dataset_grains.py
    ├── test_cadence_detection.py
    ├── test_register_resets.py
    ├── test_feature_cutoff.py
    └── test_no_target_leakage.py
```

Mapeamento do código atual:

- `src/pipeline.py`: orquestra extrações por tabela;
- `src/export_manager.py`: passa a escrever datasets particionados e atômicos;
- `src/run_manifest.py`: permanece central e recebe estatísticas por tabela;
- `scripts/join_daily_model_input.py`: deve virar construtor versionado de `meter_day_features`, não join ad hoc;
- `src/checks/period_coverage.py`: passa a usar cadência configurada por medidor/métrica;
- `src/checks/integrity_check.py`: valida chaves, schema, unidades e temporalidade;
- CSV diário: mantido apenas como exportação humana/compatibilidade, não como contrato interno.

## 11. Sequência de implementação

### Fase 1 — Contrato e normalização

1. documentar semântica, unidade e frequência de cada campo;
2. definir chaves e timestamps;
3. dividir query atual em quatro extrações normalizadas;
4. escrever Parquet particionado;
5. criar manifesto e testes de grão.

### Fase 2 — Dataset exploratório

1. coletar 30–90 dias;
2. calcular features físicas v1;
3. executar PCA apenas para diagnóstico/visualização;
4. comparar K-Means, MiniBatchKMeans e Isolation Forest;
5. selecionar casos para revisão humana.

### Fase 3 — Rotulagem e classificação

1. integrar alarmes, eventos, OS ou inspeções;
2. congelar teste temporal;
3. criar `label_version` e `feature_set_version`;
4. treinar baseline interpretável;
5. avaliar precisão, recall, F1 e PR-AUC por classe;
6. documentar erros e incerteza.

### Fase 4 — Robustez

1. testar alimentadores externos;
2. testar medidores nunca vistos;
3. medir drift temporal;
4. reproduzir execução por manifesto e commit.

## 12. Hipóteses e decisões ainda necessárias

Para fechar o contrato v1, responder:

1. Qual é a pergunta principal do TCC e o alvo operacional?
2. Entidade estável: `NIO`, `UC`, número de série ou instalação medidor–UC?
3. Unidade e semântica exatas de cada medição: intervalo, média, snapshot ou acumulado?
4. Existe cadência configurada no MDM por medidor/canal?
5. Timestamps estão em horário local, UTC ou horário do banco?
6. Como correções tardias e reprocessamentos aparecem na fonte?
7. Quantos dias, alimentadores e medidores podem ser extraídos?
8. Quais fontes podem fornecer rótulos: alarmes, eventos, OS, campo ou especialista?
9. Generalização desejada: futuro do mesmo medidor, medidor novo ou alimentador novo?
10. Quais campos precisam ser removidos/pseudonimizados por confidencialidade?

## 13. Conclusão

A separação entre dados normalizados, features e labels não é burocracia. Ela resolve quatro riscos centrais do TCC:

- perda de informação temporal pelo JSON agregado;
- pressuposto incorreto de frequência fixa;
- vazamento entre features e alvo;
- impossibilidade de reproduzir exatamente um experimento.

Como há controle sobre a extração, a melhor decisão é corrigir o dado na origem: retornar linhas por medidor-timestamp, preservar metadados temporais, registrar cadência e qualidade, e gerar a matriz scikit-learn apenas como produto versionado. Essa arquitetura permite começar com aprendizado não supervisionado sem bloquear futura classificação supervisionada.

## Referências

[1] APACHE SOFTWARE FOUNDATION. **Apache Parquet**. Formato colunar para armazenamento e recuperação analítica eficientes. Disponível em: https://parquet.apache.org/. Acesso em: 17 jul. 2026.

[2] POLARS. **Sources and sinks**. Lazy scans, otimização de leitura e execução em streaming. Disponível em: https://docs.pola.rs/user-guide/lazy/sources_sinks/. Acesso em: 17 jul. 2026.

[3] POLARS. **LazyFrame.sink_parquet**. Escrita streaming em Parquet. Disponível em: https://docs.pola.rs/api/python/stable/reference/api/polars.LazyFrame.sink_parquet.html. Acesso em: 17 jul. 2026.

[4] SCIKIT-LEARN DEVELOPERS. **Common pitfalls and recommended practices**. Data leakage e pipelines. Disponível em: https://scikit-learn.org/stable/common_pitfalls.html. Acesso em: 17 jul. 2026.

[5] SCIKIT-LEARN DEVELOPERS. **TimeSeriesSplit**. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html. Acesso em: 17 jul. 2026.

[6] SCIKIT-LEARN DEVELOPERS. **GroupShuffleSplit**. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html. Acesso em: 17 jul. 2026.

[7] SCIKIT-LEARN DEVELOPERS. **StratifiedGroupKFold**. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html. Acesso em: 17 jul. 2026.

[8] CHRIST, M.; BRAUN, N.; NEUFFER, J.; KEMPA-LIEHR, A. W. Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package). **Neurocomputing**, v. 307, p. 72–77, 2018. DOI: 10.1016/j.neucom.2018.03.067.

[9] LUBBA, C. H. et al. catch22: CAnonical Time-series CHaracteristics. **Data Mining and Knowledge Discovery**, v. 33, p. 1821–1852, 2019. DOI: 10.1007/s10618-019-00647-x.

[10] APACHE SOFTWARE FOUNDATION. **Compression – Apache Parquet**. Disponível em: https://parquet.apache.org/docs/file-format/data-pages/compression/. Acesso em: 17 jul. 2026.
