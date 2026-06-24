# Relatório Técnico — Pipeline de Extração ARAUCARIA

## Histórico do Documento

| Data | Versão | Autor | Descrição |
|------|--------|-------|-----------|
| 2026-06-23 | 1.0 | Time de Dados | Relatório completo da pipeline de extração ARAUCARIA |

---

## 1. Visão Geral

Este documento descreve a **pipeline de extração diária ARAUCARIA**, que integra dados de medidores inteligentes da região de Araucária (PR) vindos de múltiplos sistemas Oracle — **CIS** (Cadastro), **GEO** (Geográfico) e **ORCA/AMI** (Medição) — para alimentar modelos de *machine learning*.

O pipeline foi construído com base em **experiências anteriores** em projetos semelhantes, onde enfrentamos problemas recorrentes:

- **Consultas muito grandes** que travavam o banco ou estouravam memória local.
- **Arquivos que não salvavam corretamente** por falta de *flush* ou interrupção no meio da escrita.
- **Processamento monolítico** que, ao falhar no meio, exigia recomeçar do zero.

Cada um desses problemas foi endereçado com uma solução específica, detalhada nas seções a seguir.

---

## 2. Arquitetura Geral

### 2.1 Fluxo da Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DIÁRIO (pipeline.py)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CIS ──► extrai NIOs ──► GEO (batch por UC) ──► join em UC         │
│                              │                                      │
│                    ┌─────────┴─────────┐                            │
│                    ▼                    ▼                            │
│            WEATHER (Open-Meteo)   MDM/ORCA (batch por NIO)          │
│            exportado como CSV    join em NIO ──► relatório final    │
│            (separado)                                                │
│                              │                                      │
│                              ▼                                      │
│            TIMEGRID (batch por NIO) ──► grade 5 min                 │
│                              │                                      │
│                              ▼                                      │
│            Publish para OneDrive + Manifest JSON                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Nota: Weather é extraído e exportado como CSV separado. A junção
do weather no relatório final (antes do MDM) será implementada
futuramente — veja seção 10.
```

### 2.2 Tecnologias Utilizadas

| Componente | Tecnologia | Motivo |
|-----------|-----------|--------|
| Orquestração | Python 3.11+ | Maturidade, ecosistema de dados |
| Banco de dados | Oracle via SQLAlchemy + oracledb | Acesso direto às bases legadas |
| Processamento de dados | **Polars** | Paralelização nativa, lazy evaluation, streaming |
| Formato de intercâmbio | Parquet + CSV | Parquet para performance, CSV para entrega |
| Publicação | OneDrive sincronizado | Entrega simples para o time de ML |
| Rastreabilidade | JSON (run manifest) | Auditoria e debugging |

**Por que Polars?** Polars foi escolhido por sua **paralelização nativa** (aproveita todos os núcleos da CPU), **lazy evaluation** (só computa quando necessário), e **grande capacidade de lidar com volumes massivos de dados** sem estourar memória. Diferente do Pandas, que carrega tudo em RAM, o Polars consegue fazer *streaming* de dados usando `sink_parquet`/`sink_csv`, permitindo processar conjuntos que não caberiam em memória.

---

## 3. Etapas da Extração

### 3.1 Extração CIS (Cadastro)

**Arquivo:** `queries/cis_araucaria_ml_extract_lightweight_alt.sql`

A primeira etapa consulta o banco **CIS** para obter todos os medidores instalados em Araucária. A query:

1. Filtra medidores ativos (`dta_reti_reu IS NULL`) no município de Araucária.
2. Classifica medidores como **smart** ou não, com base na faixa de NIO (41.000.000–46.000.000) ou presença de medidor inteligente.
3. Enriquece com dados de cliente, coordenadas (LAT/LONG), tipo de medidor, subgrupo tarifário, etc.
4. Junta com geração distribuída (GD) para identificar UCs com painéis solares.

**Problema resolvido:** A query CIS original era extremamente pesada — demorava minutos e consumia muitos recursos. A versão `lightweight_alt` foi otimizada para rodar mais rápido, usando apenas as colunas essenciais para o pipeline.

### 3.2 Extração GEO (Geográfica)

**Arquivo:** `queries/geo_ucs.sql`

A partir das **UCs** (unidades consumidoras) obtidas no CIS, o pipeline consulta o banco **GEO** para obter dados geográficos de cada UC:

- Posto transformador, alimentador, subestação.
- Tensão nominal, capacidade instalada.
- Município e código do município da subestação.

#### Como o JOIN GEO funciona

O join entre CIS e GEO é feito pela coluna **UC** (unidade consumidora). O pipeline:

1. Extrai todas as UCs únicas do DataFrame CIS.
2. Alimenta essas UCs em lotes (batches) na query GEO via `SYS.ODCIVARCHAR2LIST`.
3. Recebe os dados GEO e faz um **left join** por UC, prefixando as colunas GEO com `GEO_`.

```
CIS (UC, NIO, cliente, lat, lon, ...)
  │
  │ LEFT JOIN por UC
  ▼
CIS+GEO (UC, NIO, ..., GEO_POT_INST_KVA, GEO_ALIMENTADOR, ...)
```

**Característica importante:** O join usa `inner` no pipeline principal (só retorna UCs com dados GEO), mas a versão de amostra usa `left join` para preservar todas as UCs do CIS.

### 3.3 Extração MDM/ORCA (Medição)

**Arquivo:** `queries/mdm_coluna.sql`

Esta é a etapa mais crítica e pesada. Para cada **NIO** (número do medidor), o pipeline consulta o banco **ORCA/AMI** para obter as medições do dia:

- Energia ativa (forward/reverse) — intervalar e totalizada.
- Tensão e corrente — média e instantânea por fase.
- Energia reativa por quadrante.
- Demanda máxima diária e por patamar tarifário.

A query usa uma **grade temporal de 288 intervalos** (5 min cada → 24h), gerando **uma linha por NIO por intervalo** via `CONNECT BY LEVEL <= 288`.

#### Normalização de NIO

Antes de usar os NIOs como chave, o pipeline aplica uma **normalização** que:

1. Remove tudo que não é dígito (`\D`).
2. Remove zeros à esquerda.
3. Converte para string.

Isso garante que `00123456` e `123456` sejam tratados como o mesmo NIO, evitando duplicatas e falhas de join.

### 3.4 Extração TIMEGRID (Grade 5 min)

**Arquivo:** `queries/memoria_de_massa_nio_list.sql`

Similar ao MDM, mas focado exclusivamente na grade horária de 5 minutos com todas as variáveis elétricas (tensão, corrente, energia ativa/reativa, demanda). Gera 288 linhas por NIO.

### 3.5 Extração WEATHER (Clima — Open-Meteo)

**Fonte:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive)

Após o GEO, o pipeline consulta a **API pública Open-Meteo** para obter dados climáticos históricos do dia do relatório. As coordenadas (LAT/LONG) vêm do DataFrame CIS.

#### Funcionamento

1. O pipeline obtém as coordenadas médias (LAT/LONG) dos NIOs do dia a partir do CIS.
2. Constrói uma URL para a API Archive do Open-Meteo com as variáveis solicitadas:
   - `temperature_2m`, `precipitation`, `rain`, `cloud_cover`
   - `wind_speed_10m`, `wind_direction_10m`
3. A API retorna um JSON com dados horários (24 linhas).
4. O JSON é convertido em um DataFrame Polars com colunas:
   - `NIO`, `DIA`, `TEMPERATURE_2M`, `PRECIPITATION`, `RAIN`, `CLOUD_COVER`, `WIND_SPEED_10M`, `WIND_DIRECTION_10M`
5. O DataFrame é salvo como Parquet + CSV no diretório `output/raw/WEATHER/`.

#### Estado Atual

Atualmente o weather é **exportado como CSV separado** do relatório final. O relatório final contém apenas CIS + GEO + MDM.

#### Plano Futuro (ver seção 10)

No futuro, o weather será **joined no relatório final antes do MDM**, seguindo a ordem:

```
CIS + GEO → JOIN weather (por NIO) → JOIN MDM (por NIO) → relatório final
```

Isso permitirá que as colunas climáticas estejam disponíveis como features nos modelos de ML, mantendo a precedência correta: weather antes de MDM para evitar conflito de nomes (ambos têm `DIA`).

---

## 4. Estratégias de Resiliência e Performance

### 4.1 Batching (Divisão de Pressão)

Este é um dos pilares do pipeline. Em vez de enviar uma consulta gigante com milhares de NIOs de uma só vez, o pipeline **divide em lotes**:

```python
# No código: _chunked() divide a lista em pedaços
def _chunked(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])
```

**Configuração padrão:**

| Etapa | Tamanho do lote | Parâmetro CLI |
|-------|----------------|---------------|
| GEO | 500 UCs | `--geo-batch-size` |
| MDM | 500 NIOs | `--mdm-batch-size` |
| TIMEGRID | 500 NIOs | (usa o mesmo `--mdm-batch-size`) |

**Benefícios:**
- **Banco não trava:** cada lote é uma consulta pequena que o Oracle processa rapidamente.
- **Paralelismo implícito:** lotes são processados sequencialmente, mas cada um já aproveita o paralelismo do Polars.
- **Retomada possível:** se um lote falha, é possível identificar exatamente qual lote foi afetado e reexecutar apenas ele.
- **Memória controlada:** cada lote gera um DataFrame pequeno que é escrito em disco como Parquet temporário e depois descartado.

### 4.2 Fetch Size (Controle de Linhas por Viagem)

Além do batching, o pipeline controla **quantas linhas são trazidas por round-trip** ao banco:

```python
while True:
    rows = result.fetchmany(fetch_size)  # 1000 linhas por vez
    if not rows:
        break
    dataframe_chunks.append(_rows_to_frame(rows, current_columns))
```

Isso evita que uma única consulta que retorna milhões de linhas consuma toda a RAM disponível. As linhas são processadas em blocos e acumuladas em chunks que são posteriormente concatenados.

### 4.3 Parquet Temporário + Sink para Arquivo Final

Em vez de manter tudo em memória, cada lote do MDM/TIMEGRID é **salvo como Parquet temporário**:

```python
batch_df.write_parquet(batch_path)   # salva lote em disco
batch_files.append(batch_path)       # guarda referência
```

Ao final, todos os Parquets temporários são **mergeados usando lazy evaluation com `sink_parquet`/`sink_csv`**:

```python
lazy_frame = pl.scan_parquet([str(path) for path in parquet_files]).sort(sort_by)
lazy_frame.sink_parquet(final_parquet)   # streaming para disco
lazy_frame.sink_csv(final_csv, separator=";")  # streaming para CSV
```

**Por que `sink` em vez de `write`?** O método `sink_parquet`/`sink_csv` do Polars faz **streaming** dos dados diretamente para o disco sem carregar tudo em memória. Isso foi uma resposta direta a problemas anteriores onde:

- Arquivos CSV enormes não salvavam por falta de espaço em RAM.
- O processo morria no meio e todo o processamento era perdido.
- Não havia checkpoint para retomada.

Com `sink`, o Polars processa em streaming: ele lê um pedaço do dado, escreve no arquivo final, libera memória, e continua. Se o processo cair, pelo menos os lotes já salvos em Parquet temporário estão preservados.

### 4.4 Modo Período e Continuação em Erro

O pipeline suporta dois modos de execução:

| Modo | Flag | Comportamento |
|------|------|---------------|
| **Dia único** | `--days-back N` | Extrai um único dia |
| **Período** | `--start-date` / `--end-date` | Itera sobre múltiplos dias |

No **modo período**, o pipeline oferece `--continue-on-error` (ativado por padrão). Se um dia falha:

```python
try:
    result = run_daily_araucaria_pipeline(...)
except Exception as exc:
    _log(f"Day {day} FAILED: {exc}")
    if not continue_on_error:
        raise
    # Registra placeholder e continua para o próximo dia
```

**Isso significa que** se o pipeline está processando 30 dias e o dia 15 falha (ex.: banco fora do ar), os dias 16–30 continuam sendo processados. No final, o relatório mostra exatamente quantos dias falharam. Basta reexecutar apenas o dia 15.

### 4.5 Run Manifest (Rastreabilidade)

Cada execução gera um **manifest JSON** em `output/runs/run_YYYYMMDD_HHMMSS.json`:

```json
{
  "run_id": "20260622_140037",
  "report_day": "2026-06-21",
  "steps": {
    "cis_extract": {"rows": 15432},
    "geo_extract": {"rows": 893},
    "geo_join": {"rows": 15432},
    "mdm_extract": {"rows": 288000},
    "timegrid_extract": {"rows": 288000},
    "final_join": {},
    "publish": {"target": "E:/mion/OneDrive...", "bytes": 52340}
  },
  "errors": [],
  "duration_seconds": 142.5
}
```

Isso permite:
- **Auditar** o que foi executado e quando.
- **Depurar** falhas vendo exatamente qual etapa falhou.
- **Comparar** execuções ao longo do tempo.

### 4.6 Diretório Temp e Flag `--keep-temp`

Durante a execução, cada batch do MDM/TIMEGRID é salvo em um diretório temporário:

```
output/tmp/araucaria_YYYYMMDD/
  ├── mdm_batch_00001.parquet
  ├── mdm_batch_00002.parquet
  └── ...
```

Por padrão, esse diretório é **excluído ao final**. Com `--keep-temp`, os arquivos temporários são preservados para debug. Se algo der errado, você pode inspecionar os batches individualmente.

### 4.7 Verificação de Integridade (Sample Pipeline)

A pipeline de amostra (200 NIOs) inclui uma etapa de **verificação pós-join** que checa:

1. Arquivo existe e não está vazio.
2. Colunas obrigatórias (`UC`, `NIO`) presentes.
3. Nenhum NULL nas colunas-chave.
4. Contagem de linhas coerente (tolerância de ±10%).
5. Sem duplicatas nas chaves primárias.

Se a verificação falha, o pipeline retorna código de erro 1 e não publica.

---

## 5. Sistema de Publicação (Export Manager)

### 5.1 Para OneDrive

O CSV final do relatório é copiado para um diretório OneDrive sincronizado:

```
E:\mion\OneDrive - copel.com\transfer_area\machine_learning_pipeline\input\sample200\
```

Junto com o CSV, é gerado um arquivo `.meta.json` com metadados:

```json
{
  "source": "output/refined/reports/araucaria_daily_report_20260621.csv",
  "target": "E:/mion/OneDrive/.../araucaria_daily_report_20260621.csv",
  "copied_at": "2026-06-22T14:05:00+00:00",
  "row_count": 15432,
  "file_size_bytes": 5242880
}
```

**Backup automático:** Se o arquivo já existe no target, ele é copiado para `backups/` antes de sobrescrever.

### 5.2 Flags de Publicação

| Flag | Efeito |
|------|--------|
| `--publish-target PATH` | Diretório customizado |
| `--no-publish` | Pula a etapa de publicação |

---

## 6. Evolução e Próximos Passos

### 6.1 Origens do Projeto

Este pipeline é o resultado direto de **lições aprendidas em projetos anteriores**:

| Problema Anterior | Solução Implementada |
|-------------------|---------------------|
| Consultas Oracle enormes travavam o banco | **Batching** com `SYS.ODCIVARCHAR2LIST` |
| Script morria no meio e perdia tudo | **Parquet temporário** + `sink` streaming |
| Arquivo CSV corrompido por falta de *flush* | Escrita via Polars com buffer controlado |
| Sem rastreabilidade de execuções | **Run manifest** JSON |
| Não sabia qual dia falhou em execução em lote | **Modo período** com `continue-on-error` |

### 6.2 Para Produção (Próximos Passos)

O código atual é uma **prova de conceito funcional** que roda na máquina local. Para produção no servidor da empresa, as seguintes melhorias são planejadas:

1. **Vetorização completa:** Substituir loops Python por operações vetorizadas do Polars sempre que possível.
2. **Streaming de dados:** Usar `sink_parquet` e `sink_csv` extensivamente para processar volumes que não cabem em memória (centenas de milhões de linhas).
3. **Paralelização distribuída:** Avaliar **Dask** para distribuir o processamento em múltiplos nós do cluster.
4. **Escrita atômica:** Escrever em arquivo temporário e renomear, garantindo que arquivos nunca fiquem corrompidos.
5. **Monitoramento:** Integrar com logging centralizado e alertas para falhas.
6. **Agendamento:** Substituir execução manual por agendamento (cron / Airflow).

### 6.3 Polars vs. Pandas vs. Dask

| Característica | Pandas | Polars | Dask |
|---------------|--------|--------|------|
| Paralelização | Limitada (1 core) | **Nativa (multicore)** | Distribuída (cluster) |
| Lazy Evaluation | ❌ | ✅ | ✅ |
| Streaming | ❌ | ✅ (`sink_*`) | ✅ |
| Memória | Tudo em RAM | **Streaming possível** | Sharding em disco |
| Curva de aprendizado | Baixa | Média | Alta |
| Ideal para | Datasets pequenos | **Médios a grandes** | **Muito grandes** |

**Para produção**, a hierarquia recomendada é:

- **Polars** para a pipeline diária (dados cabem em uma máquina potente).
- **Dask** para escala futura quando o volume superar a capacidade de uma única máquina (ex.: todo o estado do Paraná).

### 6.4 Servidor Empresarial

O plano é migrar a execução para o **servidor da empresa**, que tem:

- Múltiplos núcleos e grande quantidade de RAM.
- Acesso direto aos bancos Oracle com baixa latência.
- Capacidade de armazenamento em escala.

Isso permitirá:
- Processar **todos os medidores do estado** sem se preocupar com limites locais.
- Executar em paralelo com outros pipelines.
- Agendar execuções automáticas diárias.

---

## 7. Como Executar

### 7.1 Modo Dia Único

```bash
uv run python src/main.py --days-back 1
```

### 7.2 Modo Período (Múltiplos Dias)

```bash
uv run python src/main.py \
    --start-date 2026-06-01 \
    --end-date 2026-06-21
```

### 7.3 Parâmetros Opcionais

```bash
uv run python src/main.py \
    --days-back 1 \
    --mdm-batch-size 300 \        # Menor batch = menos pressão no banco
    --geo-batch-size 300 \
    --fetch-size 500 \             # Menor fetch = menos memória
    --keep-temp \                  # Preserva arquivos temporários para debug
    --continue-on-error \          # Continua mesmo se um dia falhar
    --no-publish                   # Só testar, sem publicar
```

### 7.4 Pipeline de Amostra (200 NIOs)

```bash
uv run python scripts/araucaria_sample_pipeline.py \
    --days-back 1 \
    --sample-size 200
```

---

## 8. Estrutura de Saída

```
output/
├── raw/
│   ├── CIS/
│   │   ├── araucaria_cis_YYYYMMDD.parquet
│   │   ├── araucaria_cis_YYYYMMDD.csv
│   │   └── daily_cadastrados/ucs_YYYYMMDD.parquet
│   ├── GEO/
│   │   └── araucaria_geo_ucs_YYYYMMDD.parquet
│   ├── WEATHER/
│   │   ├── araucaria_weather_YYYYMMDD.parquet
│   │   └── araucaria_weather_YYYYMMDD.csv
│   ├── ORCA/
│   │   ├── araucaria_mdm_YYYYMMDD.parquet
│   │   └── araucaria_mdm_YYYYMMDD.csv
│   └── TIMEGRID/
│       └── araucaria_timegrid_YYYYMMDD.parquet
├── refined/
│   └── reports/
│       ├── araucaria_daily_report_YYYYMMDD.parquet
│       └── araucaria_daily_report_YYYYMMDD.csv
├── runs/
│   └── run_YYYYMMDD_HHMMSS.json
└── tmp/
    └── araucaria_YYYYMMDD/    (excluído por padrão)
```

---

## 9. Conclusão

A pipeline ARAUCARIA foi construída com **foco em resiliência, rastreabilidade e escalabilidade**. Cada decisão de design foi tomada com base em problemas reais enfrentados em projetos anteriores:

1. **Batching** resolve consultas gigantes que travavam o banco.
2. **Parquet temporário + sink** garante que a perda de dados seja mínima se algo falhar.
3. **Run manifest** permite auditar e depurar cada execução.
4. **Modo período com continue-on-error** facilita processar grandes intervalos sem medo.
5. **Polars** traz paralelização nativa e capacidade de streaming.

Para produção, as melhorias incluirão vetorização completa, streaming obrigatório, escrita atômica, e eventualmente **Dask** para processamento distribuído no servidor da empresa, que tem capacidade muito superior à máquina local.

> *"O pipeline certo é aquele que não apenas funciona hoje, mas que te dá confiança para escalar amanhã."*

---

## 10. Próximos Passos

### 10.1 Weather no Relatório Final (JOIN antes do MDM)

**Objetivo:** Incluir as colunas climáticas (temperatura, precipitação, etc.) como features no relatório final, para que os modelos de ML possam utilizá-las.

**Ordem proposta:**

```
CIS + GEO → JOIN weather (por NIO) → JOIN MDM (por NIO) → relatório final
```

**Motivo da ordem:** Weather e MDM têm a coluna `DIA` em comum. Para evitar conflito, o weather deve ser joined **antes** do MDM, e a coluna `DIA` do weather deve ser excluída do join (apenas colunas climáticas são adicionadas). O MDM é o último a ser joined para que sua coluna `DIA` (a mais relevante para o modelo) seja a versão final no relatório.

**Status:** ⏳ Pendente — weather atualmente é exportado como CSV separado.

### 10.2 Melhorias Visuais (Rich / TUI)

**Objetivo:** Substituir `print()` por `rich` para obter saída colorida com tabelas, painéis e progress bars.

**Referência:** O projeto `gcp-validation` já utiliza `rich>=14.2.0` com bons resultados.

### 10.3 Servidor Empresarial

Migrar a execução para o servidor da empresa, permitindo agendamento automático e processamento em escala.

---

*Documento gerado em 2026-06-23 · Última atualização: 2026-06-24*
