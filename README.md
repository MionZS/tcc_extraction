# TCC Extraction — Pipeline AMI & Detecção de Anomalias (Copel / Araucária)

Pipeline de extração, normalização, engenharia de atributos e modelagem de ML
para detecção de anomalias em medição inteligente (AMI), focado em alimentadores
da Copel (piloto: **Fonte Nova**, Araucária — `GEO_ID = 6352460`).

Unidade analítica: **`UC × cutoff_date`** — todo o lake converge para a matriz final
`output/model_input/v1/training_dataset.parquet` (convenção `id__* / meta__* / x__* / y__*`).

---

## 1. Uso rápido (TUI via `uv`)

> [!IMPORTANT]
> Toda interação deve ser feita pelo ambiente gerenciado do `uv`, via TUI em Rich:
>
> ```bash
> uv run main.py
> ```

O `uv` sincroniza/instala as dependências de `pyproject.toml`
(`rich`, `polars`, `scikit-learn`, `oracledb`, `sqlalchemy`, …).

Executar `python scripts/...` fora do `uv`/venv exibe o aviso de ambiente não isolado
(`show_environment_warning()` em `src/tui/tui_app.py`).

Menu da TUI (7 opções — `src/tui/tui_app.py::RichPipelineTUI.run_menu`):

| # | Opção | O que faz (código) |
|---|-------|--------------------|
| 1 | 📥 Gerar Dataset | Extração por alimentador (`scripts/run_feeder_pipeline.py`) ou Araucária completa (`src/araucaria_sample_pipeline.py`) |
| 2 | 🤖 Treinar & Validar Modelos | `scripts/train_anomaly_model.py --mode composite\|hierarchical\|baseline` |
| 3 | 📊 Estatísticas do Data Lake | Tamanhos/dimensões das tabelas Parquet + UCs por alimentador |
| 4 | ⚡ Especialistas vs Global | Comparação no teste isolado (`compare_experts`) |
| 5 | 🩺 Auditoria de Saúde & Tests | `DailyAutomationRunner.run_daily_audit()` → `output/reports/daily_health_report_*` |
| 6 | ⚙️ Configurações Globais | `contamination` (0.001–0.50, padrão 0.05) e `random_state` (padrão 42) da sessão |
| 7 | 🚪 Sair | Encerra a TUI |

Guia de telas e parâmetros: [`docs/TUI_INSTRUCTIONS.md`](docs/TUI_INSTRUCTIONS.md).

### Atalhos sem TUI (dentro do `uv`)

```bash
# Pipeline diária CIS → GEO → weather → MDM → join → publish
uv run python -m src.main --days-back 1
uv run python -m src.main --start-date 2026-06-01 --end-date 2026-06-10

# Alimentador específico (Fonte Nova) + treino baseline
run_feeder.cmd
# ou: uv run python scripts/run_feeder_pipeline.py --days-back 1 --train-model

# Treino direto (3 arquiteturas)
uv run python scripts/train_anomaly_model.py --mode composite
uv run python scripts/train_anomaly_model.py --mode hierarchical --feeder Fonte_Nova
uv run python scripts/train_anomaly_model.py --mode baseline --contamination 0.05

# Auditoria de saúde + testes
uv run python scripts/daily_automation.py --mode health-check
uv run python scripts/daily_automation.py --mode test-only
uv run pytest tests -q
```

---

## 2. Arquitetura de dados (camadas semânticas)

Princípio: **AMI é fenômeno. Cadastro, medidor e GEO são contexto. Alarmes são eventos.**
Tudo é projetado na matriz `UC × cutoff_date`.

```text
output/
├── raw/                        # brutos por dia (CIS / GEO / ORCA / WEATHER / TIMEGRID)
├── context/                    # uc_context / meter_installation_history / electrical_hierarchy (.parquet)
├── measurements/               # ami_interval / ami_instantaneous / ami_registers (particionado por dia, zstd)
├── events/                     # alarm_events (scripts/aggregate_alarms.py)
├── features/                   # uc_day_features + uc_window_features (feature_set_version=v1)
├── model_input/v1/             # training_dataset.parquet/.csv  ← única entrada do treino
├── outputs/anomaly_scores/     # scores composite/baseline com timestamp
├── models/                     # .joblib (composite / isolation_forest / kmeans / MoE)
├── reports/                    # daily_health_report_YYYYMMDD.md/.json (+ _latest.md)
└── runs/                       # run_YYYYMMDD_HHMMSS.json (manifesto imutável por execução)
```

Leitura recomendada, nesta ordem:

1. [`docs/nova_arquitetura_dados.md`](docs/nova_arquitetura_dados.md) — arquitetura vigente v2.0 (UC como entidade central).
2. [`docs/data_contracts.md`](docs/data_contracts.md) — contratos formais por tabela (grão, tipos, unidades, partições).
3. [`docs/feature_catalog.md`](docs/feature_catalog.md) — catálogo de features (`build_meter_day_features`, `build_uc_window_features`).
4. [`docs/dataset_architecture_decision.md`](docs/dataset_architecture_decision.md) — histórico da decisão (por que não JSON-no-CSV).
5. [`docs/dataset_layout_sklearn_tcc.md`](docs/dataset_layout_sklearn_tcc.md) — layout prático p/ scikit-learn (janelas 7/30 dias).
6. [`docs/vulnerabilities.md`](docs/vulnerabilities.md) — auditoria de vulnerabilidades sutis (todas corrigidas, com testes).
7. [`docs/reports/relatorio_pipeline.md`](docs/reports/relatorio_pipeline.md) — relatório da pipeline diária (histórico 2026-06; layout `raw/refined` anterior às camadas semânticas).

> Histórico: [`docs/export_and_publish_plan.md`](docs/export_and_publish_plan.md) e
> [`docs/daily_pipeline_integrity_plan.md`](docs/daily_pipeline_integrity_plan.md) descrevem a fase
> `output/raw + refined/reports` (amostra 200 NIOs, CSV). **Superseded** pela arquitetura acima;
> mantidos apenas como registro.

---

## 3. Separação de datasets (anti-leakage)

`src/datasets/dataset_splitter.py::DatasetSplitter`:

1. **Perfil elétrico** (`circuit_profile`): agrupa alimentadores por assinatura elétrica/demográfica.
2. **Isolamento por alimentador** (≥ 3 similares): 1 alimentador só-Validação, 1 só-Teste, resto Treino.
3. **Fallback temporal**: com poucos alimentadores, split out-of-time (ordem cronológica preservada).

Convenção estrita de colunas em `training_dataset` (`src/datasets/build_training_dataset.py`):

| Prefixo | Papel | Vai para X? |
|---------|-------|-------------|
| `id__*` | identificadores/auditoria (`id__uc_id`) | Não |
| `meta__*` | corte/partição (`meta__cutoff_date`, `meta__split`, `meta__feeder`) | Não |
| `x__*` | features do modelo | **Sim** |
| `y__*` | rótulos (`y__target_irregularity`, nulo no não-supervisionado) | Só como target |

---

## 4. Modelos (`scripts/train_anomaly_model.py`)

| Modo | Técnica | Saída |
|------|---------|-------|
| `composite` (padrão) | `CompositeAnomalyDetector`: IsolationForest (0.3) + LOF (0.25) + ECOD (0.25) + PCA/SVD (0.2), + bônus +0.20 por violação física PRODIST (V < 180 ou > 255 V p/ 220 V nominal; reversão RA > 0.8); limiar no percentil `100×(1−contamination)` | `models/composite_anomaly_detector.joblib`, `outputs/anomaly_scores/composite_scores_*.parquet/.csv` |
| `hierarchical` | `HierarchicalModelOrchestrator` — MoE: especialista do alimentador vs regional vs global | modelos por nível + comparação |
| `baseline` | `IsolationForest(n=200)` + `KMeans(k=4)` em pipeline `SimpleImputer(median) → RobustScaler` | `models/isolation_forest_pipeline.joblib`, `kmeans_clustering_pipeline.joblib`, `outputs/anomaly_scores/anomaly_scores_*.parquet` |

Pré-processamento sempre dentro do `Pipeline` sklearn (imputação/mediana + `RobustScaler`),
ajustado só no treino.

---

## 5. Configuração e credenciais

**Nunca commitar `config.json`** (está no `.gitignore`). Criar a partir do exemplo:

```bash
copy config_example.json config.json
```

Precedência em `src/db.py`: **variáveis de ambiente (ou `.env`) > `config.json`**.
Targets válidos: `orca` / `cis` / `geo` → blocos `ORCA` / `CIS` / `GEO`
(`ORCA_MEU`, `MEDPRD`, `mongo.SANPLAT` no exemplo são extras, não usados pelo `create_engine`).

| Variável | Exemplo |
|----------|---------|
| `ORCA_USER` / `ORCA_PASSWORD` / `ORCA_HOST` / `ORCA_PORT` / `ORCA_SERVICE_NAME` | `E806586` / … / `DBHEXPRD19` / `1521` / `HEXPRD19` |
| `CIS_USER` / `CIS_PASSWORD` / `CIS_HOST` / `CIS_PORT` / `CIS_SERVICE_NAME` | … / … / `dbcisdprd` / `1521` / `cisdprd` |
| `GEO_USER` / `GEO_PASSWORD` / `GEO_HOST` / `GEO_PORT` / `GEO_SERVICE_NAME` | … / … / `dbgeoprd` / `1521` / `geoprd` |

Ver [`.env.example`](.env.example). Conexão: SQLAlchemy + `oracledb` thin, `pool_pre_ping=True`,
`NullPool`, listas Oracle via `SYS.ODCIVARCHAR2LIST` (batches: MDM 500 NIOs, GEO 500 UCs,
fetch 1000 linhas — flags `--mdm-batch-size`, `--geo-batch-size`, `--fetch-size`).

Weather (Open-Meteo archive, `America/Sao_Paulo`, sem chave): `src/weather.py`
(`fetch_historical_weather` → 1 linha/hora; `enrich_timegrid` prefixa `weather_*` no timegrid 5 min).

---

## 6. Estrutura do repo

```text
main.py                          # entry point da TUI (uv run main.py)
run_feeder.cmd / aggregate_alarms.cmd
config_example.json / .env.example
src/
  tui/tui_app.py                 # TUI Rich (menu 1–7)
  main.py                        # CLI pipeline diária (single-day + period mode)
  pipeline.py                    # CIS → GEO → weather → MDM → join → publish
  araucaria_sample_pipeline.py   # pipeline legada de amostra 200 NIOs (CSV)
  db.py / weather.py / export_manager.py / run_manifest.py
  automation/ (daily_runner, health_check)
  checks/ (verify_output, integrity_check, period_coverage, …)
  datasets/ (normalize, partitioning, writers, build_training_dataset, dataset_splitter)
  features/ (meter_day, uc_window, circuit_profile, neighborhood)
  labels/ / models/ (composite, hierarchical_trainer, moe_router, spurious_filter)
  visualization/diagnostics.py
scripts/ (run_feeder_pipeline, train_anomaly_model, daily_automation, aggregate_alarms, join_daily_model_input)
queries/ (cis_*, geo_ucs, geo_feeder_direct, mdm_coluna, meter_*, label_sources/, weather_timegrid)
tests/ (21 arquivos — splitter, leakage, cadence, composite, MoE, weather, automação…)
docs/ (ver §2)
```

Convenções ([`.instructions.md`](.instructions.md)): Python 3.11+, type hints,
**Polars (nunca Pandas)**, SQLAlchemy/oracledb, CSV `;` + `utf-8-sig`, `pathlib`,
dataclasses frozen, helper `_log()`. Testes: `pytest tests -q`, `tmp_path`, mocks de DB.

## 7. Dicionários e referência de queries

- Amostra 200 NIOs (legado CSV): [`docs/data_dictionary_sample200.md`](docs/data_dictionary_sample200.md).
- Queries por fonte: `queries/cis_*` (cadastro), `queries/geo_*` (topologia),
  `queries/mdm_coluna.sql` + `meter_*` (fenômeno), `queries/label_sources/` (rótulos),
  `queries/weather_timegrid` + `memoria_de_massa_nio_list.sql` (grade 5 min p/ clima).

```text
output/
├── context/
│   ├── uc_context/
│   ├── meter_installation_history/
│   └── electrical_hierarchy/
├── measurements/
│   ├── ami_interval/
│   ├── ami_instantaneous/
│   └── ami_registers/
├── events/
│   └── alarm_events/
├── features/
│   ├── uc_day_features/
│   └── uc_window_features/ (ex: 30 dias)
└── model_input/
    └── training_dataset.parquet (formatado com x__*, id__*, meta__*, y__*)
```

---

## 5. Instruções Detalhadas de Uso

Consulte o documento [TUI_INSTRUCTIONS.md](file:///d:/Projects/tcc_extraction/docs/TUI_INSTRUCTIONS.md) para um guia completo sobre as telas, parâmetros de treinamento e auditoria diária automatizada.
