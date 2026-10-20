# Revisão Completa do Projeto — 2026-10-20

Escopo: leitura estática de `scripts/run_araucaria_full.py`, `src/db.py`, `src/datasets/normalize.py`,
`src/models/*`, queries, configuração, dependências e testes.
**Limitação:** não executei a suíte de testes nem o pipeline (restrição de execução de comandos). Os achados
abaixo vêm de leitura de código e devem ser confirmados com `uv run pytest`.
Nem todos os módulos foram lidos a fundo (`src/pipeline.py`, `src/labels`, `src/events`, `src/automation`,
`src/checks` só foram inspecionados superficialmente).

Legenda: 🔴 crítico · 🟠 alto · 🟡 médio · 🟢 baixo

---

## 1. Segurança

| # | Sev. | Achado | Ação |
|---|---|---|---|
| S1 | 🔴 | [`config.json`](file:///d:/Projects/tcc_extraction/config.json) contém **senhas em texto puro** de ORCA, CIS, GEO, MEDPRD e Mongo (incluindo um usuário pessoal e um de serviço). Está no `.gitignore` e não encontrei o arquivo no índice do git, mas o conteúdo foi lido durante esta revisão. | **Rotacionar as senhas.** Mover para variáveis de ambiente/`.env` ou cofre. Manter só `config_example.json`. |
| S2 | 🟠 | O `.gitignore` ignora `config.json` apenas se nunca foi *commitado*. Se algum dia foi, o histórico guarda as senhas. | Verificar o histórico (`git log --all -- config.json`). Se aparecer, tratar como vazamento. |
| S3 | 🟡 | O fallback para `config.json` em [`db.py`](file:///d:/Projects/tcc_extraction/src/db.py#L80-L113) incentiva manter segredos em arquivo. | Tornar variáveis de ambiente o único caminho e falhar com mensagem clara. |
| S4 | 🟢 | O `.gitignore` ignora `vault/` inteiro. Scripts úteis ali ficam sem versionamento. | Arquivar no git ou mover para `docs/legacy`. |

---

## 2. Bugs de correção (confirmados por leitura)

| # | Sev. | Achado | Local |
|---|---|---|---|
| B1 | 🔴 | **Cache de lote por índice.** `mdm_batch_{b_idx}.parquet` em `batches_{data}` é reaproveitado se existir. Com o filtro temporal e/ou re-extração do CIS, o lote nº 1 pode conter **outros NIOs** que o de uma execução anterior, e o cache devolve dados errados em silêncio. | [`run_araucaria_full.py` L609-621](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py#L609-L621) |
| B2 | 🔴 | **Features/dataset só do último dia.** Após o *loop*, `daily_features`, `window_features` e `training_df` são calculados apenas para `report_day` (o último). Uma sincronização de 68 dias gera dataset de 1 dia. Se o último dia foi pulado por já existir, lê o parquet antigo. | L745-L790 |
| B3 | 🟠 | **`--output-format` ignorado no CLI.** `main()` não repassa `output_format` a `run_pipeline`, que usa sempre `"both"`. | L823-L837 |
| B4 | 🟠 | **Modo CSV-only quebra a etapa final:** `interval_parquet.exists()` etc. Sem parquet, `interval_df` fica vazio e as features saem vazias. | L746-L748 |
| B5 | 🟠 | **Lacunas infinitas.** `fill_gaps` parte de `2026-08-01` sem considerar a retenção do MDM (`_min_allowed_report_day` existe em `src/pipeline.py`, mas o script novo não usa). Datas já expurgadas voltam a ser tentadas a cada sincronização e nunca fecham. | L169-L205 |
| B6 | 🟠 | **Consulta de medidores não inteligentes.** A coluna `SMART` vem do CIS, mas o pipeline não a usa: todos os NIOs (~70 mil) vão ao MDM, que só tem telemetria de medidores inteligentes. Filtrar `SMART == 1` por vínculo reduz o volume e é correto com a regra temporal. | L471-L475, L518 |
| B7 | 🟡 | `meter_to_uc` (`NIO → UC`) é um dicionário sem tempo. Para NIO reutilizado em outra UC ele sobrescreve. | L750-L754 |
| B8 | 🟡 | O partição é considerada "existente" só pelo `instantaneous` (> 1000 B). Falha na normalização de `interval`/`registers` deixa o dia incompleto, mas marcado como pronto. | L510 |
| B9 | 🟡 | **Cache do CIS sem versão.** `araucaria_cis_{data}.csv` não sabe com qual SQL foi gerado. Mudanças na query exigem lembrar de forçar a re-extração. | L341-L345 |
| B10 | 🟡 | O corte `DATE '2026-01-01'` no SQL do CIS está fixo. | `cis_..._alt.sql` L20, L65 |
| B11 | 🟢 | Mutação de `sys.argv` para chamar `train_voltage_main()`. | L805-L815 |

---

## 3. Over-engineering

| # | Achado | Recomendação |
|---|---|---|
| O1 | **Quatro pipelines sobrepostos:** `src/pipeline.py` (1225 linhas, "legado diário"), `scripts/run_araucaria_full.py`, `scripts/run_feeder_pipeline.py`, `scripts/run_total_pipeline.py`, mais `vault/araucaria_sample_pipeline.py`. Lógica de CIS×GEO e extração duplicada; correções (como a do join de hoje) precisam ser replicadas. | Eleger **um** pipeline. Extrair funções comuns para `src/`. Apagar ou arquivar o resto. |
| O2 | `run_pipeline()` com ~540 linhas e 12 parâmetros mistura: seleção de datas, I/O de Oracle, UI Rich, threads, normalização e treino. | Separar em `plan_dates`, `load_population`, `extract_day`, `normalize_day`, `build_dataset`. UI fica numa camada fina. |
| O3 | `Progress` e `ThreadPoolExecutor` recriados por dia; conexões abertas/fechadas por dia (68 vezes). | Pool de conexões único por execução. |
| O4 | **Hierarquia de modelos:** `composite`, `moe_router`, `hierarchical_trainer`, `spurious_filter`, `voltage_classifier` (572 linhas) para um problema que, segundo `PRODIST_model.md`, ainda tem como referência um limiar determinístico. | Manter o determinístico como baseline oficial. Só promover o ML quando superar o baseline em teste *temporal*. |
| O5 | `ModelRegistry` caseiro (JSON + joblib). | Adequado para o TCC. Se crescer, usar MLflow. Não investir mais aqui agora. |
| O6 | `clawmetry` em `pyproject.toml`/`requirements.txt` sem nenhum `import` encontrado em `src`/`scripts`. | Remover se de fato não usado. |
| O7 | `requirements.txt` duplica `pyproject.toml`. `pytest` está em `dependencies` *e* em `dev`. | Manter `pyproject.toml` + `uv.lock`; remover `requirements.txt`; mover `pytest` para `dev`. |
| O8 | Dois formatos de saída em paralelo (CSV `;` + Parquet) para tudo, inclusive lotes de 70 mil NIOs × JSON de 288 slots. | CSV é útil para auditoria; considerar CSV apenas para tabelas de contexto e Parquet para telemetria. |
| O9 | Código morto: `write_parquet`, `lock`, `population_nios` (agora redundante), `ConfigKey`/`MEDPRD` não suportado em `db.py`. | Limpar. |
| O10 | SQLs de teste/variantes: `geo_2.sql`, `geo_ucs_test_HARDCODED.sql`, `geo_ucs*.sql`. | Remover o que não é usado. |

---

## 4. Desempenho e a lentidão relatada

| # | Sev. | Achado |
|---|---|---|
| P1 | 🔴 | **Gargalo provável:** [`normalize_mdm_dataframe`](file:///d:/Projects/tcc_extraction/src/datasets/normalize.py#L556-L628) percorre linha a linha (`iter_rows`) e constrói **3 DataFrames Polars por NIO por dia**, com `json.loads` em Python por coluna. Para ~70 mil NIOs/dia isso domina o tempo de CPU. A paralelização por *threads* não ajuda, pois o GIL serializa esse trabalho. Medir antes de afirmar. |
| P2 | 🟠 | Normalizar vetorialmente: `str.json_decode`/`explode` do Polars por coluna, ou no próprio SQL. Se for manter Python, usar `ProcessPoolExecutor`. |
| P3 | 🟠 | O JSON por coluna e slot (288 pontos × N medidas) é *pivot* pelo Oracle e *unpivot* em Python. Considerar devolver formato longo (`NIO, ts, medida, valor`) na query. |
| P4 | 🟡 | `map_elements(_normalize_key)` em ~72 mil linhas: trocar por `str.replace_all(r"\D","")` + `str.strip_chars_start("0")`. |
| P5 | 🟡 | `fetchall()` carrega o lote inteiro em RAM antes de gravar. Para lotes grandes usar `fetchmany` + escrita incremental. |
| P6 | 🟡 | `batch_size=500` gera 142 lotes/dia. Um lote por consulta adiciona latência fixa. Testar 1000–2000 e medir. |
| P7 | 🟡 | A consulta no MDM refaz *join* do catálogo por lote. Resolver `data_id` dos NIOs uma vez por execução. |
| P8 | 🟢 | `NullPool` em [`db.py`](file:///d:/Projects/tcc_extraction/src/db.py#L129-L133) combinado com `pool_pre_ping` não tem efeito prático. |

> Recomendo **instrumentar** (tempo por etapa: Oracle, `fetchall`, escrita, normalização, *sink*) antes de otimizar.

---

## 5. Modelagem e features

| # | Sev. | Achado |
|---|---|---|
| F1 | 🔴 | **Associação UC×NIO** (ver [`time_series_conundrum.md`](file:///d:/Projects/tcc_extraction/docs/article/time_series_conundrum.md)): a junção telemetria→UC deve ser temporal. Hoje `meter_to_uc` é estático (B7). |
| F2 | 🟠 | **Vazamento por cadastro atual:** `cad_uc_ee` entrega classe, fase, tensão e situação *de hoje* para qualquer data passada. Guardar *snapshots* diários do CIS ou versionar o cadastro (SCD2). |
| F3 | 🟠 | **Rótulos derivados de limiar:** se o alvo é "violou PRODIST" e as features incluem a própria tensão, o modelo aprende o limiar (circularidade). O objetivo declarado (aproximar o determinístico) é legítimo, mas a avaliação deve ser: (a) concordância com o determinístico, (b) casos fora do limiar (a parte realmente informativa). |
| F4 | 🟠 | **Validação:** usar divisão **temporal** (treino em datas anteriores, teste em posteriores) e **por UC/alimentador**; evita que o mesmo medidor apareça nos dois lados. Confirmar em `train_voltage_classifier.py` (não revisado a fundo). |
| F5 | 🟡 | **Desbalanceamento:** classes de falta/interrupção são raras. Reportar F1 macro *e* por classe, PR-AUC, matriz de confusão; não só *accuracy*. |
| F6 | 🟡 | **Tensão base fixa em 127 V** como padrão (`tensao_base_default`). Araucária tem 127/220 V. Tensão errada gera falsa sub/sobretensão em massa. Preferir a tensão do cadastro e sinalizar quando faltar. |
| F7 | 🟡 | Distinguir ausência estrutural (fase inexistente), ausência de telemetria (comunicação) e zero real. O código trata, mas depende de `FASE_CIRCUITO` do cadastro atual (F2). |
| F8 | 🟡 | Registradores acumulados: derivar por NIO e tratar *reset* na troca (ver conundrum, item 5.2). |
| F9 | 🟢 | Feriados, temperatura e demais *features* externas (`weather.py`) só valem se houver datas completas e sem lacuna. |

---

## 6. Defasamento e dados (obsolescência)

| # | Sev. | Achado |
|---|---|---|
| D1 | 🟠 | **Retenção do MDM de ~4–5 meses**, expurgada por mês. Dados de ago/2026 desaparecerão. É preciso **arquivar** o *data lake* (backup/versionamento), pois ele é insubstituível. `output/` está no `.gitignore` e não tem política de backup. |
| D2 | 🟠 | Cache de CIS/GEO por dia, mas sem invalidação por mudança de SQL (B9). |
| D3 | 🟡 | `default_start_date=2026-08-01` e `2026-01-01` fixos no código. Mover para configuração. |
| D4 | 🟡 | Documentação com possível defasagem: `.instructions.md` descreve `run_araucaria_full` como "Parquet"; `README.md` e `docs/dataset_*` precisam ser relidos contra o código atual (CSV+Parquet, sincronização inteligente, vigência temporal). `docs/nova_arquitetura_dados.md` e `dataset_architecture_decision.md` não foram verificados. |
| D5 | 🟡 | `docs/vulnerabilities.md` é anterior a esta revisão; confrontar com S1–S4. |

---

## 7. Ferramentas e dependências

| # | Sev. | Achado |
|---|---|---|
| T1 | 🟡 | `numpy>=2.4.6`, `polars>=1.38.1`, `pytest>=9.1.0` são limites inferiores muito altos e talvez inexistentes na data de instalação. Dependem de `uv.lock` para reprodutibilidade. Confirmar com `uv sync`. |
| T2 | 🟡 | Sem `ruff`/`mypy`/*pre-commit*. As regras do projeto exigem *type hints*, mas nada as verifica. Adicionar `ruff` e `mypy` (ou `pyright`) ao grupo `dev`. |
| T3 | 🟡 | Sem CI. Rodar `pytest` e `ruff` a cada *push*. |
| T4 | 🟡 | `matplotlib` e `joblib` declarados; confirmar uso. |
| T5 | 🟢 | `activate.bat` (16 KB) na raiz é gerado pelo `venv`; não deveria estar versionado. |
| T6 | 🟢 | `.cmd` soltos na raiz (`run_feeder.cmd`, `aggregate_alarms.cmd`): transformar em tarefas no `tasks.json` ou no TUI. |
| T7 | 🟢 | Pytest sem marcadores para separar testes lentos/E2E. |

---

## 8. Testes

- Há 30 arquivos de teste, incluindo `test_no_target_leakage`, `test_feature_cutoff` e `test_full_pipeline_e2e`: boa base.
- **Faltam testes para o que mudou hoje:** (a) regra de vigência temporal (dia da troca inclui os dois NIOs; NIO retirado antes de `D` fora; instalação após `D` fora), (b) join CIS×GEO sem `NUM_POSTO_OPERACIONAL`, (c) cache de lote com composição diferente (B1), (d) `main()` repassando `output_format` (B3), (e) CSV-only até o fim (B4).
- A regra temporal está embutida em `run_pipeline`; extrair para função pura (`active_nios_on(day, population)`) para ser testável sem Oracle.
- Os testes `test_vulnerabilities_remediation.py` e `test_smart_backfill.py` devem ser relidos após as mudanças em `determine_smart_extraction_dates`.

---

## 9. UX/TUI

- Opção 1→[4] agora pergunta sobre re-extrair o CIS, mas o padrão `s` força a re-extração em toda sincronização; considerar "somente se o cache tiver mais de N dias ou SQL mudou".
- `max_workers=5` e `batch_size=500` fixos no TUI; expor em Configurações.
- Barras por *worker* presentes; faltam métricas de vazão (linhas/s, MB/s) e tempo por etapa.

---

## 10. Prioridades sugeridas

**Antes da próxima sincronização completa**
1. Rotacionar credenciais (S1).
2. Corrigir B1 (chave de cache de lote por conteúdo/hash dos NIOs).
3. Filtrar `SMART == 1` (B6).
4. Corrigir B2: gerar features para **todos** os dias novos (ou deixar claro que é fase separada).
5. Repassar `output_format` no CLI (B3).

**Curto prazo**
6. Função pura de vigência + testes (seção 8).
7. Instrumentar o tempo; reescrever a normalização vetorizada (P1–P3).
8. Limitar `fill_gaps` à janela de retenção do MDM (B5) e planejar *backup* do *lake* (D1).
9. Remover `clawmetry`, `requirements.txt` e SQLs mortos.

**Médio prazo**
10. Consolidar os pipelines (O1/O2).
11. Snapshots diários do CIS / SCD2 (F2).
12. Validação temporal + por alimentador e avaliação orientada ao determinístico (F3–F5).
13. `ruff` + `mypy` + CI.
