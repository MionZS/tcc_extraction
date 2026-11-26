# Apostila de Domínio Específico e Guia Completo do Repositório — Plataforma Analítica AMI (TCC Engenharia Elétrica UFPR)

> **Para quem é esta apostila?**  
> Para qualquer pessoa — de um leigo curioso a um engenheiro ou membro de banca — entender **o problema elétrico no mundo real**, **cada técnica de Ciência de Dados e Machine Learning usada**, **exatamente em qual arquivo do repositório está cada peça**, **como as ferramentas se combinam**, **as melhores descobertas (takes) do projeto** e **os usos mais criativos dessa arquitetura**.

---

## 1. O Problema no Mundo Real (Explicado sem Jargão)

### O que é um Medidor Inteligente (AMI)?
Imagine a diferença entre tirar uma **foto por mês** do hidrômetro da sua casa e ter um **relógio cardíaco** medindo seu pulso a cada 5 ou 10 minutos, 24 horas por dia.
- **Medição convencional:** um leiturista passa 1 vez por mês e anota um único número acumulado (kWh do mês). Se algo estranho aconteceu no dia 3 ou se o consumo caiu pela metade no meio do mês, você só descobre semanas depois — e sem saber *em que hora do dia* mudou.
- **AMI (*Advanced Metering Infrastructure*):** medidores inteligentes registram energia, tensão e corrente em intervalos curtos (5, 10 ou 15 minutos) e transmitem esses pacotes por rádio (**rede RF Mesh**) até concentradores e daí para os bancos da distribuidora (**MDM / ORCA**).

### Glossário Essencial do Setor Elétrico (Tradução para Leigos)
| Termo | O que significa na prática | Analogia simples |
|---|---|---|
| **UC** (Unidade Consumidora) | O "CPF da conta de luz" de um imóvel/cliente. | Endereço de entrega. |
| **NIO** (Número de Identificação do Objeto) | O número de série físico do medidor instalado na parede. Uma UC pode trocar de NIO ao longo do tempo! | O número de série do aparelho celular (você troca de aparelho, mas o número da linha continua o mesmo). |
| **FA** (*Forward Active*, kWh) | Energia ativa que a rede **entrega** ao cliente (consumo normal). | Água entrando pelo cano da rua. |
| **RA** (*Reverse Active*, kWh) | Energia ativa que o cliente **injeta de volta** na rede (ex.: painéis solares / Geração Distribuída). | Água sendo bombeada de volta para a rua. |
| **PNT** (Perdas Não Técnicas) | Energia consumida mas não faturada (fraude/desvio, erro cadastral, falha de medição/medidor travado). | Vazamento no cano *antes* do relógio, ou relógio emperrado. |
| **GD** (Geração Distribuída) | Microgeração própria (quase sempre solar fotovoltaica). Quando não homologada no cadastro comercial, aparece como fluxo reverso inesperado. | Cliente gerando energia própria sem avisar a concessionária. |
| **Alimentador / Transformador (Posto)** | A "árvore" elétrica que leva energia da subestação até a rua e às casas. | Avenida principal (alimentador) e rua/esquina (transformador). |
| **PRODIST (Módulo 8)** | Norma da ANEEL que define limites aceitáveis de qualidade de tensão (afundamento, elevação, desequilíbrio entre fases). | Código de trânsito da qualidade de energia. |
| **CIS / GEO / ORCA** | Os 3 mundos de dados da distribuidora: **CIS** = cadastro comercial; **GEO** = mapa elétrico (alimentadores/postes); **ORCA/MDM** = telemetria pesada dos medidores. | Cadastro de clientes, mapa das ruas e central de sensores. |
| **TOI / OS** | Termo de Ocorrência e Inspeção / Ordem de Serviço em campo. **Importante:** este TCC **não** teve acesso a vistorias de campo; o sistema gera *hipóteses priorizadas* para análise humana, nunca "prova de fraude". | Triagem de exames laboratoriais antes do diagnóstico médico presencial. |

### Por que detectar anomalias em AMI é difícil?
1. **Sinal ruim parece fraude:** se o rádio RF Mesh falhar e perder 50% dos pacotes, a soma diária de energia cai pela metade — um algoritmo ingênuo gritaria "desvio!", quando na verdade é só falta de sinal de rádio.
2. **Troca de medidor mistura histórias:** se a UC trocou o medidor NIO-A pelo NIO-B no meio do mês, olhar só pelo NIO quebra o histórico.
3. **Cada cliente tem um perfil:** 80 kWh/dia é normal para uma padaria e absurdo para uma kitnet.
4. **Capacidade de campo é limitada:** de nada adianta apontar 28.000 "suspeitos" se a distribuidora consegue inspecionar no máximo ~2.000 imóveis por mês. É preciso **separar filas** (Telecom, Metrologia, GD, PNT) e **ordenar por prioridade**.

---

## 2. Mapa Completo do Repositório — Onde Está Cada Coisa

### A. Núcleo de Dados, Ingestão e Limpeza (`src/`)
| Arquivo | O que faz | Técnica / Regra principal |
|---|---|---|
| [src/db.py](file:///d:/Projects/tcc_extraction/src/db.py) | Conexão segura com o banco Oracle/ORCA. | SQLAlchemy `QueuePool(pool_size=5, max_overflow=0)` + fechamento explícito de cursores (corrigiu vazamento de 142 sessões abertas). |
| [src/datasets/normalize.py](file:///d:/Projects/tcc_extraction/src/datasets/normalize.py) | Normaliza tabelas brutas para `ami_interval`, `ami_instantaneous`, `ami_registers`. | Detecção automática de cadência (5, 10 ou 15 min) e cálculo exato de cobertura por denominador nativo. |
| [src/datasets/retroactive_cleaner.py](file:///d:/Projects/tcc_extraction/src/datasets/retroactive_cleaner.py) | **Coração da engenharia de dados:** reconstrói o vínculo histórico `UC ↔ NIO` por vigência temporal e gera features diárias e de janela (30 dias) sem vazamento futuro. | Polars vetorizado + saneamento retroativo + flags `meta__is_clean_for_train` e `meta__data_quality`. |
| [src/datasets/build_training_dataset.py](file:///d:/Projects/tcc_extraction/src/datasets/build_training_dataset.py) | Junta features de janela com cadastro CIS (`uc_context`) e topologia GEO (`electrical_hierarchy`). | Separação estrita de prefixos (`id__*`, `meta__*`, `x__*`, `y__*`) para impedir vazamento de variáveis administrativas no ML. |
| [src/features/meter_day.py](file:///d:/Projects/tcc_extraction/src/features/meter_day.py) & [src/features/electrical.py](file:///d:/Projects/tcc_extraction/src/features/electrical.py) | Calcula métricas físicas diárias por medidor. | FA/RA sum, fator de carga, taxa de reversão, desequilíbrio de tensão e corrente sob carga (`>= 1 A`), rampas, autocorrelação lag-1, hora de pico cíclica (`sin/cos`). |
| [src/features/uc_window.py](file:///d:/Projects/tcc_extraction/src/features/uc_window.py) | Agrega 30 dias em estatísticas robustas até a data de corte (*cutoff*). | Mediana, MAD (*Median Absolute Deviation*), slope linear OLS, dias confiáveis, idade do medidor, troca em 30/90d. |
| [src/features/neighborhood.py](file:///d:/Projects/tcc_extraction/src/features/neighborhood.py) & [src/features/circuit_profile.py](file:///d:/Projects/tcc_extraction/src/features/circuit_profile.py) | Calcula contraste espacial entre vizinhos do mesmo transformador/alimentador e perfil do circuito (ex.: penetração de GD). | Z-score dentro do transformador e mediana de cobertura dos pares. |
| [src/events/alarm_extractor.py](file:///d:/Projects/tcc_extraction/src/events/alarm_extractor.py) & [src/events/alarm_categories.py](file:///d:/Projects/tcc_extraction/src/events/alarm_categories.py) | Extrai e categoriza alarmes do medidor (tampa aberta, falta de energia, comunicação offline, corrente reversa). | Classificação por regex/padrões sobre `CONTENT` em partições Parquet (`output/events/alarm_events/`). |

### B. Modelos, Especialistas e Regras de Negócio (`src/models/` e `src/eda/`)
| Arquivo | O que faz | Técnica principal |
|---|---|---|
| [src/models/composite.py](file:///d:/Projects/tcc_extraction/src/models/composite.py) | **Comitê Elétrico de 6 Detectores (`CompositeAnomalyDetector`)** + isolamento estrito de features elétricas puras (`select_pure_electrical_features`). | IForest + LOF (`log1p`) + ECOD + COPOD + HBOS + RobustPCA (`StandardScaler`) + normalização congelada `frozen_quantile` (`P01–P99`) + guardrails PRODIST (`strict_voltage_imbalance_rule` na `v2`). |
| [src/models/specialized_detectors.py](file:///d:/Projects/tcc_extraction/src/models/specialized_detectors.py) | **Orquestrador Multivisão (`MultiViewAnomalyPipeline`):** separa `S_comm`, `S_elec`, `S_tamper` e `BehavioralClusterer`. | Gate de qualidade (`cov >= 0,70`, e `reliable_days >= 18` na `v2`) que impede telemetria ruim de virar anomalia elétrica. |
| [src/eda/uc_timeseries.py](file:///d:/Projects/tcc_extraction/src/eda/uc_timeseries.py) | Séries temporais diárias/horárias e **Detector de Mudança de Regime (`_detect_step_change_cusum`)**. | CUSUM (soma acumulada centrada) + teste de magnitude relativa à escala robusta MAD para localizar exatamente o dia de um degrau de consumo. |
| [src/models/moe_router.py](file:///d:/Projects/tcc_extraction/src/models/moe_router.py) & [src/models/hierarchical_trainer.py](file:///d:/Projects/tcc_extraction/src/models/hierarchical_trainer.py) | **Mixture of Experts (MoE) Hierárquico:** pondera especialista do alimentador local vs. modelo global conforme volume de amostras e penetração solar. | Encolhimento empírico (*shrinkage*): circuitos grandes confiam 70–88% no modelo local; circuitos pequenos apoiam-se no modelo global. |
| [src/models/spurious_filter.py](file:///d:/Projects/tcc_extraction/src/models/spurious_filter.py) | Filtra perturbações coletivas de rede (ex.: desligamento/manobra no transformador) e ruídos não isolados. | Se `>= 60%` dos vizinhos no mesmo transformador ficaram anômalos simultaneamente → reclassifica como `GRID_DISTURBANCE` e atenua o escore em 65%. |
| [src/models/category_reporting.py](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py) | **Árvore de Triagem Delimitada (9 categorias não sobrepostas)** + contraste espacial Telecom (Posto → grade 250m → Alimentador) + transições de classe comportamental (`C1..C6`). | Separa fisicamente filas de Telecom, Ciclo de Vida (UC Nova / Troca), GD não cadastrada, Medidor Travado e Anomalia Elétrica/PNT. |
| [src/utils/censorship.py](file:///d:/Projects/tcc_extraction/src/utils/censorship.py) | Blindagem LGPD automática de UCs (`908***00`), NIOs (`NIO-435***63`) e alimentadores (`Circuito F-01..F-23`). | Mascaramento determinístico sem expor clientes ou nomes comerciais em figuras e relatórios. |

### C. Experimentos, Interfaces e Testes (`scripts/`, `src/experiments/`, `tests/`)
| Arquivo | O que faz |
|---|---|
| [src/experiments/oot_pipeline.py](file:///d:/Projects/tcc_extraction/src/experiments/oot_pipeline.py) & [src/experiments/daily_rules.py](file:///d:/Projects/tcc_extraction/src/experiments/daily_rules.py) | Runners oficiais da campanha congelada de 25/11 (`OOT_MULTIVIEW`, `DAILY_RULES`, `ABLATION_M0_M5`, `SCORE_STABILITY`). |
| [scripts/audit_experimental_results.py](file:///d:/Projects/tcc_extraction/scripts/audit_experimental_results.py) | **Auditoria P0:** reconciliação populacional, auditoria de correlações, gating congelado (sem *refit*), sensibilidade de pesos (23 configs), sensibilidade do bônus `+0,15` e matriz de especialistas. |
| [scripts/run_multiview_evaluation.py](file:///d:/Projects/tcc_extraction/scripts/run_multiview_evaluation.py) | **Avaliação Multivisão P1 (Base A):** executa todos os especialistas sobre as 65.686 UCs OOT (incluindo 16 modelos MoE por alimentador sem vazamento e CPD em lote), ablação `V0–V5` sob `K = 2.000`, estudos de caso e projeção operacional. |
| [scripts/run_controlled_perturbations.py](file:///d:/Projects/tcc_extraction/scripts/run_controlled_perturbations.py) | **Perturbações Controladas P1 (Base B):** injeta 23 cenários fisicamente coerentes em cópias isoladas de 100 séries AMI reais (433 mil medições) antes da extração de features para medir detecção, erro temporal CPD e isolamento de eventos coletivos. |
| [scripts/run_v2_temporal_validation.py](file:///d:/Projects/tcc_extraction/scripts/run_v2_temporal_validation.py) | **Validação Temporal Multi-Corte (`19/11 → 25/11`) e Variante `v2` Saneada:** testa persistência real de +6 dias, ativa 1,7M de alarmes reais no `S_tamper` e prova que o gate `v2` zera vazamentos de blocos consecutivos ausentes. |
| [src/tui/tui_app.py](file:///d:/Projects/tcc_extraction/src/tui/tui_app.py) & [scripts/run_web_cockpit.py](file:///d:/Projects/tcc_extraction/scripts/run_web_cockpit.py) | **Interfaces operacionais:** Terminal UI (Textual/Rich) para governança/extração/auditoria e Cockpit Web para raio-X individual de UCs, séries temporais e filas priorizadas. |
| [tests/test_p0_p1_experimental_closure.py](file:///d:/Projects/tcc_extraction/tests/test_p0_p1_experimental_closure.py) (e 44 outros arquivos em `tests/`) | Garante matematicamente zero vazamento temporal, soma unitária dos pesos, conservação de partições, anonimização LGPD e invariantes das figuras/relatórios (**67/67 testes nucleares aprovados**). |

---

## 3. Como Cada Técnica de ML Funciona (Explicação Intuitiva)

### Por que usar 6 detectores elétricos juntos em vez de 1?
Imagine 6 inspetores com "lentes" completamente diferentes olhando a mesma folha de dados elétricos (16 variáveis puras, sem nenhuma variável de comunicação):

1. **Isolation Forest (`IF`, peso 0,22):** *O cortador aleatório.* Tenta isolar cada cliente fazendo cortes aleatórios nos dados. Quem fica sozinho em poucos cortes é um ponto extremo global. **Ponto fraco:** sofre de *masking* quando vários fraudadores/anômalos formam um pequeno grupinho parecido.
2. **Local Outlier Factor (`LOF`, peso 0,20):** *O comparador de vizinhos.* Pergunta: "A densidade ao redor desta UC é muito menor que a dos seus 20 vizinhos mais parecidos?" **Superpoder:** encontra anomalias locais e microgrupos que o IF deixa passar. *(Por isso removê-lo trocou 180 vagas no Top-2.000!)*
3. **ECOD (peso 0,18):** *O fiscal das pontas.* Sem assumir distribuição normal, olha em qual percentil extremo (cauda esquerda ou direita) cada variável do cliente caiu.
4. **COPOD (peso 0,18):** *O fiscal das pontas assimétricas.* Usa cópulas empíricas corrigidas por assimetria (*skewness*) — perfeito para energia elétrica (que tem cauda longa à direita) e desequilíbrios.
5. **HBOS (peso 0,10):** *O contador de histogramas.* Ultrarrápido; pontua quem cai em faixas muito raras (ex.: consumo quase zero com tensão normal).
6. **Robust PCA (peso 0,12):** *O detector de quebra de correlação.* Aprende as relações lineares normais entre variáveis (ex.: alta energia anda junto com certa corrente e demanda). Se alguém "quebra a física" dessas correlações, o erro de reconstrução dispara.

### Por que a Normalização Congelada (`frozen_quantile`) é crucial?
Se você normalizar um lote usando `min` e `max` do próprio dia (`batch_minmax`), basta **1 único medidor defeituoso** com valor absurdo para esmagar o escore de todas as outras 65.685 UCs!  
Com `frozen_quantile`, fixamos os limites `P01` e `P99` **no período de treino (Ago–Out)**. Nos testes de estabilidade (`score_stability`), remover o Top-1% mais extremo causou deslocamento de até **0,2291** no `batch_minmax`, contra **0,000000** (estabilidade absoluta!) no `frozen_quantile`.

### Como os Especialistas se Complementam?
- **`S_comm` + Gate de Qualidade:** "Os dados são confiáveis para julgar física?" Se não → **Quarentena Telecom** (e ainda compara com os vizinhos do transformador/250m para dizer se o defeito é só na antena daquela casa ou no bairro inteiro!).
- **`S_elec` (Transversal):** "Hoje, olhando os últimos 30 dias, esta UC parece esquisita comparada às outras 65 mil?"
- **`CPD / CUSUM` (Longitudinal):** "Esta UC *mudou bruscamente* em relação a **si mesma** em algum dia específico?" *(No experimento controlado, localizou quedas de -50% com erro mediano de **0,0 dia**, enquanto `S_elec` sozinho mal via quedas de -20% a -30%!)*
- **`S_behavior` (Intradiário 24h):** "O consumo total diário não mudou nada, mas a rotina de horários mudou?" *(Detectou deslocamento de +4h mantendo ΔkWh/dia = 0,0!)*
- **`S_tamper` (Eventos Físicos):** "O próprio medidor gritou alarme de tampa aberta, remoção de módulo ou corrente reversa?"
- **`MoE` + `SpuriousFilter` (Topologia):** "Isso é uma anomalia desta casa ou o transformador inteiro sofreu uma queda junto?"

---

## 4. Combinações de Ferramentas (As "Receitas" que Fazem Tudo Rodar Rápido)

| Objetivo | Combinação de Ferramentas | Por que funciona tão bem |
|---|---|---|
| **Processar 117 dias × 8,8 milhões de linhas/dia em um notebook** | **Polars (LazyFrame + predicate pushdown) + Parquet particionado (`year/month/day`)** | Carregar 100 UCs ao longo de 30 dias (433 mil linhas de um universo de ~260 milhões) levou **3,0 segundos** sem estourar memória! |
| **Extrair do Oracle (ORCA) sem derrubar o banco** | **SQLAlchemy `QueuePool(size=5, max_overflow=0)` + `ThreadPoolExecutor(5)` + `result.close()`** | Mantém paralelismo real de 5 consultas reutilizando exatamente 5 sessões em vez de vazar 142 cursores abertos. |
| **Persistir modelos pesados sem quebrar o Git/disco** | **Joblib + `joblib_splitter.py` (chunks de 25 MB comprimidos)** | Permite versionar, carregar e congelar o `MultiViewAnomalyPipeline` inteiro (`frozen_multiview_model.joblib`) com integridade SHA-256. |
| **Avaliar algoritmos sem depender de vistorias de campo** | **Injeção controlada de perturbações sobre cópias em memória de séries reais + `pytest` de invariantes** | Mede observabilidade, atraso temporal e vazamento de lacunas sem tocar 1 byte do Data Lake e sem criar curvas sintéticas artificiais. |
| **Operação humana segura (LGPD)** | **Cockpit Web + TUI Rich + `censorship.py` + contrato Parquet com `NULL` explícito** | O analista vê o raio-X completo (curva 5 min, CUSUM, radar de detectores, vizinhança) sem expor dados sensíveis em relatórios externos. |

---

## 5. Engenharia Criativa e Inovações no Código: O que Nosso Código Faz Diferente do Padrão de Mercado

Muitos projetos acadêmicos e comerciais de detecção de anomalias em energia elétrica seguem uma receita genérica: jogam uma planilha pronta no `IsolationForest` do scikit-learn, pegam o Top 5% e encerram. O nosso código nasceu enfrentando **falhas reais de campo em 65 mil medidores inteligentes** — e foi exatamente resolvendo esses choques com a realidade física que criamos soluções fora do padrão.

### Como normalmente se faz por aí vs. O que contribuímos de especial

| Problema Real em AMI | Como normalmente se faz por aí (padrão comum) | O que construímos de especial neste repositório |
|---|---|---|
| **Mistura de sinal de rádio com física elétrica** | Joga todas as colunas (consumo + cobertura + alarmes) no mesmo modelo; falta de sinal vira "suspeita de fraude". | **Firewall programático de subespaço** (`select_pure_electrical_features`) + **Gate de qualidade de duplo critério** (`cov >= 70%` E `dias_confiaveis >= 18`). |
| **Escala dos scores mudando todo dia** | Normaliza com `MinMaxScaler` no lote atual (`batch_minmax`); 1 medidor quebrado distorce o ranking da cidade inteira. | **Normalização quantílica congelada** (`P01–P99` do treino com cascata de fallback), alcançando **0,000000** de drift sob perturbação de outliers. |
| **Bibliotecas prontas quebrando em dados reais** | Usa `LOF` e `PCA` padrão "out-of-the-box", sem perceber que empates exatos explodem LOF para `1e10` e `kVA` domina a PCA. | **Compressão `log1p` anti-colapso no LOF** + **`StandardScaler` interno na `RobustPCADetector`**, restaurando a complementaridade real do comitê. |
| **Medidores com cadências e fases diferentes** | Assume grade fixa de 288 pontos/dia (5 min) e calcula desequilíbrio de corrente mesmo em madrugada (0,1 A) ou fase inexistente. | **Denominador dinâmico por cadência nativa** (`1440 / cadência`) + **detecção automática de fases ativas** e desequilíbrio restrito a regime sob carga (`I >= 1,0 A`). |
| **Telecom ruim: problema na casa ou no bairro?** | Trata toda baixa cobertura igual (ou ignora o espaço geográfico). | **Contraste espacial em 3 camadas vetorizado em Polars** (`Transformador → Grade 250 m → Alimentador`) sem nenhum loop Python. |
| **Redes heterogêneas e eventos coletivos** | Um modelo único para a cidade inteira; uma manobra no transformador gera dezenas de falsos positivos simultâneos. | **Roteador MoE com encolhimento empírico + boost solar (> 5% GD)** acoplado ao **Filtro de Perturbação Coletiva (`>= 60%` do trafo)**. |
| **Quedas graduais e mudanças de hábito** | Olha só a média de 30 dias; perde quedas de -20% a -30% ou trocas de horário com mesmo kWh/dia. | **CUSUM robusto mascarado por cobertura e escalado por MAD** (erro mediano de **0,0 dia**) + **especialista intradiário 24h**. |
| **Validação sem vistorias de campo (TOI)** | Cria curvas senoidais 100% artificiais ou só apresenta listas sem saber se o modelo detectaria falhas conhecidas. | **Motor de perturbações fisicamente coerentes em memória sobre séries reais** (Lei de Ohm, rotação circular de rotina, sino solar), sem tocar 1 byte no Data Lake. |

---

### 10 Trechos Reais do Nosso Código que Fazem a Diferença

#### 1. Firewall Programático de Subespaço Elétrico
📍 [src/models/composite.py](file:///d:/Projects/tcc_extraction/src/models/composite.py#L39-L71)

Em vez de confiar que ninguém esquecerá de remover colunas de comunicação manualmente, o código impõe uma **barreira estrutural**: qualquer coluna que contenha tokens de cobertura, latência, alarme ou ciclo de vida do hardware é **proibida** de entrar no comitê elétrico.

```python
NON_ELECTRICAL_FEATURE_TOKENS: tuple[str, ...] = (
    "coverage",
    "reliable_day",
    "null_ratio",
    "latency",
    "alarm",
    "meter_age",
    "meter_changed",
    "n_meters",
    "peer_median_coverage",
    "coverage_deficit",
)

def select_pure_electrical_features(df_or_cols: pl.DataFrame | Sequence[str]) -> list[str]:
    if isinstance(df_or_cols, pl.DataFrame):
        candidate_cols = [
            c for c in df_or_cols.columns
            if c.startswith("x__") and df_or_cols[c].dtype.is_numeric()
        ]
    else:
        candidate_cols = [c for c in df_or_cols if c.startswith("x__")]

    pure_elec = [
        c for c in candidate_cols
        if not any(tok in c.lower() for tok in NON_ELECTRICAL_FEATURE_TOKENS)
    ]
    return pure_elec if pure_elec else list(candidate_cols)
```
> **Por que é especial:** garante por construção de software que o escore elétrico (`S_elec`) enxerga *exclusivamente* física de carga, tensão e corrente — impedindo que perda de sinal RF Mesh contamine o aprendizado não supervisionado.

---

#### 2. Normalização Quantílica Congelada com Cascata de Fallback
📍 [src/models/composite.py](file:///d:/Projects/tcc_extraction/src/models/composite.py#L86-L146)

Bibliotecas comuns normalizam cada lote usando o mínimo e máximo daquele dia. Aqui congelamos `P01`, `P99` e `P99.9` **no período de treino** e criamos uma cascata de segurança para distribuições muito concentradas (quando `P01 == P99`), evitando divisão por zero sem perder a âncora temporal.

```python
def _normalize_with_params(
    raw_scores: np.ndarray,
    params: dict[str, Any] | None,
    mode: str = "frozen_quantile",
) -> np.ndarray:
    s = np.nan_to_num(np.asarray(raw_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(s) == 0:
        return s
    if params is None or mode == "batch_minmax":
        return _min_max_normalize(s)

    if mode == "frozen_ecdf":
        sorted_ref = params["sorted_raw"]
        n_ref = max(1, len(sorted_ref))
        ranks = np.searchsorted(sorted_ref, s, side="right")
        return np.clip(ranks.astype(float) / float(n_ref), 0.0, 1.0)

    # Default: "frozen_quantile" (anchored at training P01 and P99, fallback to P99.9 then min/max)
    lo = float(params["p01"])
    hi = float(params["p99"])
    if hi - lo < 1e-12:
        hi = float(params.get("p999", params["max"]))
    if hi - lo < 1e-12:
        lo = float(params["min"])
        hi = float(params["max"])
    if hi - lo < 1e-12:
        return np.zeros_like(s, dtype=float)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)
```
> **Resultado medido:** quando removemos o Top-1% mais extremo da população, o `batch_minmax` sofre deslocamento de até **0,2291**, enquanto nosso `frozen_quantile` tem deslocamento **0,000000** (invariância exata).

---

#### 3. Resgate Matemático do LOF (`log1p`) e Padronização Interna da RobustPCA
📍 [src/models/composite.py](file:///d:/Projects/tcc_extraction/src/models/composite.py#L302-L323) e [L403-L411](file:///d:/Projects/tcc_extraction/src/models/composite.py#L403-L411)

Em parques AMI reais há centenas de imóveis desocupados com consumo exatamente igual a zero. No `LocalOutlierFactor` padrão, vizinhos idênticos geram `k-distance = 0`, fazendo o escore explodir para `~10¹⁰` e esmagar todos os demais pontos na normalização. Na `PCA`, variáveis de grande magnitude (`kVA`, `kWh`) dominam completamente o erro de reconstrução se não houver padronização interna.

```python
# Compressão logarítmica anti-colapso no LOF (src/models/composite.py, L403-L411)
return {
    "iso_forest": -self.iso_forest.score_samples(X_scaled),
    # Log-compress raw LOF to prevent k-distance=0 tie spikes (1e10) from crushing P01-P99
    "lof": np.log1p(np.maximum(-self.lof.score_samples(X_scaled) - 1.0, 0.0)),
    "ecod": self.ecod.score_samples(X_scaled),
    "copod": copod_det.score_samples(X_scaled),
    "hbos": hbos_det.score_samples(X_scaled),
    "pca": self.pca.score_samples(X_scaled),
}
```
```python
# Padronização interna na RobustPCA (src/models/composite.py, L309-L323)
self.scaler_ = StandardScaler()
X_std = self.scaler_.fit_transform(X)
n_comp = max(1, min(n_samples - 1, n_features - 1, int(np.ceil(n_features * 0.6))))
self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
self.pca_.fit(X_std)
...
X_eval = self.scaler_.transform(X) if self.scaler_ is not None else X
X_proj = self.pca_.transform(X_eval)
X_rec = self.pca_.inverse_transform(X_proj)
return np.mean((X_eval - X_rec) ** 2, axis=1)
```
> **Por que é especial:** essas poucas linhas transformaram dois detectores que estavam praticamente mortos (`ρ = 0,02` no LOF e `ρ = -0,20` na PCA) em pilares complementares do comitê (`ρ = +0,46` e `+0,41`).

---

#### 4. Saneamento Físico de Corrente Sob Carga (`>= 1,0 A`) e Cadência Dinâmica Nativa
📍 [src/datasets/retroactive_cleaner.py](file:///d:/Projects/tcc_extraction/src/datasets/retroactive_cleaner.py#L320-L366) e [L493](file:///d:/Projects/tcc_extraction/src/datasets/retroactive_cleaner.py#L493)

Na madrugada, uma geladeira desligando gera correntes minúsculas (`0,05 A` vs `0,15 A`) que parecem "100% de desequilíbrio" se você calcular a fórmula crua. Além disso, um medidor bifásico tem a 3ª fase permanentemente zerada. Nosso limpador descobre automaticamente quais fases existem no medidor (`max > 0,1 A`), filtra apenas momentos sob carga real (`I_AVG >= 1,0 A`) e usa o percentil `P90` (não o máximo absoluto). E para cobertura, divide pela cadência nativa do próprio medidor:

```python
# Identifica fases fisicamente ativas no dia (> 0.1 A) e calcula desequilíbrio só sob carga >= 1.0 A
phase_activity = work.group_by("_NORM_NIO").agg([
    (pl.col(inm).max().fill_null(0.0) > 0.1).alias(f"_ACT_{idx}")
    for idx, inm in enumerate(i_names)
])
...
work = work.with_columns([
    k_valid.alias("_K_VALID"),
    pl.when(k_valid >= 2).then(sum_i / k_valid.cast(pl.Float64)).otherwise(0.0).alias("_I_AVG"),
]).filter((pl.col("_K_VALID") >= 2) & (pl.col("_I_AVG") >= 1.0))

imb_i = (pl.max_horizontal(dev_i) / pl.col("_I_AVG")).clip(0.0, 2.0)
return (
    work.with_columns(imb_i.alias("_C_IMB"))
    .group_by("_NORM_NIO")
    .agg(pl.col("_C_IMB").quantile(0.90).round(4).alias("CURRENT_IMBALANCE_MAX"))
)
```
```python
# Cobertura pela cadência nativa do medidor (5 min -> 288, 10 min -> 144, 15 min -> 96)
((pl.col("_VALID_FA").cast(pl.Float64) / (1440.0 / pl.col("_CADENCE"))).clip(0.0, 1.0))
    .round(4).alias("INTERVAL_COVERAGE")
```
> **Por que é especial:** é Engenharia Elétrica pura traduzida em Polars vetorizado — resolve de uma vez só o ruído noturno, a heterogeneidade mono/bi/trifásica e o artefato da UC `685***20` (50% vs 99,49%).

---

#### 5. Gate de Qualidade de Duplo Critério e Guardrail PRODIST `v2`
📍 [src/models/specialized_detectors.py](file:///d:/Projects/tcc_extraction/src/models/specialized_detectors.py#L175-L190) e [src/models/composite.py](file:///d:/Projects/tcc_extraction/src/models/composite.py#L554-L562)

Nosso experimento de perturbações descobriu uma brecha sutil que quase ninguém vê: um medidor que fica **15 dias completamente mudo** ainda tem média de 99% nos 15 dias em que transmitiu! O gate duplo da `v2` exige **cobertura média suficiente E número mínimo de dias confiáveis no mês**:

```python
# Gate duplo v2: bloqueia tanto perda aleatória quanto apagão consecutivo de dias
dq_arr = np.asarray(data_quality, dtype=float)
is_reliable = dq_arr >= self.min_quality_threshold
min_rel_d = getattr(self, "min_reliable_days", None)
if min_rel_d is not None:
    if reliable_days is not None:
        rd_arr = np.asarray(reliable_days, dtype=float)
        is_reliable = is_reliable & (rd_arr >= float(min_rel_d))
    elif raw_dataframe is not None and "x__coverage_reliable_days" in raw_dataframe.columns:
        rd_arr = raw_dataframe["x__coverage_reliable_days"].fill_null(0).to_numpy()
        is_reliable = is_reliable & (rd_arr >= float(min_rel_d))

res = self.detector.predict_composite(X, raw_dataframe=raw_dataframe)
# Apenas observações confiáveis podem virar candidatas a anomalia elétrica
is_anomaly = res["is_anomaly"] & is_reliable
```
```python
# Correção v2 do bônus PRODIST (+0.15): restringe estritamente a desequilíbrio de TENSÃO
strict_v_rule = bool(getattr(self, "strict_voltage_imbalance_rule", False))
if strict_v_rule:
    v_imb_cols = [c for c in raw_dataframe.columns if "VOLTAGE_IMBALANCE" in c.upper()]
else:
    v_imb_cols = [c for c in raw_dataframe.columns if "VOLTAGE_IMBALANCE" in c.upper() or "IMBALANCE_MAX" in c.upper()]
```
> **Resultado medido:** o gate duplo derrubou o vazamento de apagões de 15 dias de **31% para 0,0%**, e o parâmetro `strict_voltage_imbalance_rule` reduziu disparos indevidos do bônus `+0,15` de 46,7% para **0,19%** da cidade, mantendo compatibilidade retroativa com a `v1`.

---

#### 6. Contraste Espacial em 3 Camadas para Telecom (Sem Loop Python)
📍 [src/models/category_reporting.py](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py#L246-L322)

Quando uma UC perde sinal, como saber se o defeito é no medidor daquela casa ou no repetidor RF Mesh do bairro? Construímos uma **cascata espacial hierárquica** em expressões de janela Polars (`.over(...)`): primeiro compara com o mesmo **Transformador** (`>= 3` vizinhos); se faltar cadastro de posto, quantiza latitude/longitude em uma **grade espacial de ~250 metros (`0,0025°`)**; por fim, usa o **Alimentador**.

```python
# Grade geográfica ~250m (0.0025 deg ~ 250m) criada direto em Polars
work = work.with_columns(
    pl.when(pl.col("meta__latitude").is_not_null() & pl.col("meta__longitude").is_not_null())
    .then(
        (pl.col("meta__latitude") / 0.0025).round(0).cast(pl.Int32).cast(pl.String)
        + pl.lit("_")
        + (pl.col("meta__longitude") / 0.0025).round(0).cast(pl.Int32).cast(pl.String)
    )
    .otherwise(None)
    .alias("_SPATIAL_GRID_250M")
)

# Cascata Posto (>= 3 pares) -> Grade 250m (>= 3 pares) -> Alimentador
use_posto = pl.col("meta__posto").is_not_null() & (posto_cnt >= 3) if has_posto else pl.lit(False)
use_grid = pl.col("_SPATIAL_GRID_250M").is_not_null() & (grid_cnt >= 3)

work = work.with_columns([
    pl.when(use_posto).then(posto_med)
      .when(use_grid).then(grid_med)
      .otherwise(feeder_med)
      .fill_null(0.95).round(4).alias("meta__peer_median_coverage"),
    pl.when(use_posto).then(pl.lit("POSTO_TRANSFORMADOR"))
      .when(use_grid).then(pl.lit("VIZINHANCA_GEO_250M"))
      .otherwise(pl.lit("ALIMENTADOR"))
      .alias("meta__peer_proximity_basis"),
]).with_columns(
    (pl.col("meta__peer_median_coverage") - cov_col)
    .clip(0.0, 1.0).round(4).alias("meta__coverage_deficit_vs_peers")
)
```
> **Por que é especial:** roda em milissegundos sobre 65.686 UCs, resiste a falhas de cadastro topológico usando coordenadas GPS em grade de 250 m e entrega a justificativa auditável (`meta__peer_proximity_basis`) para cada despacho.

---

#### 7. Roteador MoE com Encolhimento Empírico + Boost Solar e Filtro Coletivo de Transformador
📍 [src/models/moe_router.py](file:///d:/Projects/tcc_extraction/src/models/moe_router.py#L48-L81) e [src/models/spurious_filter.py](file:///d:/Projects/tcc_extraction/src/models/spurious_filter.py#L94-L119)

Em vez de escolher entre "1 modelo global" (que ignora particularidades de circuitos rurais ou com muita energia solar) ou "1 modelo por alimentador" (que sofre *overfitting* em circuitos pequenos), implementamos **encolhimento empírico contínuo** com reforço físico quando a penetração solar passa de 5%. Depois, o `SpuriousFilter` bloqueia falsos positivos coletivos no transformador:

```python
# Encolhimento empírico por densidade de amostras + reforço de física local para circuitos solares
if feeder_sample_count >= self.large_feeder_samples:
    w_feeder, w_mun, w_global = 0.70, 0.20, 0.10
elif feeder_sample_count >= self.min_feeder_samples:
    ratio = (feeder_sample_count - self.min_feeder_samples) / (
        self.large_feeder_samples - self.min_feeder_samples
    )
    w_feeder = 0.35 + 0.35 * ratio
    w_mun = 0.40 - 0.20 * ratio
    w_global = 0.25 - 0.15 * ratio
else:
    w_feeder, w_mun, w_global = 0.15, 0.45, 0.40

if circuit_profile:
    gd_pen = float(circuit_profile.get("CIRCUIT_SOLAR_GD_PENETRATION", 0.0) or 0.0)
    if gd_pen > 0.05 and feeder_sample_count >= self.min_feeder_samples:
        w_feeder = min(0.85, w_feeder + 0.10)
        w_global = max(0.05, w_global - 0.05)
        w_mun = max(0.05, w_mun - 0.05)
```
```python
# Filtro de evento coletivo no transformador (src/models/spurious_filter.py, L98-L118)
trafos = result_indexed.partition_by(part_keys, as_dict=True)
for _, t_df in trafos.items():
    if t_df.height >= 3 and t_df["TRAFO_ID"][0] is not None:
        anom_rate = float(t_df.get_column("is_anomaly").mean())
        if anom_rate >= grid_event_threshold:  # padrão: 0.60 (60% dos vizinhos)
            idx_arr = t_df.get_column("_ROW_IDX").to_numpy()
            is_grid_event[idx_arr] = True

filtered_scores = np.where(base_is_anom & is_grid_event, base_scores * 0.35, base_scores)
final_is_anomaly = base_is_anom & (~is_grid_event) & (~is_spurious)
```
> **Resultado medido:** no experimento controlado, separou **16 de 16 (100%)** eventos coletivos de rede contra anomalias isoladas, atenuando o escore em 65% e reclassificando para `GRID_DISTURBANCE`.

---

#### 8. Detector de Mudança de Regime CUSUM Robusto com Escala MAD
📍 [src/eda/uc_timeseries.py](file:///d:/Projects/tcc_extraction/src/eda/uc_timeseries.py#L185-L230)

Um detector de mudança de patamar ingênuo dispara sempre que há 1 dia com falta de sinal ou feriado. Nossa implementação (1) descarta dias com cobertura `< 0,70`, (2) protege as bordas da janela (`margin = max(2, len(y) // 6)`), e (3) exige que o salto entre medianas supere **ao mesmo tempo** a variabilidade natural do cliente (`1,25 * robust_scale`) e um piso relativo (`25%` do patamar anterior):

```python
reliable_idx = [
    i for i, (v, c) in enumerate(zip(values, coverages))
    if v is not None and not math.isnan(v) and c >= 0.70
]
if len(reliable_idx) < 8:
    return None

y = np.array([values[i] for i in reliable_idx], dtype=np.float64)
cusum = np.cumsum(y - float(np.mean(y)))
margin = max(2, len(y) // 6)
sub_cusum = np.abs(cusum[margin:-margin])
k = int(np.argmax(sub_cusum)) + margin

pre_med = float(np.median(y[: k + 1]))
post_med = float(np.median(y[k + 1 :]))
delta_abs = post_med - pre_med

# Confirma significância estatística e operacional contra escala robusta (MAD)
if abs(delta_abs) < max(1.25 * robust_scale, 0.25 * max(abs(pre_med), 0.1)):
    return None
```
> **Resultado medido:** alcançou **99% de detecção** em quedas sustentadas de -50% com **erro temporal mediano de exatamente 0,0 dia**, e **0% de falso disparo** em picos isolados de 1 dia!

---

#### 9. Injeção Fisicamente Coerente de Perturbações sobre Séries Reais em Memória
📍 [scripts/run_controlled_perturbations.py](file:///d:/Projects/tcc_extraction/scripts/run_controlled_perturbations.py#L253-L341)

Em vez de inventar dados sintéticos de brinquedo ou alterar o Data Lake, clonamos 100 séries AMI reais (433 mil leituras de 5/10 min) **estritamente em memória** e injetamos fenômenos respeitando a física elétrica **antes** de toda a extração de features:

```python
# 1. Queda sustentada respeitando a Lei de Ohm: reduz FA e correntes I_L1..L3 preservando tensões U_L1..L3
elif scenario_type == "sustained_drop":
    factor = 1.0 - float(intensity)
    cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
    exprs = [pl.when(cond).then(pl.col("FA_INTERVAL") * factor).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
    for ic in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"):
        if ic in df.columns:
            exprs.append(pl.when(cond & pl.col(ic).is_not_null()).then(pl.col(ic) * factor).otherwise(pl.col(ic)).alias(ic))
    df = df.with_columns(exprs)

# 2. Rotação circular de rotina intradiária: muda os horários mantendo kWh diário EXATAMENTE igual
elif scenario_type == "time_shift":
    new_secs = (secs + shift_h * 3600) % 86400
    new_ts = pl.col("REPORT_DAY").cast(pl.Datetime("us")) + pl.duration(seconds=new_secs)
    df = df.with_columns(pl.when(cond).then(new_ts).otherwise(pl.col(ts_col)).alias(ts_col))

# 3. Injeção solar fotovoltaica diurna (sino parabólico 09h-16h com pico às 12h30 + autoconsumo)
elif scenario_type == "reverse_flow_gd":
    solar_bell = (1.0 - ((hr - 12.5) / 4.0) ** 2).clip(0.0, 1.0) * float(intensity)
    df = df.with_columns([
        pl.when(cond).then(pl.col("RA_INTERVAL") + solar_bell).otherwise(pl.col("RA_INTERVAL")).alias("RA_INTERVAL"),
        pl.when(cond).then((pl.col("FA_INTERVAL") - solar_bell * 0.5).clip(lower_bound=0.0)).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL"),
    ])
```
> **Por que é especial:** é um laboratório experimental reprodutível que testa o pipeline *inteiro de ponta a ponta* (do dado bruto 5 min até a fila final) e revelou falhas invisíveis na análise estática — como a brecha dos 15 dias consecutivos sem sinal.

---

#### 10. Engenharia de Produção: Pool Oracle Limitado + Versionamento Binário SHA-256
📍 [src/db.py](file:///d:/Projects/tcc_extraction/src/db.py#L122-L157) e [src/utils/joblib_splitter.py](file:///d:/Projects/tcc_extraction/src/utils/joblib_splitter.py#L41-L95)

Duas peças de engenharia de sistemas que separam um notebook frágil de uma plataforma operável em concessionária real:
1. **`QueuePool(pool_size=5, max_overflow=0)` + `pool_pre_ping=True`** no Oracle/ORCA: permite extração paralela com exatamente 5 sessões reutilizadas, erradicando o vazamento de 142 cursores abertos que derrubaria o banco corporativo.
2. **`dump_split_joblib` com manifesto SHA-256**: serializa modelos grandes em memória, fatia automaticamente em blocos comprimidos `<= 25 MB` com hash criptográfico do arquivo completo e de cada parte, permitindo versionar modelos congelados no Git com reconstrução transparente e verificada (`load_split_joblib`).

```python
# src/db.py — Pool estritamente limitado para não sobrecarregar sessões Oracle/ORCA
return _sa_create_engine(
    url,
    pool_pre_ping=True,
    poolclass=QueuePool,
    pool_size=pool_size,
    max_overflow=max_overflow,
)
```
```python
# src/utils/joblib_splitter.py — Fatiamento determinístico com integridade SHA-256
buf = io.BytesIO()
joblib.dump(obj, buf, compress=compress)
data = buf.getvalue()
full_sha256 = hashlib.sha256(data).hexdigest()
...
for idx in range(num_parts):
    chunk = data[idx * chunk_bytes : min((idx + 1) * chunk_bytes, total_bytes)]
    part_path = target.parent / f"{target.name}.part{idx:02d}"
    part_path.write_bytes(chunk)
    part_hashes.append(hashlib.sha256(chunk).hexdigest())
```

---

## 6. Os 8 Melhores "Takes" (Descobertas Reais de Engenharia e Ciência do Projeto)

1. **O mistério da cobertura de 50% vs. 99,49% (UC `685***20`):** O painel antigo dividia 6 leituras/hora por 12 slots de uma grade fixa de 5 min → parecia 50% de perda de pacotes! Na verdade o medidor operava em cadência nativa de **10 minutos** (`6/6 = 100%` nas 168h; `4.298/4.320 = 99,49%` em 30 dias). **Take:** *nunca assuma cadência fixa em parques AMI heterogêneos.*
2. **O colapso do LOF (`1e10`) por empates exatos:** Medidores idênticos (ex.: consumo zero) geravam distância k-NN igual a zero e escore LOF `~10¹⁰`, esmagando toda a normalização (`ρ = 0,02`). Aplicar `log1p(max(-score - 1, 0))` resgatou o LOF (`ρ = +0,46`), transformando-o no detector mais complementar do comitê.
3. **PCA sem padronização interna (`StandardScaler`):** Como a potência instalada (kVA) e consumo (kWh) tinham escalas muito maiores que desequilíbrio (0–2), a PCA reconstruía apenas a variável de maior variância e gerou correlação negativa (`-0,20`) antes do saneamento. Padronizar internamente corrigiu para `+0,41`.
4. **O bug silencioso do bônus `+0,15` (descoberto na auditoria P0):** A regra `v_imb_cols` buscava `"IMBALANCE_MAX"` e acabou pegando **desequilíbrio de corrente** (`> 0,20`), dando `+0,15` para **46,7% da cidade inteira (30.694 UCs)** em vez de apenas **0,19% (124 UCs)** de violações reais de tensão/reversão! Corrigir na `v2` manteve 92,4% do Top-2.000 e **aumentou** a persistência temporal de 6 dias para **91,5%**.
5. **Os 1,7 milhão de alarmes "invisíveis" (descoberto na auditoria P1):** O pipeline de 25/11 lia `alarm_events.parquet` (arquivo único), enquanto os alarmes estavam em **41 partições** dentro de `output/events/alarm_events/`. Ler particionado revelou **1.377 UCs** com `S_tamper >= 0,70` (496 concordantes com anomalia elétrica e 881 sinais complementares exclusivos).
6. **A brecha dos 15 dias de silêncio consecutivo (descoberta nas perturbações P1):** Um medidor que ficou 15 dias mudo teve média de 99% nos 15 dias em que transmitiu — enganando o gate baseado só em `avg_interval_coverage >= 70%` (31% de vazamento na `v1`)! Combinar `cov >= 70%` **E** `reliable_days >= 18` na `v2` zerou o vazamento para **0,0% (100% em quarentena)**.
7. **Separação espacial de falhas de Telecom:** Cruzar a cobertura da UC com a mediana do mesmo transformador (ou grade geográfica de 250 m) permitiu separar automaticamente **910 falhas isoladas de medidor/antena** de **1.829 casos em gargalo coletivo RF Mesh** — evitando enviar equipe comercial para problema de repetidor de rádio.
8. **Persistência real de +6 dias (`19/11 → 25/11`):** Sem precisar de TOI, os 6 dias adicionais de dados provaram que **89% a 92% da carteira priorizada persiste** de uma semana para outra (`Spearman 0,95–0,97`), provando que o algoritmo captura fenômenos estruturais persistentes e não ruído aleatório de um dia.

---

## 7. Usos Mais Criativos da Plataforma (Além de Fiscalização de Fraude)

A mesma arquitetura construída neste repositório pode ser reaproveitada imediatamente para pelo menos **6 aplicações de alto valor** em distribuidoras de energia:

1. **Caça à Geração Distribuída (GD) Não Homologada e Proteção de Rede:** Detectou **100%** das injeções solares reversas no experimento controlado e identificou **1.451 UCs** com fluxo reverso no OOT — essencial para evitar inversão de fluxo em transformadores, sobretensão local e riscos de segurança para eletricistas em manutenção.
2. **Auditoria Automática de Cadastro Comercial (CIS × Campo):** Como visto no caso real `291***96` (medidor constava removido em `11/06/2026`, mas transmitiu ~28 kWh/dia até `05/11`!), o cruzamento temporal `UC–NIO` encontra "medidores fantasmas", ligações ativas sem faturamento e cadastros defasados.
3. **Manutenção Preditiva da Rede de Telecomunicações RF Mesh:** O mapa de `GARGALO_COLETIVO_RF_MESH` por transformador/grade de 250 m funciona como um **radar de saúde de concentradores e repetidores**, direcionando equipes de automação antes que a área inteira fique cega.
4. **Detector de Medidores Travados e Defeitos Metrológicos:** A regra física `FA ~ 0` com `U_L1 > 100 V` e comunicação ativa identifica bobinas de corrente abertas ou registradores congelados (515 UCs no OOT) sem confundir com falta de energia.
5. **Radar de Eventos Coletivos e Contingências em Transformadores:** O `SpuriousFilter` (16/16 eventos coletivos separados no experimento) pode ser invertido para monitorar **quedas de carga ou afundamentos simultâneos por transformador**, ajudando o Centro de Operação (COD) a detectar sobrecargas, fases abertas ou interrupções parciais em baixa tensão.
6. **Planejamento de Capacidade de Equipes sob Orçamento Fixo:** O módulo de projeção operacional (`K = 2.000/mês`) permite simular diferentes políticas de despacho (por severidade estatística, por materialidade em kWh/dia, por persistência CPD ou por qualidade de evidência), transformando ML em ferramenta direta de gestão de carteira.

---

## 8. Transparência Total para a Banca: O que é Derivado vs. Heurístico

Para nunca ser pego de surpresa na arguição, tenha esta divisão na ponta da língua:

- **O que é estritamente empírico / calculado dos dados:** normalização `P01–P99` no treino, limiar `P95` (`0,5131`), todas as correlações, reconciliações, persistência `19/11 → 25/11`, matriz de perturbações controladas e ablação `V0–V5`.
- **O que é escolha heurística inicial de projeto (não otimizada por falta de rótulos de campo):**
  - Pesos do ensemble (`0,22 / 0,20 / 0,18 / 0,18 / 0,10 / 0,12`) → **validados por sensibilidade** (±20% muda `< 1,4%` do Top-2.000; uniformes têm `93,6%` de sobreposição).
  - Contaminação nominal (`5%`), bônus `+0,15`, pesos de `S_tamper` (`0,70/0,30/0,15`), pontos de corte do `MoE` (`150` e `1.000` amostras) e pesos aditivos da variante P1 `V5` (`+0,06/+0,04/+0,05`) → **escolhas de projeto explicáveis fisicamente**, cuja calibração ótima depende de rótulos independentes no piloto prospectivo (Cap. 14).
  - Premissas econômicas (`K=2.000`, `40%`, `+5 p.p.`, `R$ 180 / R$ 1.800`) → **cenário paramétrico hipotético**, jamais resultado medido.
