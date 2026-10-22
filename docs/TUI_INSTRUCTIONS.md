# Guia da TUI em Rich & Instruções do Sistema

> Entrada: `uv run main.py` (raiz do repo) → `main.py` → `src/tui/tui_app.py::main_tui()`.
> Fora do `uv`/venv, `show_environment_warning()` exibe o banner de ambiente não isolado.
> Configuração de credenciais: `.env` (ver `.env.example`) ou `config.json` (ver `config_example.json`) —
> precedência: env > config (`src/db.py`). Nunca commitar `config.json` / `.env`.

A nova Interface em Modo Texto (TUI) foi completamente reescrita utilizando o **Rich** para oferecer uma experiência visual moderna e um menu completo de controle do pipeline.

---

## 🚀 Como Executar

O comando oficial via **`uv`**:

```bash
uv run main.py
```

Ao rodar o comando acima, o `uv` instala e sincroniza automaticamente todas as dependências (`rich`, `polars`, `scikit-learn`, `oracledb`, etc.).

---

## 📋 Menu Completo da TUI

Ao iniciar, você verá o painel com as 7 opções do menu:

```text
Selecione uma opção do Menu Principal:
  [1] 📥 Gerar Dataset (Pipeline de Extração CIS/GEO/MDM)
  [2] 🤖 Treinar & Validar Modelos de IA (Composto / MoE / Baseline)
  [3] 📊 Visualizar Estatísticas do Data Lake
  [4] ⚡ Comparar Especialistas vs Modelo Global (No Teste)
  [5] 🩺 Executar Auditoria de Saúde & Tests
  [6] ⚙️ Configurações Globais
  [7] 🚪 Sair
```

### Detalhamento das Opções:

#### 📥 Opção [1]: Gerar Dataset
- **O que faz**: Dispara a extração completa dos bancos de dados Oracle (CIS, GEO e ORCA/MDM), normaliza as camadas do datalake em formato Parquet e/ou CSV (`output/`) e constrói a matriz consolidada `training_dataset`.
- **Formato de Saída Flexível**:
  - `Ambos (Parquet + CSV)` [Padrão]: Garante máxima interoperabilidade e persistência em ambos os formatos.
  - `Apenas CSV (.csv)`: Exporta todas as tabelas e partições em arquivos `.csv` delimitados por `;`.
  - `Apenas Parquet (.parquet)`: Modo de alta performance colunar comprimido com Zstandard.
- **Implementação**: `RichPipelineTUI.generate_dataset()` em `src/tui/tui_app.py` integrando com `scripts/run_araucaria_full.py`:
  1. **Araucária Completa (Single-Day)**: Extração de todos os alimentadores do município via `scripts/run_araucaria_full.py --days-back N`.
  2. **Alimentador Específico**: Filtro direcionado via `scripts/run_araucaria_full.py --feeder-name … --feeder-geo-id …`.
  3. **Período Contínuo (Multi-Day Batch)**: Extração iterativa de período histórico (ex: `2026-08-01` a ontem) com verificação automática de dias já extraídos.
  4. **⚡ Sincronização Inteligente (Auto-Backfill)**: Detecta a data mais recente gravada no Data Lake e extrai automaticamente até o dia de ontem completo (ex.: de sábado a segunda, caso a última parada tenha sido sexta), identificando e preenchendo quaisquer lacunas históricas pendentes.
- **Visualização em Tempo Real de Filas e Workers**:
  - **Barra de Fila Geral**: Monitora o total de lotes pendentes e concluídos do MDM para o dia.
  - **Barras por Worker**: Cada worker paralelo possui uma barra e status individual (Aguardando na fila, Consultando Oracle MDM, Baixando registros, Gravando Parquet).
  - **Reutilização Persistente de Conexões**: Conexões de workers dedicadas que mantêm o cursor aberto eliminando o overhead de handshake TCP/TLS a cada lote.

#### 🤖 Opção [2]: Treinar & Validar Modelos de IA
- **O que faz**: Executa a divisão estrita de dados (`DatasetSplitter`), separando alimentadores por similaridade de circuitos elétricos em conjuntos de **Treino**, **Validação** e **Teste**.
- **Controle de Período Temporal de Treinamento**:
  - Permite selecionar se o treinamento utilizará todo o histórico ou um **período específico** (Data Inicial e Data Final), com amostragem balanceada e sem vazamento de UCs.
- **Hierarquia & Ordem de Treinamento (3 Níveis Estruturados)**:
  - **Nível 1 (Físico da Rede / Qualidade de Energia)**:
    1. **⚡ Classificador de Perturbações de Tensão (PRODIST M8 / IEEE 1159)**: Classificação física multiclasse (Sag, Swell, Desbalanço, Faltas Monofásica/Bifásica, Blackout) com amostragem diária homogênea e sem risco de OOM.
  - **Nível 2 (Modelos Comportamentais da UC — Matriz $UC \times \text{cutoff}$)**:
    2. **🌍 Modelo Global Âncora**: Comitê Composto (IsoForest + LOF + ECOD + PCA + Regras PRODIST).
    3. **🔌 Especialistas Locais por Alimentador**: Treinamento dedicado para cada circuito elétrico (Fonte Nova, Angélica, etc.).
    4. **🔀 Mistura Hierárquica de Especialistas (MoE Router)**: Roteamento inteligente adaptativo (Especialista vs Global).
    5. **🎯 Pipeline Especialista Multi-Visão**: Decomposição em escores $S_{comm}$, $S_{elec}$, $S_{tamper}$ e diagnóstico operacional por UC.
  - **Nível 3 (Referência, Cadeia Completa & Governança)**:
    6. **📊 Modelo Baseline de Referência**: Isolation Forest + KMeans.
    7. **🚀 Treinamento Completo em Cadeia**: Executa todos os níveis na sequência ideal (Físico → Global → Especialistas → MoE).
    8. **🏛️ Registro & Governança de Modelos (Model Registry)**: Catálogo versionado com auditoria e comparação lado a lado.

#### 📊 Opção [3]: Visualizar Estatísticas do Data Lake
- Exibe tabelas formatadas com Rich mostrando:
  - Tamanhos e dimensões das tabelas Parquet e CSV no Data Lake (`output/`).
  - Alimentadores cadastrados e contagem de UCs vinculadas.
  - **Filtro de Visualização de Integridade por Tipo de Arquivo**:
    - `[1] Geral (Ambos Formatos)`: Exibe o calendário com sinalização verde (ambos sincronizados), ciano (formato parcial) e vermelho (lacuna).
    - `[2] Apenas Parquet`: Audita exclusivamente arquivos `.parquet`.
    - `[3] Apenas CSV`: Audita exclusivamente arquivos `.csv`.
  - **🔄 Sincronização de Formatos (Converter Faltantes Parquet ↔ CSV)**:
    - Converte automaticamente partições que possuem apenas `.parquet` para `.csv` e partições com apenas `.csv` para `.parquet`, garantindo 100% de preenchimento e integridade sem necessidade de nova consulta ao Oracle.

#### ⚡ Opção [4]: Comparar Especialistas vs Modelo Global
- Roda o teste isolado comparando os modelos especialistas do alimentador/região contra o modelo global âncora, exibindo a tabela comparativa de métricas.

#### 🩺 Opção [5]: Auditoria de Saúde do Data Lake
- Executa checagens de integridade física dos dados e dispara a suíte de testes.
- **Implementação**: `DailyAutomationRunner.run_daily_audit()` → `output/reports/daily_health_report_YYYYMMDD.md/.json`
  (+ `daily_health_report_latest.md`); statuses `HEALTHY / WARNING / INCOMPLETE / CRITICAL`.

#### ⚙️ Opção [6]: Configurações Globais
- Permite alterar dinamicamente:
  - Formato padrão de saída dos dados (`both`, `csv`, `parquet`).
  - Taxa padrão de contaminação dos detectores de anomalia.
  - Semente aleatória (*random state*).
