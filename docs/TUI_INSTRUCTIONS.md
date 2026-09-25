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
- **O que faz**: Dispara a extração completa dos bancos de dados Oracle (CIS, GEO e ORCA/MDM), normaliza as camadas do datalake em formato Parquet (`output/`) e constrói a matriz consolidada `training_dataset.parquet`.
- **Implementação**: `RichPipelineTUI.generate_dataset()` em `src/tui/tui_app.py` —
  modo alimentador único via `scripts/run_feeder_pipeline.py --feeder-name … --feeder-geo-id … --days-back N`
  (padrão Fonte Nova `6352460`), ou Araucária completa via `src/araucaria_sample_pipeline.py`.
- **Parâmetros Interativos**:
  - Nome do Alimentador (padrão: `Fonte_Nova`).
  - ID GEO do Alimentador (padrão: `6352460`).
  - Dias atrás / retroativos (`days_back`, padrão: 1).

#### 🤖 Opção [2]: Treinar & Validar Modelos de IA
- **O que faz**: Executa a divisão estrita de dados (`DatasetSplitter`), separando alimentadores por similaridade de circuitos elétricos em conjuntos de **Treino**, **Validação** e **Teste**.
- **Implementação**: `RichPipelineTUI.train_models()` com entrada fixa
  `output/model_input/v1/training_dataset.parquet` (+ `output/context/electrical_hierarchy.parquet` no modo MoE)
  e contaminação/random_state da sessão (opção 6; padrões 0.05 / 42). Chama `scripts/train_anomaly_model.py`
  (`--mode composite|hierarchical|baseline`):
- **Modelos Suportados**:
  1. Modelo Composto Multi-Técnica (IsoForest + LOF + ECOD + PCA + Regras PRODIST).
  2. Mistura de Especialistas (MoE Router - Alimentador vs Regional vs Global).
  3. Baseline (Isolation Forest + KMeans).

#### 📊 Opção [3]: Visualizar Estatísticas do Data Lake
- Exibe tabelas formatadas com Rich mostrando:
  - Tamanhos e dimensões das tabelas Parquet no Data Lake.
  - Alimentadores cadastrados e contagem de UCs vinculadas.

#### ⚡ Opção [4]: Comparar Especialistas vs Modelo Global
- Roda o teste isolado comparando os modelos especialistas do alimentador/região contra o modelo global âncora, exibindo a tabela comparativa de métricas.

#### 🩺 Opção [5]: Auditoria de Saúde do Data Lake
- Executa checagens de integridade física dos dados e dispara a suíte de testes.
- **Implementação**: `DailyAutomationRunner.run_daily_audit()` → `output/reports/daily_health_report_YYYYMMDD.md/.json`
  (+ `daily_health_report_latest.md`); statuses `HEALTHY / WARNING / INCOMPLETE / CRITICAL`
  (tensão fora de [100, 300] V, timestamps futuros = `CRITICAL`). Equivalente CLI:
  `python scripts/daily_automation.py --mode health-check` (ou `--mode test-only` / `--mode feeder` / `--mode full`).

#### ⚙️ Opção [6]: Configurações Globais
- Permite alterar dinamicamente a taxa de contaminação e a semente aleatória (*random state*).
