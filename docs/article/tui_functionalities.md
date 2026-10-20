# Mapeamento Completo de Funcionalidades do Sistema (TUI & CLI)

Este documento apresenta a matriz exaustiva de todas as capacidades do ecossistema **TCC Extraction (Araucária / Copel)**, demarcando explicitamente o que está acessível via terminal interativo (**TUI - Rich**) e o que está disponível via scripts de linha de comando (**CLI / VS Code Tasks**).

---

## 1. Resumo Executivo: O que Você Perguntou

### ❓ "Posso parar essa extração e rodar do dia que parou?"
**SIM!** 
O pipeline possui idempotência estrita e detecção de continuidade:
1. Se você interromper a extração (`Ctrl+C`), os dias cujas partições foram consolidadas estão salvos com segurança em `output/measurements/ami_instantaneous/report_year=*/...`.
2. Ao reiniciar via TUI (**Opção 1 $\rightarrow$ [4] Sincronização Inteligente**), a função `find_datalake_date_coverage()` detecta a última data consolidada (ex: `11/09/2026`) e retoma exatamente a partir de `12/09/2026` até ontem, pulando os dias já extraídos sem retrabalho.

### ❓ "Se eu pedir em CSV, ele vai converter os parquets já existentes e consultar o resto direto pra CSV?"
**O fluxo é ainda mais seguro e flexível:**
1. **Para os Parquets já existentes:** Na TUI (**Opção 3 $\rightarrow$ [4] 🔄 Sincronizar Formatos**), o sistema converte em lote todos os `.parquet` existentes para `.csv` em poucos segundos usando Polars, sem fazer nenhuma consulta ao Oracle.
2. **Para a nova extração:** Ao iniciar a extração (Opção 1), selecione o formato `Ambos (Parquet + CSV)` [Padrão] ou `Apenas CSV`. Os novos dias já serão consultados, normalizados e gravados diretamente em `.csv` (com a opção de manter o Parquet colunar para performance analítica).

### ❓ "Tudo é acessível pela TUI? Todas as opções?"
**A TUI cobre 100% dos fluxos operacionais de ponta a ponta:** Extração, Sincronização Inteligente, Treinamento com filtro de data/alimentador, Governança/Auditoria de Continuidade, Sincronização Parquet $\leftrightarrow$ CSV, Comparação de Modelos e Auditoria de Saúde.
Apenas rotinas analíticas pontuais de pesquisa (ex: simulação manual de weather, extrações legadas monocircuito em CSV de 200 NIOs) são scripts utilitários auxiliares de laboratório.

---

## 2. Matriz de Cobertura: TUI vs CLI

| Categoria | Funcionalidade | Presente na TUI? | Disponível via CLI? | Arquivo / Módulo Responsável |
| :--- | :--- | :---: | :---: | :--- |
| **Extração** | Sincronização Inteligente (Auto-Backfill até ontem) | ✅ Sim (Opção 1, [4]) | ✅ Sim (`--auto-backfill`) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Extração** | Extração de Período Histórico Customizado (Start/End) | ✅ Sim (Opção 1, [3]) | ✅ Sim (`--start-date`, `--end-date`) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Extração** | Extração Single-Day por `days_back` | ✅ Sim (Opção 1, [1]) | ✅ Sim (`--days-back N`) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Extração** | Extração Direcionada por Alimentador Específico | ✅ Sim (Opção 1, [2]) | ✅ Sim (`--feeder-name`, `--feeder-geo-id`) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Extração** | Seleção de Formato de Saída (CSV, Parquet, Ambos) | ✅ Sim (Opção 1 Prompt) | ✅ Sim (`--output-format both\|csv\|parquet`) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Extração** | Monitoramento Multi-Barra por Worker e Fila Geral | ✅ Sim (Rich Progress) | ✅ Sim (Terminal Output) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Extração** | Reutilização Persistente de Conexão/Cursor por Worker | ✅ Sim (Transparente) | ✅ Sim (Automático) | [`scripts/run_araucaria_full.py`](file:///d:/Projects/tcc_extraction/scripts/run_araucaria_full.py) |
| **Data Lake** | Auditoria de Dimensões e Contagens de Tabelas | ✅ Sim (Opção 3) | 🟡 Parcial (Scripts manuais) | [`src/tui/tui_app.py`](file:///d:/Projects/tcc_extraction/src/tui/tui_app.py) |
| **Data Lake** | Calendário Mensal de Continuidade Temporal | ✅ Sim (Opção 3) | ✅ Sim (`test_continuity.py`) | [`src/visualization/continuity.py`](file:///d:/Projects/tcc_extraction/src/visualization/continuity.py) |
| **Data Lake** | Filtro de Auditoria de Integridade por Formato (.parquet vs .csv) | ✅ Sim (Opção 3, [1],[2],[3]) | 🟡 Parcial (via API Python) | [`src/visualization/continuity.py`](file:///d:/Projects/tcc_extraction/src/visualization/continuity.py) |
| **Data Lake** | Conversão Automática de Faltantes Parquet $\leftrightarrow$ CSV | ✅ Sim (Opção 3, [4]) | ✅ Sim (Python script) | [`src/datasets/format_converter.py`](file:///d:/Projects/tcc_extraction/src/datasets/format_converter.py) |
| **Modelagem** | Treinamento Classificador PRODIST M8 (Tensão/Desbalanceamento) | ✅ Sim (Opção 2, [4]) | ✅ Sim (`train_voltage_classifier.py`) | [`scripts/train_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/scripts/train_voltage_classifier.py) |
| **Modelagem** | Filtro Temporal (Data Início / Fim) no Treinamento | ✅ Sim (Opção 2) | ✅ Sim (`--start-date`, `--end-date`) | [`scripts/train_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/scripts/train_voltage_classifier.py) |
| **Modelagem** | Filtro por Alimentador no Treinamento | ✅ Sim (Opção 2) | ✅ Sim (`--feeder`) | [`scripts/train_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/scripts/train_voltage_classifier.py) |
| **Modelagem** | Leitura Transparente de Treino a partir de Parquet ou CSV | ✅ Sim (Automático) | ✅ Sim (Automático) | [`scripts/train_voltage_classifier.py`](file:///d:/Projects/tcc_extraction/scripts/train_voltage_classifier.py) |
| **Modelagem** | Treinamento Modelo Composto Multi-Técnica (IsoForest/LOF/ECOD/PCA) | ✅ Sim (Opção 2, [1]) | ✅ Sim (`train_anomaly_model.py`) | [`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py) |
| **Modelagem** | Treinamento Mistura de Especialistas (MoE Hierárquico) | ✅ Sim (Opção 2, [2]) | ✅ Sim (`train_anomaly_model.py`) | [`src/models/hierarchical_trainer.py`](file:///d:/Projects/tcc_extraction/src/models/hierarchical_trainer.py) |
| **Modelagem** | Treinamento Baseline (IsoForest + KMeans) | ✅ Sim (Opção 2, [3]) | ✅ Sim (`train_anomaly_model.py`) | [`src/models/detector.py`](file:///d:/Projects/tcc_extraction/src/models/detector.py) |
| **Governança** | Repositório de Modelos Versionado (Model Registry) | ✅ Sim (Opção 2, [5]) | 🟡 Parcial (API Python) | [`src/models/registry.py`](file:///d:/Projects/tcc_extraction/src/models/registry.py) |
| **Governança** | Comparação Lado a Lado de Duas Versões ($\Delta$ Accuracy, $\Delta$ F1) | ✅ Sim (Opção 2, [5]) | 🟡 Parcial (API Python) | [`src/models/registry.py`](file:///d:/Projects/tcc_extraction/src/models/registry.py) |
| **Avaliação** | Comparativo de Especialistas vs Modelo Global no Conjunto de Teste | ✅ Sim (Opção 4) | ✅ Sim (Script CLI) | [`src/tui/tui_app.py`](file:///d:/Projects/tcc_extraction/src/tui/tui_app.py) |
| **Qualidade** | Auditoria de Saúde do Data Lake (Health Audit + Relatório) | ✅ Sim (Opção 5) | ✅ Sim (`scripts/daily_automation.py`) | [`src/automation/daily_runner.py`](file:///d:/Projects/tcc_extraction/src/automation/daily_runner.py) |
| **Configuração** | Configuração de Hiperparâmetros (Contaminação, Seed, Formato Padrão) | ✅ Sim (Opção 6) | ✅ Sim (Flags de linha de comando) | [`src/tui/tui_app.py`](file:///d:/Projects/tcc_extraction/src/tui/tui_app.py) |
| **Pesquisa** | Pipeline Completo com Clima / Meteorologia SIMEPAR | ❌ Não na TUI (Opção CLI) | ✅ Sim (`scripts/run_total_pipeline.py`) | [`scripts/run_total_pipeline.py`](file:///d:/Projects/tcc_extraction/scripts/run_total_pipeline.py) |
| **Pesquisa** | Extração Legada em CSV Monolítico (200 NIOs sample) | ❌ Arquivado (`vault/`) | ❌ Obsoleto | [`vault/araucaria_sample_pipeline_script.py`](file:///d:/Projects/tcc_extraction/vault/araucaria_sample_pipeline_script.py) |

---

## 3. O que Está e o que NÃO Está na TUI

### ✅ O que ESTÁ na TUI:
1. **Controle Total da Ingestão:** Inicialização do Data Lake, extração diária, recuperação de períodos retroativos e auto-backfill inteligente que continua do ponto exato da última parada.
2. **Formato Flexível na Ponta dos Dedos:** Seleção entre Parquet, CSV ou Ambos, e botão de sincronização em lote para converter arquivos faltantes Parquet $\leftrightarrow$ CSV sem reconsultar o Oracle.
3. **Treinamento Parametrizado:** Disparo de treino com recorte customizado de datas (ex.: treinar apenas com setembro ou com os últimos 15 dias) e alimentadores específicos.
4. **Governança e Visualização:** Linha do tempo visual em terminal Rich, calendário mensal com diagnóstico por cores (sincronizado, parcial ou ausente) e histórico de versões do Model Registry.

### ❌ O que NÃO ESTÁ na TUI (Acessível via CLI / Tasks):
1. **Integração de Clima Histórico SIMEPAR / Weather:** Executado via `uv run python scripts/run_total_pipeline.py` (ou `python -m src.main`), reservado para pipelines batch agendados em background devido a APIs externas de tempo.
2. **Scripts Legados do Vault:** Scripts arquivados da fase de exploração prévia (ex: `vault/araucaria_sample_pipeline_script.py`).
3. **Suíte Completa de Testes Pytest:** Executada via `uv run pytest tests/` ou pela tarefa do VS Code `Run Tests`.
