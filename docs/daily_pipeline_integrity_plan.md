# Plano: Integridade e Verificação da Pipeline Diária

> Baseado nos requisitos do usuário para pipeline diária robusta:
> - Verificação de nomes de arquivos diários
> - Verificação de período completo (entre menor e maior)
> - Cálculo de média de espaço consumido por arquivo
> - Detecção de dias que saem da média
> - Geração de arquivo com days_back específico e nome correto
> - Script de verificação de integridade
> - Manutenção de arquivo auxiliar de UCs cadastradas diário em parquet

---

## 1. Objetivos

1. **Verificação de Nomes de Arquivos Diários** — Validar que cada arquivo de saída segue o padrão `araucaria_*_YYYYMMDD.csv` e corresponde ao `days_back` correto.

2. **Verificação de Período Completo** — Garantir que o conjunto de arquivos cobre todo o período entre a data mais antiga e a mais nova (sem lacunas).

3. **Monitoramento de Espaço** — Calcular tamanho médio de arquivo e detectar outliers (dias que consomem > X% da média).

4. **Geração Sob Demanda** — Permitir extração de arquivo específico por `days_back` com nome correto.

5. **Verificação de Integridade** — Script autônomo que verifica todo o conjunto de arquivos diários.

6. **Arquivo Auxiliar de UCs** — Manter `output/raw/cis/daily_cadastrados/*.parquet` com UCs únicas por dia para referência rápida.

---

## 2. Arquitetura de Saída Diária (pipeline.py)

```
output/
├── raw/CIS/                          # daily: araucaria_cis_YYYYMMDD.csv
├── raw/CIS/daily_cadastrados/        # daily: ucs_YYYYMMDD.parquet
├── raw/ORCA/                         # daily: araucaria_mdm_YYYYMMDD.csv
├── refined/reports/                  # daily: araucaria_model_input_YYYYMMDD.csv
└── runs/                             # daily: run_YYYYMMDD_HHMMSS.json
```

**Padrão de Nomenclatura:**
- CIS: `araucaria_cis_YYYYMMDD.csv`
- ORCA: `araucaria_mdm_YYYYMMDD.csv`
- Join final: `araucaria_model_input_YYYYMMDD.csv`
- UCs: `ucs_YYYYMMDD.parquet`

---

## 3. Módulos Planejados

### 3.1 `daily_file_naming.py` — Verificação de Nomes

**Responsabilidade:** Validar nomes de arquivos diários.

**Funções:**
```python
def validate_file_naming(directory: Path, expected_prefix: str) -> list[str]:
    """Retorna lista de arquivos com nomes inválidos."""

def extract_date_from_filename(filename: str) -> date | None:
    """Extrai YYYYMMDD do nome do arquivo."""

def check_days_back_consistency(csv_path: Path, days_back: int) -> bool:
    """Verifica se nome do arquivo corresponde ao days_back."""
```

**Uso:**
```python
invalid = validate_file_naming(raw_cis_dir, "araucaria_cis_")
if invalid:
    print(f"Arquivos com nomes inválidos: {invalid}")
```

---

### 3.2 `period_coverage.py` — Verificação de Período Completo

**Responsabilidade:** Garantir que não há lacunas no período.

**Funções:**
```python
def get_file_dates(directory: Path, pattern: str) -> set[date]:
    """Extrai todas as datas de arquivos em um diretório."""

def check_period_coverage(dates: set[date], start_date: date, end_date: date) -> list[date]:
    """Retorna datas faltantes no intervalo."""

def generate_missing_dates(start_date: date, end_date: date) -> list[date]:
    """Gera todas as datas no intervalo (inclusive)."""
```

**Lógica:**
1. Coletar todas as datas de arquivos em `raw/CIS/`
2. Calcular `min_date` e `max_date`
3. Gerar todas as datas no intervalo
4. Identificar faltantes

---

### 3.3 `file_size_analysis.py` — Monitoramento de Espaço

**Responsabilidade:** Calcular tamanho médio e detectar outliers.

**Funções:**
```python
def analyze_file_sizes(directory: Path) -> dict:
    """Retorna estatísticas: count, mean, std, min, max, outliers."""

def detect_outliers(sizes: list[int], threshold_percent: float = 1.5) -> list[tuple[date, int]]:
    """Identifica arquivos com tamanho > threshold * std."""
```

**Lógica:**
1. Coletar tamanho de cada arquivo CSV
2. Calcular média e desvio padrão
3. Identificar dias com tamanho > média + 1.5*std

---

### 3.4 `extract_by_days_back.py` — Geração Sob Demanda

**Responsabilidade:** Extrair arquivo específico por `days_back`.

**Funções:**
```python
def extract_file_by_days_back(
    source_dir: Path,
    days_back: int,
    output_dir: Path,
    prefix: str = "araucaria_cis_",
) -> Path:
    """Copia arquivo para target com nome correto."""
```

**Uso:**
```python
extract_file_by_days_back(
    source_dir=raw_cis_dir,
    days_back=1,
    output_dir=target_dir,
    prefix="araucaria_cis_",
)
```

---

### 3.5 `integrity_check.py` — Verificação de Integridade

**Responsabilidade:** Script autônomo que verifica todo o conjunto de arquivos diários.

**Checks:**
1. Nomes de arquivos válidos
2. Período completo (sem lacunas)
3. Tamanhos de arquivo dentro da faixa esperada
4. Colunas obrigatórias presentes
5. Sem duplicatas em chaves primárias
6. Arquivo auxiliar de UCs existe e está atualizado

**Interface:**
```python
def run_integrity_check(
    base_dir: Path,
    expected_key_columns: list[str] = ["UC", "NIO"],
) -> IntegrityReport:
    """Executa todas as verificações e retorna relatório."""
```

**Relatório:**
```json
{
  "passed": false,
  "checks": [
    {"name": "file_naming", "passed": true, "details": "Todos os 30 arquivos têm nomes válidos"},
    {"name": "period_coverage", "passed": false, "details": "Faltando datas: 2026-06-12"},
    {"name": "file_size", "passed": true, "details": "Tamanho médio: 1.2MB, outliers: 0"},
    {"name": "columns", "passed": true, "details": "Colunas obrigatórias presentes"},
    {"name": "duplicates", "passed": true, "details": "Sem duplicatas"},
    {"name": "ucs_file", "passed": false, "details": "Arquivo auxiliar ucs_20260617.parquet não encontrado"}
  ]
}
```

---

### 3.6 `daily_ucs_tracker.py` — Arquivo Auxiliar de UCs

**Responsabilidade:** Manter `output/raw/cis/daily_cadastrados/*.parquet`.

**Funções:**
```python
def save_daily_ucs(
    ucs: list[str],
    output_dir: Path,
    report_date: date,
) -> Path:
    """Salva UCs únicas em parquet com timestamp."""

def load_daily_ucs(
    input_dir: Path,
    report_date: date,
) -> DataFrame:
    """Carrega UCs para um dia específico."""
```

**Schema do Parquet:**
```python
@dataclass
class DailyUCs:
    date: str  # YYYYMMDD
    ucs: list[str]
    count: int
    created_at: str  # ISO timestamp
```

**Uso no pipeline:**
```python
# Após _collect_unique_values()
save_daily_ucs(selected_ucs, daily_ucs_dir, report_day)
```

---

## 4. Fluxo de Trabalho Diário (pipeline.py)

```
1. Executar extrações CIS → ORCA → join final
2. Salvar daily_ucs.parquet em output/raw/cis/daily_cadastrados/
3. Registrar run manifest em output/runs/
4. Executar verificação de integridade (opcional)
5. Publicar para OneDrive (se necessário)
```

## 5. Scripts de Utilitários

### 5.1 `run_daily_integrity.py`
```bash
python run_daily_integrity.py --base-dir output --days-back 30
```

### 5.2 `extract_single_day.py`
```bash
python extract_single_day.py --days-back 1 --output-dir ./extract
```

### 5.3 `analyze_file_sizes.py`
```bash
python analyze_file_sizes.py --directory output/raw/CIS --days-back 30
```

---

## 6. Ordem de Implementação

1. ✅ `verify_output.py` — verificação de integridade (já feito)
2. ✅ `export_manager.py` — cópia para OneDrive (já feito)
3. ✅ `run_manifest.py` — rastreabilidade (já feito)
4. 🔄 `daily_file_naming.py` — verificação de nomes
5. 🔄 `period_coverage.py` — verificação de período completo
6. 🔄 `file_size_analysis.py` — monitoramento de espaço
7. 🔄 `extract_by_days_back.py` — extração sob demanda
8. 🔄 `integrity_check.py` — script de verificação completo
9. 🔄 `daily_ucs_tracker.py` — arquivo auxiliar de UCs
10. 🔄 Integrar no `pipeline.py` (pipeline diária)

---

## 7. Adaptação do Fluxo_BI

| Conceito Fluxo_BI | Adaptação Diária |
|---|---|
| `KEY_COLUMNS = ["DATA", "MUNICIPIO", "INTELIGENTE"]` | `KEY_COLUMNS = ["UC"]` |
| Shadow + incremental merge | **Não necessário** — pipeline diária sobrescreve arquivos diários
| Streaming com Parquet | Usar Polars para daily_cadastrados.parquet
| `CONCURRENCY = 2` | Execução single-threaded por dia
| Backup antes de publish | `output/runs/` com manifest por execução
| Reference date tracking | `days_back` no nome do arquivo + token de execução

---

## 8. Observações Específicas do Projeto

- **Pipeline diária** (`pipeline.py`) usa Polars, saída Parquet para join final
- **Pipeline de amostra** (`araucaria_sample_pipeline.py`) usa CSV, saída para OneDrive
- **daily_cadastrados.parquet** é auxiliar, não parte do modelo final
- **Verificação de integridade** pode ser opcional (rápida) ou completa (detalhada)
- **Monitoramento de espaço** ajuda a identificar dias anormais (ex: extração mal sucedida)

---

## 9. Próximos Passos

1. Implementar `daily_file_naming.py`
2. Implementar `period_coverage.py`
3. Implementar `file_size_analysis.py`
4. Implementar `extract_by_days_back.py`
5. Implementar `integrity_check.py`
6. Implementar `daily_ucs_tracker.py`
7. Integrar no `pipeline.py`
8. Testar fluxo completo com dados reais

---

## 10. Templates

### Template: daily_file_naming.py
```python
# daily_file_naming.py
from pathlib import Path
from datetime import date
import re

FILENAME_PATTERN = re.compile(r"^araucaria_(cis|mdm)_?(\d{8})\.csv$")

def validate_file_naming(directory: Path, expected_prefix: str) -> list[str]:
    invalid = []
    for file in directory.iterdir():
        if file.is_file() and file.name.startswith(expected_prefix):
            match = FILENAME_PATTERN.match(file.name)
            if not match:
                invalid.append(file.name)
    return invalid
```

### Template: period_coverage.py
```python
# period_coverage.py
from pathlib import Path
from datetime import date, timedelta

def get_file_dates(directory: Path, pattern: str) -> set[date]:
    dates = set()
    for file in directory.glob(f"{pattern}_*.csv"):
        try:
            date_str = file.stem.split("_")[-1]
            dates.add(date.fromisoformat(date_str))
        except (ValueError, IndexError):
            continue
    return dates

def check_period_coverage(dates: set[date], start_date: date, end_date: date) -> list[date]:
    all_dates = {start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)}
    return sorted(all_dates - dates)
```

---

*Plano criado em 2026-06-17. Atualize conforme necessário.*