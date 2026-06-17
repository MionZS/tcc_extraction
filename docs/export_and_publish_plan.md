# Plano: Export, Verify e Publish — Pipeline ARAUCARIA

> Baseado nos padrões do Fluxo_BI (`daily_orchestrator.py`, `incremental_publish.py`)  
> Adaptado para o pipeline de amostra ARAUCARIA (200 NIOs, execução single-shot)

---

## 1. Prioridades do Usuário

| # | Prioridade | Descrição |
|---|-----------|-----------|
| 1 | futuro | Run manifest — JSON de rastreabilidade por execução |
| 2 | futuro | Verify output — checagens de integridade antes de publicar |
| 3 | **agora** | **Básico: export + copia para OneDrive** |
| 4 | revisar | Revisar plano e analisar diferenças com Fluxo_BI |
| 5 | futuro | Bootstrap + publish para SharePoint: `E:\mion\OneDrive - copel.com\transfer_area\machine_learning_pipeline\input\sample200` |

---

## 2. Arquitetura Atual

```
CIS → sample → GEO join → MDM → join final
                                  ↓
                    output/refined/reports/sample200/
                    araucaria_model_input_sample_200_YYYYMMDD.csv
```

## 3. Arquitetura Proposta

```
CIS → sample → GEO join → MDM → join final
                                  ↓
                    output/refined/reports/sample200/
                                  ↓
                    [verify_output] → checagens de integridade
                                  ↓
                    [export_manager] → cópia para OneDrive target
                                  ↓
                    [run_manifest] → JSON de rastreabilidade
                                  ↓
                    OneDrive target:
                    E:\mion\OneDrive - copel.com\transfer_area\
                      machine_learning_pipeline\input\sample200\
```

---

## 4. Módulos Planejados

### 4.1 `verify_output.py` — Verificação de Integridade

**Inspiração:** Fluxo_BI `daily_orchestrator.py` — validação pós-merge

**Responsabilidade:** Verificar que o CSV final atende critérios de qualidade antes de publicar.

**Checks:**
1. **Arquivo existe** e não está vazio
2. **Colunas obrigatórias presentes** — lista definida em config ou constantes
3. **Sem NULLs em colunas-chave** — `UC`, `NIO` (chaves primárias do modelo)
4. **Contagem de linhas coerente** — deve ser ~200 (tamanho da amostra ± tolerância)
5. **Tipos básicos** — colunas numéricas contêm números
6. **Sem duplicatas** em chaves primárias

**Interface:**
```python
@dataclass
class VerificationResult:
    passed: bool
    checks: list[dict]  # [{name, passed, detail}]
    row_count: int
    column_count: int

def verify_model_input(csv_path: Path, *, expected_rows: int = 200, key_columns: list[str] | None = None) -> VerificationResult:
    ...
```

**Uso no pipeline:**
```python
result = verify_model_input(joined_csv_path, expected_rows=args.sample_size)
if not result.passed:
    for check in result.checks:
        if not check["passed"]:
            print(f"  FAIL: {check['name']} — {check['detail']}")
    raise SystemExit(1)
```

---

### 4.2 `export_manager.py` — Cópia para Target

**Inspiração:** Fluxo_BI `incremental_publish.py` — `_copy_file_to_baseline()`

**Responsabilidade:** Copiar o CSV final para o diretório OneDrive target.

**Diretório target (OneDrive sincronizado):**
```
E:\mion\OneDrive - copel.com\transfer_area\machine_learning_pipeline\input\sample200
```

**Lógica:**
1. Criar diretório target se não existir
2. Copiar com `shutil.copy2()` (preserva timestamps)
3. Criar arquivo `.meta.json` ao lado do CSV com:
   - `source`: caminho de origem
   - `copied_at`: ISO timestamp
   - `row_count`: linhas do CSV
   - `file_size_bytes`: tamanho do arquivo
   - `pipeline_version`: versão do script (do pyproject.toml ou tag)
4. Log da operação

**Interface:**
```python
@dataclass
class PublishResult:
    source: Path
    target: Path
    bytes_copied: int
    meta_path: Path

def publish_to_target(
    csv_source: Path,
    target_dir: Path,
    *,
    filename_override: str | None = None,
) -> PublishResult:
    ...
```

**Observação:** Não é necessário "shadow" para a amostra (são 200 linhas, execução diária simples). O padrão Fluxo_BI usa shadow para dados incrementais com millions de linhas — aqui basta cópia direta + `.meta.json`.

---

### 4.3 `run_manifest.py` — Rastreabilidade

**Inspiração:** Fluxo_BI — run logs e metadata

**Responsabilidade:** Salvar JSON com todos os metadados da execução.

**Salvo em:** `output/runs/run_YYYYMMDD_HHMMSS.json`

**Conteúdo do manifest:**
```json
{
  "run_id": "20260615_143022",
  "started_at": "2026-06-15T14:30:22",
  "finished_at": "2026-06-15T14:31:45",
  "duration_seconds": 83,
  "report_day": "2026-06-14",
  "sample_size": 200,
  "steps": {
    "cis_extract": {"rows": 123456, "duration_seconds": 12.3, "output": "raw/CIS/...csv"},
    "sample_selection": {"nio_count": 200, "uc_count": 187},
    "geo_extract": {"rows": 187, "duration_seconds": 5.1, "output": "raw/GEO/sample200/...csv"},
    "geo_join": {"rows": 200, "output": "raw/CIS/sample200/..._geo.csv"},
    "mdm_extract": {"rows": 400, "duration_seconds": 8.7, "output": "raw/ORCA/sample200/...csv"},
    "final_join": {"rows": 200, "output": "refined/reports/sample200/...csv"},
    "verify": {"passed": true, "checks": 6},
    "publish": {"target": "E:\\mion\\OneDrive...", "bytes": 52340}
  },
  "errors": []
}
```

**Interface:**
```python
@dataclass
class RunManifest:
    run_id: str
    started_at: datetime
    report_day: date
    sample_size: int
    steps: dict
    errors: list[str]

    def record_step(self, name: str, **kwargs) -> None: ...
    def finalize(self) -> None: ...
    def save(self, output_dir: Path) -> Path: ...
```

---

### 4.4 `publish_manager.py` — Bootstrap + Publish (Futuro)

**Inspiração:** Fluxo_BI `incremental_publish.py` — `_bootstrap_shadow_from_baseline()`

**Responsabilidade (quando aplicável):**
1. **Bootstrap:** Ler CSV existente do target OneDrive → trazer para local como "shadow"
2. **Merge:** Comparar novo output com shadow → detectar duplicatas, conflitos
3. **Publish:** Escrever resultado final no target + backup do anterior
4. **Verify:** Validação pós-publish

**Nota:** Para a amostra (200 NIOs), o bootstrap é opcional — pode ser um overwrite limpo. Para a pipeline diária completa (pipeline.py), será necessário incremental merge.

---

## 5. Adaptação do Padrão Fluxo_BI

| Conceito Fluxo_BI | Adaptação ARAUCARIA |
|---|---|
| `KEY_COLUMNS = ["DATA", "MUNICIPIO", "INTELIGENTE"]` | `KEY_COLUMNS = ["UC"]` ou `["NIO"]` |
| Shadow + incremental merge | Overwrite direto (amostra é re-executável) |
| Streaming com `pq.ParquetWriter` | Não necessário (200 linhas, csv simples) |
| `CONCURRENCY = 2` | Não necessário |
| Backup antes de publish | `.meta.json` + timestamp no filename |
| Reference date tracking | Token `YYYYMMDD` no nome do arquivo |
| Deduplication por KEY_COLUMNS | Verificação de unicidade no verify |

---

## 6. Fluxo Completo (Pós-Implementação)

```
python araucaria_sample_pipeline.py --days-back 1

  [1] CIS extract          → output/raw/CIS/araucaria_cis_20260614.csv
  [2] Sample 200 NIOs      → output/raw/CIS/sample200/...sample_200_20260614.csv
  [3] GEO extract por UCs  → output/raw/GEO/sample200/...geo_ucs_sample_200_20260614.csv
  [4] GEO join             → output/raw/CIS/sample200/..._geo.csv
  [5] MDM extract          → output/raw/ORCA/sample200/...mdm_sample_200_20260614.csv
  [6] Final join           → output/refined/reports/sample200/...model_input_sample_200_20260614.csv
  [7] Verify               → OK / FAIL
  [8] Publish              → E:\mion\OneDrive...\input\sample200\...model_input_sample_200_20260614.csv
  [9] Run manifest         → output/runs/run_20260614_143022.json
```

---

## 7. Ordem de Implementação

1. ✅ Ajustar diretório GEO (`raw/GEO/`) — **FEITO**
2. ✅ `verify_output.py` — checks de integridade (colunas, NULLs, duplicatas, row count) — **FEITO**
3. ✅ `export_manager.py` — cópia para OneDrive + `.meta.json` — **FEITO**
4. ✅ `run_manifest.py` — JSON de rastreabilidade por execução — **FEITO**
5. ✅ Integrar no `araucaria_sample_pipeline.py` — passos 7 (verify), 8 (publish), 9 (manifest) — **FEITO**
6. 🔲 `publish_manager.py` — bootstrap + merge (quando pipeline diária precisar)
7. 🔲 Testar fluxo completo
