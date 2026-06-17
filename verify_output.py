"""verify_output.py — Verificação de integridade do CSV final do pipeline ARAUCARIA.

Inspiração: Fluxo_BI daily_orchestrator.py — validação pós-merge.
Adaptação: verificações simples para amostra de 200 NIOs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


# Colunas que NÃO podem ter NULL no CSV final
_DEFAULT_KEY_COLUMNS = ["UC", "NIO"]

# Tolerância para contagem de linhas (±10% do esperado)
_ROW_COUNT_TOLERANCE = 0.10


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class VerificationResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def verify_model_input(
    csv_path: Path,
    *,
    expected_rows: int = 200,
    key_columns: list[str] | None = None,
) -> VerificationResult:
    """Verifica integridade de um CSV de modelo de ML.

    Checks:
    1. Arquivo existe e não está vazio
    2. Colunas obrigatórias presentes
    3. Sem NULLs em colunas-chave
    4. Contagem de linhas coerente (± tolerância)
    5. Sem duplicatas nas chaves primárias
    """
    if key_columns is None:
        key_columns = list(_DEFAULT_KEY_COLUMNS)

    checks: list[CheckResult] = []
    rows: list[dict[str, str]] = []
    columns: list[str] = []

    # Check 1: Arquivo existe
    if not csv_path.exists():
        checks.append(CheckResult("file_exists", False, f"Arquivo não encontrado: {csv_path}"))
        return VerificationResult(passed=False, checks=checks)

    if csv_path.stat().st_size == 0:
        checks.append(CheckResult("file_not_empty", False, "Arquivo está vazio (0 bytes)"))
        return VerificationResult(passed=False, checks=checks)

    checks.append(CheckResult("file_exists", True, f"{csv_path.stat().st_size:,} bytes"))

    # Ler CSV
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            checks.append(CheckResult("has_header", False, "CSV sem cabeçalho"))
            return VerificationResult(passed=False, checks=checks)

        columns = list(reader.fieldnames)
        rows = list(reader)

    checks.append(CheckResult("has_header", True, f"{len(columns)} colunas: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}"))

    row_count = len(rows)

    # Check 2: Colunas obrigatórias presentes
    columns_upper = {c.strip().upper() for c in columns}
    missing = [kc for kc in key_columns if kc.upper() not in columns_upper]
    if missing:
        checks.append(CheckResult("key_columns_present", False, f"Colunas ausentes: {missing}"))
    else:
        checks.append(CheckResult("key_columns_present", True, f"Todas as {len(key_columns)} colunas-chave presentes"))

    # Check 3: Sem NULLs em colunas-chave
    null_counts: dict[str, int] = {}
    for kc in key_columns:
        kc_col = next((c for c in columns if c.strip().upper() == kc.upper()), None)
        if kc_col is None:
            continue
        null_count = sum(1 for row in rows if not row.get(kc_col, "").strip())
        if null_count > 0:
            null_counts[kc] = null_count

    if null_counts:
        detail = ", ".join(f"{k}: {v} NULLs" for k, v in null_counts.items())
        checks.append(CheckResult("no_nulls_in_keys", False, detail))
    else:
        checks.append(CheckResult("no_nulls_in_keys", True, "Nenhum NULL nas colunas-chave"))

    # Check 4: Contagem de linhas coerente
    tolerance = max(1, int(expected_rows * _ROW_COUNT_TOLERANCE))
    min_expected = expected_rows - tolerance
    max_expected = expected_rows + tolerance
    in_range = min_expected <= row_count <= max_expected
    checks.append(CheckResult(
        "row_count_coherent",
        in_range,
        f"{row_count} linhas (esperado ~{expected_rows}, faixa aceitável: {min_expected}-{max_expected})",
    ))

    # Check 5: Sem duplicatas nas chaves primárias
    key_col_map = {}
    for kc in key_columns:
        kc_col = next((c for c in columns if c.strip().upper() == kc.upper()), None)
        if kc_col:
            key_col_map[kc] = kc_col

    if key_col_map:
        seen: set[tuple[str, ...]] = set()
        duplicates = 0
        for row in rows:
            key_values = tuple(row.get(key_col_map[kc], "").strip().upper() for kc in key_col_map)
            if key_values in seen:
                duplicates += 1
            else:
                seen.add(key_values)

        if duplicates > 0:
            checks.append(CheckResult("no_duplicates", False, f"{duplicates} linhas duplicadas nas chaves"))
        else:
            checks.append(CheckResult("no_duplicates", True, f"Todos os {row_count} registros são únicos"))
    else:
        checks.append(CheckResult("no_duplicates", True, "Sem verificação de duplicatas (keys não mapeadas)"))

    # Resultado final
    all_passed = all(c.passed for c in checks)

    return VerificationResult(
        passed=all_passed,
        checks=checks,
        row_count=row_count,
        column_count=len(columns),
    )


def print_verification_report(result: VerificationResult) -> None:
    """Imprime relatório de verificação formatado."""
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"\n  Verification: {status} ({result.row_count} rows, {result.column_count} cols)")
    for check in result.checks:
        icon = "✓" if check.passed else "✗"
        print(f"    {icon} {check.name}: {check.detail}")
