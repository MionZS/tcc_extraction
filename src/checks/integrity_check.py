"""integrity_check.py — Verificação de integridade da pipeline diária.

Script autônomo que verifica:
- Nomes de arquivos válidos
- Período completo (sem lacunas)
- Tamanhos de arquivo dentro da faixa esperada
- Arquivo auxiliar de UCs existe
- Schema, chaves, unidades, temporality dos datasets normalizados
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.checks.daily_file_naming import validate_file_naming, extract_date_from_filename
from src.checks.period_coverage import get_file_dates, check_period_coverage
from src.checks.file_size_analysis import collect_file_sizes, compute_size_stats, detect_size_outliers, format_size
from src.datasets.schemas import get_schema, get_keys, TABLE_REGISTRY
from src.checks.period_coverage import get_expected_points


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"

# Directories that should contain dated files
_DATED_DIRS: dict[str, dict] = {
    "raw_CIS": {
        "path": "raw/CIS",
        "prefix": "araucaria_cis",
        "suffix": ".csv",
    },
    "raw_ORCA": {
        "path": "raw/ORCA",
        "prefix": "araucaria_mdm",
        "suffix": ".csv",
    },
    "refined_reports": {
        "path": "refined/reports",
        "prefix": "araucaria_model_input",
        "suffix": ".csv",
    },
}

# Required columns in final model input CSVs
_DEFAULT_KEY_COLUMNS = ["UC", "NIO"]

# Max age for daily_cadastrados parquet (days)
_MAX_UCS_AGE_DAYS = 3

# Datasets normalizados para verificar
_NORMALIZED_TABLES = [
    "meter_day_metadata",
    "meter_interval",
    "meter_instantaneous",
    "meter_register_snapshot",
]

# Faixas físicas esperadas por coluna (nome → (min, max, unidade))
_PHYSICAL_RANGES: dict[str, tuple[float, float, str]] = {
    "FA_INTERVAL": (0.0, 100.0, "kWh"),
    "RA_INTERVAL": (0.0, 100.0, "kWh"),
    "FA_ENERGY_C": (0.0, 100.0, "kWh"),
    "RA_ENERGY_C": (0.0, 100.0, "kWh"),
    "FA_ENERGY_D": (0.0, 100.0, "kWh"),
    "RA_ENERGY_D": (0.0, 100.0, "kWh"),
    "FA_MD_D": (0.0, 100.0, "kW"),
    "U_L1": (0.0, 500.0, "V"),
    "U_L2": (0.0, 500.0, "V"),
    "U_L3": (0.0, 500.0, "V"),
    "I_L1": (0.0, 1000.0, "A"),
    "I_L2": (0.0, 1000.0, "A"),
    "I_L3": (0.0, 1000.0, "A"),
    "I_INSTANT_L1": (0.0, 1000.0, "A"),
    "I_INSTANT_L2": (0.0, 1000.0, "A"),
    "I_INSTANT_L3": (0.0, 1000.0, "A"),
    "QUALITY_CODE": (0.0, 9.0, "código"),
}


@dataclass
class CheckResult:
    """Resultado de uma verificação individual."""
    name: str
    passed: bool
    detail: str


@dataclass
class IntegrityReport:
    """Relatório consolidado de integridade."""
    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        if not check.passed:
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "summary": self.summary,
        }


def check_file_naming(output_dir: Path) -> CheckResult:
    """Verifica nomenclatura de arquivos em todos os diretórios datados."""
    all_invalid: list[str] = []
    checked = 0

    for dir_key, info in _DATED_DIRS.items():
        dir_path = output_dir / info["path"]
        if not dir_path.exists():
            continue
        invalid = validate_file_naming(dir_path, prefix=None)
        all_invalid.extend(f"{info['path']}/{f}" for f in invalid)
        checked += 1

    if not all_invalid:
        return CheckResult(
            "file_naming",
            True,
            f"Todos os arquivos seguem o padrão em {checked} diretórios.",
        )
    return CheckResult(
        "file_naming",
        False,
        f"Arquivos com nomes inválidos: {', '.join(all_invalid[:10])}"
        + (f" (+{len(all_invalid) - 10} mais)" if len(all_invalid) > 10 else ""),
    )


def check_period(
    output_dir: Path,
    *,
    min_days_back: int = 30,
) -> CheckResult:
    """Verifica se não há lacunas no período dos últimos N dias."""
    end_date = date.today() - timedelta(days=1)  # yesterday
    start_date = end_date - timedelta(days=min_days_back)

    # Focus on CIS as the primary dated output
    cis_dir = output_dir / "raw" / "CIS"
    if not cis_dir.exists():
        return CheckResult(
            "period_coverage",
            True,
            "Diretório CIS não existe — sem dados para verificar.",
        )

    dates = get_file_dates(cis_dir, suffix=".csv", prefix="araucaria_cis")
    result = check_period_coverage(dates, start_date=start_date, end_date=end_date)

    if result["complete"]:
        return CheckResult(
            "period_coverage",
            True,
            f"Período completo: {result['start_date']} a {result['end_date']} "
            f"({result['total_days']} dias).",
        )
    return CheckResult(
        "period_coverage",
        False,
        f"Lacunas detectadas: {len(result['missing_dates'])} datas faltando. "
        f"Faltando: {', '.join(result['missing_dates'][:5])}"
        + (f" (+{len(result['missing_dates']) - 5} mais)" if len(result['missing_dates']) > 5 else "")
        + f" | Cobertura: {result['coverage_percent']}%",
    )


def check_sizes(output_dir: Path) -> CheckResult:
    """Verifica tamanhos de arquivo e detecta outliers."""
    cis_dir = output_dir / "raw" / "CIS"
    if not cis_dir.exists():
        return CheckResult("file_sizes", True, "Sem arquivos para analisar.")

    records = collect_file_sizes(cis_dir, suffix=".csv", prefix="araucaria_cis")
    stats = compute_size_stats(records)
    if stats is None:
        return CheckResult("file_sizes", True, "Nenhum arquivo encontrado.")

    outliers = detect_size_outliers(records)

    if not outliers:
        return CheckResult(
            "file_sizes",
            True,
            f"Média: {format_size(int(stats.mean_bytes))}, "
            f"desvio: {format_size(int(stats.std_bytes))}, "
            f"arquivos: {stats.count}.",
        )
    outlier_info = ", ".join(
        f"{o.file_date or '?'} ({format_size(o.size_bytes)})" for o in outliers[:5]
    )
    return CheckResult(
        "file_sizes",
        False,
        f"Outliers: {len(outliers)} arquivo(s) fora da média. "
        f"Média: {format_size(int(stats.mean_bytes))}. "
        f"Detalhes: {outlier_info}",
    )


def check_ucs_file(output_dir: Path) -> CheckResult:
    """Verifica se o arquivo auxiliar de UCs existe e está atualizado."""
    ucs_dir = output_dir / "raw" / "CIS" / "daily_cadastrados"
    if not ucs_dir.exists():
        return CheckResult(
            "ucs_file",
            False,
            f"Diretório não existe: {ucs_dir}",
        )

    ucs_files = list(ucs_dir.glob("ucs_*.parquet"))
    if not ucs_files:
        return CheckResult(
            "ucs_file",
            False,
            "Nenhum arquivo de UCs encontrado em daily_cadastrados/",
        )

    # Check if most recent is within MAX_UCS_AGE_DAYS
    dates = []
    for f in ucs_files:
        d = extract_date_from_filename(f.name)
        if d:
            dates.append(d)

    if not dates:
        return CheckResult("ucs_file", False, "Nenhum arquivo de UCs com data válida.")

    most_recent = max(dates)
    age = (date.today() - most_recent).days

    if age <= _MAX_UCS_AGE_DAYS:
        return CheckResult(
            "ucs_file",
            True,
            f"Arquivo mais recente: {most_recent.isoformat()} "
            f"({age} dias atrás, {len(ucs_files)} arquivo(s) total).",
        )
    return CheckResult(
        "ucs_file",
        False,
        f"Arquivo mais recente: {most_recent.isoformat()} "
        f"({age} dias atrás, máximo aceitável: {_MAX_UCS_AGE_DAYS} dias).",
    )


# --- Novas verificações para datasets normalizados ---


def _resolve_parquet_path(base_dir: Path, table: str, report_day: date | None = None) -> Path | None:
    """Tenta localizar o arquivo Parquet de uma tabela normalizada.

    Procura em ``base_dir/normalized/<table>/report_year=.../data.parquet``
    ou ``base_dir/<table>.parquet``.
    """
    from src.datasets.partitioning import partition_path

    if report_day is not None:
        p = partition_path(table, report_day, base_dir=base_dir / "normalized") / "data.parquet"
        if p.exists():
            return p

    # Fallback: procurar em normalized/ recursivamente
    norm_dir = base_dir / "normalized" / table
    if norm_dir.exists():
        for f in norm_dir.rglob("*.parquet"):
            return f

    # Fallback: arquivo direto
    direct = base_dir / f"{table}.parquet"
    return direct if direct.exists() else None


def check_dataset_schema(
    base_dir: Path,
    *,
    report_day: date | None = None,
) -> CheckResult:
    """Valida schemas dos datasets normalizados contra definições.

    Para cada tabela em ``_NORMALIZED_TABLES``, carrega o Parquet
    e verifica se todas as colunas obrigatórias existem e têm tipos
    compatíveis.

    Returns:
        CheckResult com detalhes de schemas válidos ou divergências.
    """
    import polars as pl

    issues: list[str] = []
    checked = 0

    for table in _NORMALIZED_TABLES:
        schema_def = get_schema(table)
        if schema_def is None:
            issues.append(f"Tabela '{table}' sem definição de schema registrada")
            continue

        file_path = _resolve_parquet_path(base_dir, table, report_day)
        if file_path is None:
            issues.append(f"Arquivo Parquet não encontrado para '{table}'")
            continue

        try:
            actual_schema = pl.read_parquet_schema(file_path)
        except Exception as exc:
            issues.append(f"Erro ao ler schema de '{table}': {exc}")
            continue

        checked += 1

        # Verificar colunas obrigatórias
        for field in schema_def:
            col_name = field.name
            if col_name not in actual_schema:
                issues.append(f"'{table}': coluna obrigatória '{col_name}' ausente")

        # Verificar tipos (compatibilidade básica)
        for field in schema_def:
            col_name = field.name
            if col_name not in actual_schema:
                continue
            expected_dtype = field.dtype
            actual_dtype = actual_schema[col_name]
            # Apenas reportar divergência maior (ex.: Float vs Int)
            if type(expected_dtype).__name__ != type(actual_dtype).__name__:
                issues.append(
                    f"'{table}'.{col_name}: esperado {expected_dtype}, "
                    f"encontrado {actual_dtype}"
                )

    if not issues:
        return CheckResult(
            "dataset_schema",
            True,
            f"Schemas válidos para {checked} tabela(s).",
        )
    return CheckResult(
        "dataset_schema",
        False,
        f"Problemas em {len(issues)} schema(s): " + "; ".join(issues[:10]),
    )


def check_dataset_keys(
    base_dir: Path,
    *,
    report_day: date | None = None,
) -> CheckResult:
    """Verifica unicidade e não-nulidade das chaves primárias.

    Para cada tabela, carrega os dados e checa:
    - Nenhuma chave nula
    - Nenhuma duplicata na grain definida

    Returns:
        CheckResult com resultado da verificação.
    """
    import polars as pl

    issues: list[str] = []

    for table in _NORMALIZED_TABLES:
        keys = get_keys(table)
        if not keys:
            continue

        file_path = _resolve_parquet_path(base_dir, table, report_day)
        if file_path is None:
            continue

        try:
            df = pl.read_parquet(file_path)
        except Exception as exc:
            issues.append(f"Erro ao ler '{table}': {exc}")
            continue

        if df.height == 0:
            continue

        # Verificar nulos nas chaves
        for key_col in keys:
            null_count = df.filter(pl.col(key_col).is_null()).height
            if null_count > 0:
                issues.append(
                    f"'{table}': coluna chave '{key_col}' tem {null_count} nulo(s)"
                )

        # Verificar duplicatas
        try:
            dup_count = df.group_by(keys).agg(pl.len().alias("_cnt")).filter(pl.col("_cnt") > 1).height
            if dup_count > 0:
                issues.append(
                    f"'{table}': {dup_count} duplicata(s) na grain ({', '.join(keys)})"
                )
        except Exception as exc:
            issues.append(f"Erro ao verificar duplicatas em '{table}': {exc}")

    if not issues:
        return CheckResult("dataset_keys", True, "Todas as chaves são únicas e não-nulas.")
    return CheckResult(
        "dataset_keys",
        False,
        "; ".join(issues[:10]),
    )


def check_dataset_ranges(
    base_dir: Path,
    *,
    report_day: date | None = None,
) -> CheckResult:
    """Valida faixas físicas de colunas numéricas.

    Verifica se valores estão dentro dos limites físicos esperados
    definidos em ``_PHYSICAL_RANGES``.

    Returns:
        CheckResult com valores fora da faixa.
    """
    import polars as pl

    violations: list[str] = []
    checked_cols = 0

    for table in _NORMALIZED_TABLES:
        file_path = _resolve_parquet_path(base_dir, table, report_day)
        if file_path is None:
            continue

        try:
            df = pl.read_parquet(file_path)
        except Exception:
            continue

        if df.height == 0:
            continue

        for col_name, (vmin, vmax, unit) in _PHYSICAL_RANGES.items():
            if col_name not in df.columns:
                continue

            checked_cols += 1
            col = df.get_column(col_name)
            if col.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                actual_min = col.min()
                actual_max = col.max()
                if actual_min is not None and actual_max is not None:
                    if actual_min < vmin or actual_max > vmax:
                        violations.append(
                            f"'{table}'.{col_name} [{actual_min:.2f}, {actual_max:.2f}] "
                            f"fora da faixa [{vmin}, {vmax}] {unit}"
                        )

    if not violations:
        return CheckResult(
            "dataset_ranges",
            True,
            f"Faixas físicas válidas para {checked_cols} colunas.",
        )
    return CheckResult(
        "dataset_ranges",
        False,
        "; ".join(violations[:10]),
    )


def check_temporal_order(
    base_dir: Path,
    *,
    report_day: date | None = None,
) -> CheckResult:
    """Verifica ordenação temporal e particionamento.

    Para tabelas com coluna TIMESTAMP_UTC, verifica:
    - Timestamps estão em ordem crescente dentro de cada NIO
    - Timestamps correspondem ao report_day se informado

    Returns:
        CheckResult com resultado da verificação.
    """
    import polars as pl

    issues: list[str] = []
    time_tables = ["meter_interval", "meter_instantaneous", "meter_register_snapshot"]

    for table in time_tables:
        file_path = _resolve_parquet_path(base_dir, table, report_day)
        if file_path is None:
            continue

        try:
            df = pl.read_parquet(file_path)
        except Exception:
            continue

        if df.height == 0:
            continue

        if "TIMESTAMP_UTC" not in df.columns:
            continue

        # Verificar se timestamps estão ordenados por NIO
        if "NIO" in df.columns:
            disordered = (
                df.sort("TIMESTAMP_UTC")
                .with_columns(
                    pl.col("TIMESTAMP_UTC")
                    .diff()
                    .cast(pl.Duration("ms"))
                    .dt.total_milliseconds()
                    .alias("_diff_ms")
                )
                .filter(pl.col("_diff_ms") < 0)
                .height
            )
            if disordered > 0:
                issues.append(f"'{table}': {disordered} timestamps fora de ordem")

        # Verificar se timestamps correspondem ao report_day
        if report_day is not None:
            date_col = df.get_column("TIMESTAMP_UTC").dt.date().unique()
            expected = {report_day}
            actual = set(date_col)
            if not actual.issubset(expected):
                extra = actual - expected
                issues.append(
                    f"'{table}': {len(extra)} data(s) não correspondem a {report_day}"
                )

    if not issues:
        return CheckResult(
            "temporal_order",
            True,
            "Ordenação temporal correta para todas as tabelas.",
        )
    return CheckResult(
        "temporal_order",
        False,
        "; ".join(issues[:10]),
    )


def check_datasets(
    base_dir: Path,
    *,
    report_day: date | None = None,
) -> list[CheckResult]:
    """Executa verificações de dataset em lote.

    Args:
        base_dir: Diretório raiz do output.
        report_day: Data de referência (opcional).

    Returns:
        Lista de CheckResult.
    """
    return [
        check_dataset_schema(base_dir, report_day=report_day),
        check_dataset_keys(base_dir, report_day=report_day),
        check_dataset_ranges(base_dir, report_day=report_day),
        check_temporal_order(base_dir, report_day=report_day),
    ]


def run_integrity_check(
    base_dir: Path,
    *,
    min_days_back: int = 30,
    check_normalized: bool = False,
    report_day: date | None = None,
) -> IntegrityReport:
    """Executa todas as verificações e retorna relatório consolidado.

    Args:
        base_dir: Diretório raiz do output.
        min_days_back: Número de dias para trás para verificar período.
        check_normalized: Se True, executa verificações de schema/chaves/ranges
                          nos datasets normalizados.
        report_day: Data de referência para datasets normalizados.
    """
    report = IntegrityReport()

    report.add(check_file_naming(base_dir))
    report.add(check_period(base_dir, min_days_back=min_days_back))
    report.add(check_sizes(base_dir))
    report.add(check_ucs_file(base_dir))

    if check_normalized:
        for result in check_datasets(base_dir, report_day=report_day):
            report.add(result)

    report.summary = {
        "base_dir": str(base_dir),
        "total_checks": len(report.checks),
        "passed_checks": sum(1 for c in report.checks if c.passed),
        "failed_checks": sum(1 for c in report.checks if not c.passed),
    }

    return report


def print_integrity_report(report: IntegrityReport) -> None:
    """Imprime relatório de integridade formatado."""
    status = "✅ PASS" if report.passed else "❌ FAIL"
    print(f"\nIntegrity Check: {status}")
    print(f"  {report.summary.get('passed_checks', 0)}/{report.summary.get('total_checks', 0)} checks passed\n")

    for check in report.checks:
        icon = "✓" if check.passed else "✗"
        print(f"  {icon} {check.name}: {check.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifica integridade dos arquivos diários da pipeline ARAUCARIA.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório raiz do output (default: output/).",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Número de dias para trás para verificar período (default: 30).",
    )
    parser.add_argument(
        "--check-normalized",
        action="store_true",
        help="Executa verificações de schema/chaves/ranges nos datasets normalizados.",
    )
    parser.add_argument(
        "--report-day",
        type=date.fromisoformat,
        default=None,
        help="Data de referência para datasets normalizados (AAAA-MM-DD).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Saída em formato JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_dir = args.base_dir.resolve()

    report = run_integrity_check(
        base_dir,
        min_days_back=args.days_back,
        check_normalized=args.check_normalized,
        report_day=args.report_day,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_integrity_report(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
