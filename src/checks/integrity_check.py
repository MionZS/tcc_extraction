"""integrity_check.py — Verificação de integridade da pipeline diária.

Script autônomo que verifica:
- Nomes de arquivos válidos
- Período completo (sem lacunas)
- Tamanhos de arquivo dentro da faixa esperada
- Arquivo auxiliar de UCs existe
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from src.checks.daily_file_naming import validate_file_naming, extract_date_from_filename
from src.checks.period_coverage import get_file_dates, check_period_coverage
from src.checks.file_size_analysis import collect_file_sizes, compute_size_stats, detect_size_outliers, format_size


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


def run_integrity_check(
    base_dir: Path,
    *,
    min_days_back: int = 30,
) -> IntegrityReport:
    """Executa todas as verificações e retorna relatório consolidado.

    Args:
        base_dir: Diretório raiz do output.
        min_days_back: Número de dias para trás para verificar período.
    """
    report = IntegrityReport()

    report.add(check_file_naming(base_dir))
    report.add(check_period(base_dir, min_days_back=min_days_back))
    report.add(check_sizes(base_dir))
    report.add(check_ucs_file(base_dir))

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
        "--json",
        action="store_true",
        help="Saída em formato JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_dir = args.base_dir.resolve()

    report = run_integrity_check(base_dir, min_days_back=args.days_back)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_integrity_report(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
