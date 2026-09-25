"""Daily Automation Runner and Health Reporting Engine.

Orchestrates daily operational tasks:
1. Environment & Database Pre-flight Verification
2. Data Lake Partition Health Audit
3. Generation of Markdown & JSON Daily Health Reports
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.automation.health_check import DataLakeHealthChecker


class DailyAutomationRunner:
    """Orchestrates daily audits, pipeline validation, and health report publication."""

    def __init__(self, base_dir: Path | str = "output"):
        self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.checker = DataLakeHealthChecker(base_dir=self.base_dir)

    def run_daily_audit(self) -> dict[str, Any]:
        """Execute full daily health audit and write diagnostic report."""
        audit_result = self.checker.run_full_health_audit()

        # Build Markdown Report
        ts = audit_result["audit_timestamp"]
        g_status = audit_result["global_status"]

        status_badge = {
            "HEALTHY": "🟢 **SAUDÁVEL (HEALTHY)**",
            "WARNING": "🟡 **ATENÇÃO (WARNING)**",
            "INCOMPLETE": "⚪ **INCOMPLETO (DADOS PARCIAIS)**",
            "CRITICAL": "🔴 **CRÍTICO (FALHA DETECTADA)**",
        }.get(g_status, g_status)

        lines = [
            f"# Relatório Diário de Saúde do Pipeline (Data Lake AMI)",
            f"",
            f"- **Data/Hora da Auditoria:** `{ts}`",
            f"- **Status Geral:** {status_badge}",
            f"",
            f"---",
            f"",
            f"## 1. Camada de Medições (Phenomenon Layer)",
            f"",
            f"| Tabela | Partição Recente | Registros | Medidores Únicos | Tensão Média | Status |",
            f"|:---|:---|:---|:---|:---|:---|",
        ]

        for t_name, t_info in audit_result["measurements"].items():
            f_name = t_info.get("partition_file", "N/A")
            rows = t_info.get("total_rows", 0)
            meters = t_info.get("unique_meters", 0)
            v_mean = f"{t_info.get('voltage_mean', 0.0):.1f} V" if "voltage_mean" in t_info else "N/A"
            st = t_info.get("status", "UNKNOWN")
            lines.append(f"| `{t_name}` | `{f_name}` | {rows:,} | {meters:,} | {v_mean} | `{st}` |")

        lines.extend([
            f"",
            f"## 2. Camada Cadastral e Topológica (Context Layer)",
            f"",
            f"| Tabela de Contexto | Presente | Total de Linhas | Colunas | Status |",
            f"|:---|:---:|:---|:---|:---|",
        ])

        for c_name, c_info in audit_result["context"].items():
            exists = "Sim" if c_info.get("exists") else "Não"
            rows = f"{c_info.get('rows', 0):,}" if c_info.get("exists") else "N/A"
            cols = str(c_info.get("columns", "N/A"))
            st = c_info.get("status", "N/A")
            lines.append(f"| `{c_name}` | {exists} | {rows} | {cols} | `{st}` |")

        lines.extend([
            f"",
            f"---",
            f"*Relatório gerado automaticamente por `DailyAutomationRunner`.*",
        ])

        report_md = "\n".join(lines)

        date_token = datetime.now().strftime("%Y%m%d")
        md_path = self.reports_dir / f"daily_health_report_{date_token}.md"
        json_path = self.reports_dir / f"daily_health_report_{date_token}.json"

        md_path.write_text(report_md, encoding="utf-8")
        json_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")

        # Also write latest symlink/file
        (self.reports_dir / "daily_health_report_latest.md").write_text(report_md, encoding="utf-8")

        print(f"[+] Daily health report generated:\n    - {md_path}\n    - {json_path}")
        return audit_result
