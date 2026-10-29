"""Audit Voltage Disturbance Classifier against Deterministic PRODIST M8 Baseline.

Loads telemetry measurements, trains or loads the ML classifier, performs a fine-grained
row-by-row comparison against the deterministic baseline rules, diagnoses the exact source
of noise for each discrepancy, and registers error records into structured CSVs in output/reports/
for manual engineering review.

Usage:
    # Run full audit and export error CSVs
    uv run python scripts/audit_voltage_classifier.py

    # Filter by specific date range or feeder
    uv run python scripts/audit_voltage_classifier.py --start-date 2026-08-01 --end-date 2026-10-18 --feeder Fonte_Nova

    # Test with 100% deterministic rule alignment enabled
    uv run python scripts/audit_voltage_classifier.py --align-with-rules
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.models.voltage_classifier import (
    MLVoltageDisturbanceClassifier,
    NoiseSourceCategory,
    RuleBasedVoltageClassifier,
    VoltageDiscrepancyAuditor,
    extract_voltage_classifier_features,
    group_train_test_split,
)
from scripts.train_voltage_classifier import (
    load_alarms_data,
    load_instantaneous_data,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ML Voltage Classifier against Deterministic Baseline.")
    parser.add_argument(
        "--instantaneous-dir",
        type=Path,
        default=Path("output/measurements/ami_instantaneous"),
        help="Path to instantaneous telemetry files.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Filter telemetry starting from this date (AAAA-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Filter telemetry up to this date (AAAA-MM-DD).",
    )
    parser.add_argument(
        "--feeder",
        type=str,
        default=None,
        help="Optional: Filter telemetry exclusively to a specific electrical feeder.",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        default=Path("output/context/uc_context.parquet"),
        help="Path to uc_context.parquet.",
    )
    parser.add_argument(
        "--hierarchy-file",
        type=Path,
        default=Path("output/context/electrical_hierarchy.parquet"),
        help="Path to electrical_hierarchy.parquet.",
    )
    parser.add_argument(
        "--meter-history-file",
        type=Path,
        default=Path("output/context/meter_installation_history.parquet"),
        help="Path to meter_installation_history.parquet.",
    )
    parser.add_argument(
        "--alarms-dir",
        type=Path,
        default=Path("output/events"),
        help="Path to alarms directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum telemetry rows to sample (0 = use all available data, default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/reports"),
        help="Directory to save audit CSVs and reports (default: output/reports).",
    )
    parser.add_argument(
        "--align-with-rules",
        action="store_true",
        default=False,
        help="If set, activates 100% deterministic rule alignment on predictions.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def run_voltage_audit(
    *,
    instantaneous_dir: Path = Path("output/measurements/ami_instantaneous"),
    start_date: date | None = None,
    end_date: date | None = None,
    feeder: str | None = None,
    context_file: Path = Path("output/context/uc_context.parquet"),
    hierarchy_file: Path = Path("output/context/electrical_hierarchy.parquet"),
    meter_history_file: Path = Path("output/context/meter_installation_history.parquet"),
    alarms_dir: Path = Path("output/events"),
    max_samples: int = 0,
    output_dir: Path = Path("output/reports"),
    align_with_rules: bool = False,
    random_state: int = 42,
) -> int:
    """Executes full discrepancy audit and noise categorization, exporting error CSVs."""
    console.print(
        Panel(
            "⚡ [bold white]AUDITORIA DE RUÍDO & CADASTRO DE ERROS DE CLASSIFICAÇÃO DE TENSÃO (NÍVEL 1)[/bold white] ⚡\n"
            "[dim]Compara predições do modelo com a norma PRODIST Módulo 8 / IEEE 1159 e cadastra discrepâncias em CSV.[/dim]",
            border_style="cyan",
        )
    )

    # 1. Load context and hierarchy
    hierarchy_df = pl.read_parquet(hierarchy_file) if hierarchy_file.exists() else None
    context_df = pl.read_parquet(context_file) if context_file.exists() else None
    meter_history_df = pl.read_parquet(meter_history_file) if meter_history_file.exists() else None
    alarms_df = load_alarms_data(alarms_dir)

    # 2. Load telemetry measurements
    inst_df, _ = load_instantaneous_data(
        instantaneous_dir=instantaneous_dir,
        start_date=start_date,
        end_date=end_date,
        feeder_name=feeder,
        hierarchy_df=hierarchy_df,
        sample_quota_total=max_samples,
        random_state=random_state,
    )

    # Robust enrichment of UC by matching clean NIOs
    if not inst_df.is_empty():
        mapping_dfs = []
        for ref_df in [meter_history_df, hierarchy_df]:
            if ref_df is not None and not ref_df.is_empty() and "NIO" in ref_df.columns and "UC" in ref_df.columns:
                m = (
                    ref_df.select(["NIO", "UC"])
                    .drop_nulls()
                    .with_columns([
                        pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("CLEAN_NIO"),
                        pl.col("UC").cast(pl.String).str.strip_chars().alias("MAPPED_UC"),
                    ])
                    .filter(pl.col("CLEAN_NIO") != "")
                    .unique(subset=["CLEAN_NIO"])
                )
                mapping_dfs.append(m)

        if mapping_dfs and "NIO" in inst_df.columns:
            full_map = pl.concat(mapping_dfs).unique(subset=["CLEAN_NIO"])
            inst_clean = inst_df.with_columns(
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("CLEAN_NIO")
            )
            if "UC" in inst_clean.columns:
                inst_clean = inst_clean.drop("UC")
            inst_df = (
                inst_clean.join(full_map.select(["CLEAN_NIO", "MAPPED_UC"]), on="CLEAN_NIO", how="left")
                .with_columns(
                    pl.coalesce([pl.col("MAPPED_UC"), pl.col("CLEAN_NIO"), pl.col("NIO").cast(pl.String)]).alias("UC")
                )
                .drop(["CLEAN_NIO", "MAPPED_UC"])
            )
        elif "NIO" in inst_df.columns and "UC" not in inst_df.columns:
            inst_df = inst_df.with_columns(pl.col("NIO").cast(pl.String).alias("UC"))

    if inst_df.is_empty():
        console.print("[bold yellow]Aviso: Nenhuma telemetria encontrada no Data Lake para o filtro. Gerando benchmark sintético...[/bold yellow]")
        rows = []
        for rep in range(25):
            for i in range(8):
                rows.append({"UC": f"UC_NORM_{i}", "NIO": f"NIO_N_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 127.0, "U_L2": 127.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                rows.append({"UC": f"UC_SAG_{i}", "NIO": f"NIO_S_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 95.0, "U_L2": 96.0, "U_L3": 95.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                rows.append({"UC": f"UC_SWELL_{i}", "NIO": f"NIO_W_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 142.0, "U_L2": 143.0, "U_L3": 141.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                rows.append({"UC": f"UC_MONO_{i}", "NIO": f"NIO_M_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 127.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                rows.append({"UC": f"UC_BI_{i}", "NIO": f"NIO_B_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 0.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                rows.append({"UC": f"UC_BLACKOUT_{i}", "NIO": f"NIO_K_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 0.0, "U_L3": 0.0, "NEIGHBOR_COVERAGE_RATIO": 0.1})
                rows.append({"UC": f"UC_DESBAL_{i}", "NIO": f"NIO_D_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 127.0, "U_L2": 127.0, "U_L3": 118.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                # Inserir alguns pontos propositalmente limítrofes para demonstrar o auditor de ruído
                rows.append({"UC": f"UC_BORDER_SAG_{i}", "NIO": f"NIO_BS_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 114.2, "U_L2": 127.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                rows.append({"UC": f"UC_BORDER_IMB_{i}", "NIO": f"NIO_BI_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 127.0, "U_L2": 127.0, "U_L3": 123.1, "NEIGHBOR_COVERAGE_RATIO": 1.0})
        inst_df = pl.DataFrame(rows)

    console.print(f"[+] Total de medições carregadas: [bold white]{inst_df.height:,}[/bold white]")

    # 3. Extract features & baseline labels
    console.print("[*] Extraindo features físicas PRODIST Módulo 8 e rótulos normativos...")
    feat_df = extract_voltage_classifier_features(
        instantaneous_df=inst_df,
        context_df=context_df,
        alarms_df=alarms_df,
    )
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    # 4. Train or evaluate ML Model
    console.print("[*] Treinando modelo supervisionado para mapeamento de fronteiras difusas...")
    clf = MLVoltageDisturbanceClassifier(random_state=random_state, align_with_rules=align_with_rules)
    clf.fit(labeled_df)

    console.print("[*] Gerando predições e comparando com o baseline determinístico...")
    pred_res = clf.predict(labeled_df, align_with_rules=align_with_rules)

    # 5. Run Discrepancy Auditor
    auditor = VoltageDiscrepancyAuditor(output_dir=output_dir)
    audit_report = auditor.audit(pred_res.predictions_df, target_column="BASELINE_VOLTAGE_LABEL")

    # 6. Render Results in Rich Table
    console.print("\n")
    summary_table = Table(title="Resumo da Auditoria de Concordância e Ruído", border_style="green")
    summary_table.add_column("Métrica Analítica", style="bold cyan")
    summary_table.add_column("Valor Observado", justify="right", style="bold white")
    summary_table.add_column("Interpretação Operacional", style="dim")

    summary_table.add_row(
        "Total de Amostras Avaliadas",
        f"{audit_report.total_samples:,}",
        "Conjunto de medições de tensão",
    )
    summary_table.add_row(
        "Concordância com a Norma (Match)",
        f"{audit_report.matched_samples:,}",
        f"{audit_report.match_rate:.2%} de fidelidade normativa",
    )
    summary_table.add_row(
        "Discrepâncias / Divergências",
        f"{audit_report.discrepancy_count:,}",
        "Registradas para revisão manual em CSV",
    )
    summary_table.add_row(
        "Modo Alinhado 100% Determinístico",
        "ATIVADO" if align_with_rules else "DESATIVADO (Modo Puro de ML)",
        "Garante conformidade estrita na entrega" if align_with_rules else "Permite mapear o ruído natural da árvore",
    )
    console.print(summary_table)

    if audit_report.discrepancy_count > 0:
        console.print("\n[bold yellow]🔍 Fontes de Ruído Diagnosticadas nas Discrepâncias:[/bold yellow]\n")
        noise_table = Table(border_style="yellow")
        noise_table.add_column("Origem do Ruído", style="bold yellow")
        noise_table.add_column("Casos", justify="right", style="cyan")
        noise_table.add_column("Proporção", justify="right", style="magenta")
        noise_table.add_column("Diagnóstico Físico / Causa-Raiz", style="white")

        explanations_map = {
            NoiseSourceCategory.BORDERLINE_SUBTENSAO.value: "Tensão mínima na zona limítrofe de 0.90 p.u. (transição suave).",
            NoiseSourceCategory.BORDERLINE_SOBRETENSAO.value: "Tensão máxima na zona limítrofe de 1.05 p.u. (transição suave).",
            NoiseSourceCategory.BORDERLINE_DESBALANCEAMENTO.value: "Desbalanceamento PVUR próximo do limiar crítico regulatório de 3.0%.",
            NoiseSourceCategory.ZERO_VOLTAGE_CUTOFF.value: "Fase em patamar limítrofe do corte físico de interrupção (<= 5.0V).",
            NoiseSourceCategory.PRECEDENCE_CONFLICT.value: "Múltiplas perturbações simultâneas na onda (prioridade normativa).",
            NoiseSourceCategory.CLASS_WEIGHT_BIAS.value: "Superestimação de classe minoritária rara induzida por pesos.",
            NoiseSourceCategory.PROBABILITY_UNCERTAINTY.value: "Incerteza alta do classificador na região (confiança < 60%).",
            NoiseSourceCategory.TELEMETRY_OUTAGE_AMBIGUITY.value: "Ambiguidade entre perda de comunicação e queda física.",
            NoiseSourceCategory.UNCLASSIFIED_DIVERGENCE.value: "Divergência não categorizada; candidata para revisão manual.",
        }

        for k, v in sorted(audit_report.noise_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = v / max(1, audit_report.discrepancy_count)
            expl = explanations_map.get(k, "Revisão manual prioritária.")
            noise_table.add_row(k, f"{v:,}", f"{pct:.1%}", expl)

        console.print(noise_table)

        # Show first 5 discrepancy records for immediate feedback
        console.print("\n[bold cyan]📋 Amostra dos Primeiros Casos Cadastrados para Revisão Manual:[/bold cyan]")
        sample_table = Table(border_style="cyan")
        sample_table.add_column("UC", style="bold white")
        sample_table.add_column("Norma PRODIST", style="bold green")
        sample_table.add_column("Predição ML", style="bold red")
        sample_table.add_column("V_MIN_PU", justify="right", style="yellow")
        sample_table.add_column("V_MAX_PU", justify="right", style="yellow")
        sample_table.add_column("Desbalanço", justify="right", style="magenta")
        sample_table.add_column("Origem do Ruído", style="bold yellow")

        sample_rows = audit_report.discrepancies_df.head(5).to_dicts()
        for s in sample_rows:
            v_min_s = f"{float(s['V_MIN_PU']):.3f}" if s.get("V_MIN_PU") is not None else "-"
            v_max_s = f"{float(s['V_MAX_PU']):.3f}" if s.get("V_MAX_PU") is not None else "-"
            imb_s = f"{float(s['VOLTAGE_IMBALANCE_RATIO']):.1%}" if s.get("VOLTAGE_IMBALANCE_RATIO") is not None else "-"
            sample_table.add_row(
                str(s.get("UC", "-")),
                str(s.get("BASELINE_RULE_LABEL", "-")),
                str(s.get("ML_PREDICTED_LABEL", "-")),
                v_min_s,
                v_max_s,
                imb_s,
                str(s.get("NOISE_SOURCE", "-")),
            )
        console.print(sample_table)

    console.print("\n" + "=" * 65)
    console.print(f"[bold green]✔ Cadastro de Erros e Auditoria Concluídos com Sucesso![/bold green]")
    if audit_report.export_csv_path:
        console.print(f"[+] Arquivo CSV de Erros Cadastrados: [underline]{audit_report.export_csv_path}[/underline]")
        console.print(f"[+] Arquivo CSV Mais Recente (Pointer): [underline]{output_dir / 'voltage_classification_errors_latest.csv'}[/underline]")
    if audit_report.export_report_path:
        console.print(f"[+] Relatório Técnico Markdown:       [underline]{audit_report.export_report_path}[/underline]")
    console.print("=" * 65)

    return 0


def main() -> None:
    args = parse_args()
    ret = run_voltage_audit(
        instantaneous_dir=args.instantaneous_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        feeder=args.feeder,
        context_file=args.context_file,
        hierarchy_file=args.hierarchy_file,
        meter_history_file=args.meter_history_file,
        alarms_dir=args.alarms_dir,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        align_with_rules=args.align_with_rules,
        random_state=args.random_state,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
