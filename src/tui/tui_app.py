"""Interactive Rich TUI for TCC Extraction, Dataset Generation & ML Modeling.

Built with Rich for high-aesthetic terminal interactions.
Provides a comprehensive menu to:
1. Generate datasets (Extraction & Pipeline execution for Feeders/CIS/GEO/MDM).
2. Train & Validate Models (Baseline, Composite Multi-Technique, Hierarchical MoE).
3. View Datalake Statistics & Partition Status.
4. Compare Expert Models vs Global Anchor Model on isolated Test datasets.
5. Run Data Lake Health Audits & Automated Tests.
6. Configure global execution parameters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from src.automation.daily_runner import DailyAutomationRunner
from src.datasets.dataset_splitter import DatasetSplitter
from src.models.composite import CompositeAnomalyDetector
from src.models.hierarchical_trainer import HierarchicalModelOrchestrator
from src.visualization.diagnostics import generate_diagnostic_plots

console = Console()


def check_uv_environment() -> bool:
    """Checks if running inside a uv-managed virtual environment."""
    is_uv = any(k.startswith("UV_") for k in os.environ) or "uv" in sys.executable.lower()
    in_venv = sys.prefix != sys.base_prefix
    return is_uv or in_venv


def show_environment_warning() -> None:
    """Displays a Rich Warning banner if not running via uv."""
    if not check_uv_environment():
        banner = Panel(
            Text(
                "⚠️ AVISO DE AMBIENTE RECOMENDADO:\n"
                "Você não está executando este pipeline através do `uv run main.py`.\n"
                "Para garantir reprodutibilidade e dependências isoladas, execute:\n\n"
                "    uv run main.py",
                justify="center",
                style="bold yellow",
            ),
            title="[bold red]Ambiente Não Isolado[/bold red]",
            border_style="yellow",
        )
        console.print(banner)


class RichPipelineTUI:
    """Rich Console & Interactive TUI Manager."""

    def __init__(self, base_dir: Path = Path("output")) -> None:
        self.base_dir = base_dir
        self.contamination = 0.05
        self.random_state = 42
        self.default_feeder = "Fonte_Nova"
        self.default_geo_id = 6352460

    @property
    def input_file(self) -> Path:
        return self.base_dir / "model_input" / "v1" / "training_dataset.parquet"

    @property
    def hierarchy_file(self) -> Path:
        return self.base_dir / "context" / "electrical_hierarchy.parquet"

    def render_header(self) -> None:
        header_text = Text(
            "⚡ TCC COPEL ARAUCÁRIA - EXTRAÇÃO, PIPELINE & DETECÇÃO DE ANOMALIAS ⚡",
            style="bold white on blue",
            justify="center",
        )
        console.print(Panel(header_text, border_style="blue", padding=(0, 1)))

    def generate_dataset(self) -> None:
        """Triggers dataset generation — full Araucária or single feeder."""
        console.print("\n[bold cyan]📥 OPÇÃO 1: GERAR DATASET & EXECUTAR PIPELINE DE EXTRAÇÃO[/bold cyan]\n")

        console.print("Escopo de Extração:")
        console.print("  [bold yellow][1][/bold yellow] 🌍 Araucária Completa (todos os alimentadores — recomendado para ML)")
        console.print("  [bold yellow][2][/bold yellow] 🔌 Alimentador específico (teste rápido ou re-extração pontual)")
        scope_choice = Prompt.ask("Escopo", choices=["1", "2"], default="1")

        days_back = IntPrompt.ask("Dias Atrás (Days Back)", default=1)
        no_cis_cache = Prompt.ask("Forçar re-extração do CIS (ignorar cache)?", choices=["s", "n"], default="n") == "s"

        feeder_name: str | None = None
        feeder_geo_id: int | None = None

        if scope_choice == "2":
            feeder_name = Prompt.ask("Nome do Alimentador", default=self.default_feeder)
            feeder_geo_id = IntPrompt.ask("ID GEO do Alimentador", default=self.default_geo_id)

        scope_label = f"Alimentador '{feeder_name}'" if feeder_name else "Araucária Completa"
        console.print(
            f"\n[bold green][*] Iniciando extração: {scope_label} "
            f"(days_back={days_back})...[/bold green]\n"
        )

        try:
            from scripts.run_araucaria_full import run_pipeline
            ret = run_pipeline(
                days_back=days_back,
                batch_size=500,
                no_cis_cache=no_cis_cache,
                feeder_name=feeder_name,
                feeder_geo_id=feeder_geo_id,
                max_workers=5,
                train_model=False,
            )
            if ret == 0:
                console.print(
                    Panel(
                        f"[bold green]✔ Dataset gerado com sucesso![/bold green]\n"
                        f"Escopo: [bold white]{scope_label}[/bold white]\n"
                        f"Arquivo: [underline]{self.input_file}[/underline]\n\n"
                        f"[dim]Use a Opção 2 para treinar modelos sobre este dataset.[/dim]",
                        title="Extração Concluída",
                        border_style="green",
                    )
                )
            else:
                console.print("[bold red][-] O pipeline retornou com erro. Verifique os logs acima.[/bold red]")
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Falha na execução do pipeline:[/bold red]\n{e}",
                    title="Erro",
                    border_style="red",
                )
            )

    def display_stats(self) -> None:
        """Displays datalake partitions and summary metrics."""
        console.print("\n[bold cyan]📊 OPÇÃO 3: ESTATÍSTICAS DO DATA LAKE & MODEL INPUT[/bold cyan]\n")

        table = Table(title="Resumo do Datalake & Datasets", border_style="cyan")
        table.add_column("Item / Camada", style="bold yellow")
        table.add_column("Status / Caminho", style="dim")
        table.add_column("Linhas", justify="right", style="green")
        table.add_column("Colunas", justify="right", style="blue")

        # Check training dataset
        if self.input_file.exists():
            df = pl.read_parquet(self.input_file)
            table.add_row("Model Input (v1)", str(self.input_file.name), f"{df.height:,}", f"{df.width:,}")
        else:
            table.add_row("Model Input (v1)", "Não Encontrado", "0", "0")

        # Check hierarchy
        if self.hierarchy_file.exists():
            hdf = pl.read_parquet(self.hierarchy_file)
            table.add_row("Electrical Hierarchy", str(self.hierarchy_file.name), f"{hdf.height:,}", f"{hdf.width:,}")
        else:
            table.add_row("Electrical Hierarchy", "Não Encontrado", "0", "0")

        # Check UC Context
        ctx_file = self.base_dir / "context" / "uc_context.parquet"
        if ctx_file.exists():
            cdf = pl.read_parquet(ctx_file)
            table.add_row("UC Context", str(ctx_file.name), f"{cdf.height:,}", f"{cdf.width:,}")

        console.print(table)

        if self.hierarchy_file.exists():
            hdf = pl.read_parquet(self.hierarchy_file)
            if "ALIMENTADOR" in hdf.columns:
                feeders = hdf.get_column("ALIMENTADOR").drop_nulls().unique().to_list()
                f_table = Table(title="Alimentadores Identificados na Hierarquia", border_style="magenta")
                f_table.add_column("Alimentador", style="bold green")
                f_table.add_column("Contagem de UCs", justify="right", style="cyan")
                for f in feeders:
                    cnt = hdf.filter(pl.col("ALIMENTADOR") == f).height
                    f_table.add_row(str(f), f"{cnt:,}")
                console.print(f_table)

    def train_models(self) -> None:
        """Trains models using strict Train/Val/Test isolation."""
        console.print("\n[bold cyan]🤖 OPÇÃO 2: TREINAR & VALIDAR MODELOS DE IA[/bold cyan]\n")

        if not self.input_file.exists():
            console.print(
                Panel(
                    f"[bold red]Dataset de treinamento não encontrado em {self.input_file}.[/bold red]\n"
                    "Por favor, selecione a [bold yellow]Opção 1[/bold yellow] para gerar o dataset primeiro.",
                    title="Erro de Pré-requisito",
                    border_style="red",
                )
            )
            return

        console.print("Modos de Treinamento Disponíveis:")
        console.print("  [1] Modelo Composto Multi-Técnica (IsoForest + LOF + ECOD + PCA + Regras)")
        console.print("  [2] Mistura de Especialistas (MoE Router)")
        console.print("  [3] Modelo Baseline (Isolation Forest + KMeans)")

        mode_choice = Prompt.ask("Escolha o modelo para treinamento", choices=["1", "2", "3"], default="1")
        contam = FloatPrompt.ask("Taxa de Contaminação (Contamination) [0.001 - 0.50]", default=self.contamination)
        if contam <= 0.0 or contam > 0.5:
            safe_contam = float(np.clip(contam, 0.001, 0.5))
            console.print(
                f"[bold yellow]⚠️ Taxa de contaminação deve estar no intervalo (0.0, 0.5]. "
                f"Ajustando de {contam} para {safe_contam:.3f}[/bold yellow]"
            )
            contam = safe_contam

        df = pl.read_parquet(self.input_file)
        hdf = pl.read_parquet(self.hierarchy_file) if self.hierarchy_file.exists() else None

        # Apply Dataset Splitter for zero-leakage isolation
        console.print("\n[bold green][*] Aplicando DatasetSplitter (Divisão inteligente por similaridade de alimentadores)...[/bold green]")
        splitter = DatasetSplitter(val_ratio=0.15, test_ratio=0.20, random_state=self.random_state)
        split_res = splitter.split(df, hierarchy_df=hdf)
        summary = split_res.split_summary

        split_table = Table(title="Divisão do Dataset (Sem Vazamento de Dados)", border_style="green")
        split_table.add_column("Conjunto", style="bold yellow")
        split_table.add_column("Amostras", justify="right", style="cyan")
        split_table.add_column("Proporção", justify="right", style="magenta")

        split_table.add_row("Treino (Train)", f"{summary['train_rows']:,}", f"{summary['train_ratio']:.1%}")
        split_table.add_row("Validação (Val)", f"{summary['val_rows']:,}", f"{summary['val_ratio']:.1%}")
        split_table.add_row("Teste (Test)", f"{summary['test_rows']:,}", f"{summary['test_ratio']:.1%}")
        console.print(split_table)

        feature_cols = [c for c in df.columns if c.startswith("x__") and df[c].dtype.is_numeric()]

        if mode_choice == "1":
            console.print("\n[bold yellow][*] Treinando Modelo Composto Multi-Técnica no conjunto de TREINO...[/bold yellow]")
            X_train = split_res.train_df.select(feature_cols).to_numpy()
            detector = CompositeAnomalyDetector(contamination=contam, random_state=self.random_state)
            detector.fit(X_train)

            console.print("[bold yellow][*] Avaliando performance estritamente no conjunto de TESTE...[/bold yellow]")
            X_test = split_res.test_df.select(feature_cols).to_numpy()
            preds = detector.predict_composite(X_test, raw_dataframe=split_res.test_df)

            n_anom = int(preds["is_anomaly"].sum())
            res_panel = Panel(
                f"[bold green]✔ Treinamento e Validação Concluídos![/bold green]\n\n"
                f"• Total de Amostras no Teste: [bold white]{split_res.test_df.height:,}[/bold white]\n"
                f"• Anomalias Identificadas:    [bold red]{n_anom:,} ({n_anom/max(1, split_res.test_df.height):.1%})[/bold red]\n"
                f"• Score Médio de Consenso:    [bold yellow]{preds['consensus_score'].mean():.4f}[/bold yellow]",
                title="Resultado do Teste - Modelo Composto",
                border_style="green",
            )
            console.print(res_panel)

            # Generate Diagnostic Visualizations
            eval_df = split_res.test_df.with_columns([
                pl.Series("consensus_score", preds["consensus_score"]),
                pl.Series("is_anomaly", preds["is_anomaly"]),
                pl.Series("iso_score", preds["iso_score"]),
                pl.Series("lof_score", preds["lof_score"]),
                pl.Series("ecod_score", preds["ecod_score"]),
                pl.Series("pca_score", preds["pca_score"]),
            ])
            plots = generate_diagnostic_plots(eval_df, output_dir=self.base_dir / "output" / "reports" / "figures", title_prefix="Modelo Composto")
            if plots:
                console.print("\n[bold cyan]📊 Gráficos Diagnósticos Gerados com Sucesso:[/bold cyan]")
                for p_name, p_path in plots.items():
                    console.print(f"  • {p_name}: [underline]{p_path}[/underline]")

        elif mode_choice == "2":
            console.print("\n[bold yellow][*] Executando Orchestrator MoE (Treinamento Hierárquico)...[/bold yellow]")
            orchestrator = HierarchicalModelOrchestrator(
                output_dir=self.base_dir,
                contamination=contam,
                random_state=self.random_state,
            )
            scores_df = orchestrator.train_and_score(split_res.train_df, hierarchy_df=hdf)
            console.print("[bold green]✔ Treinamento Hierárquico MoE Concluído![/bold green]")

            plots = generate_diagnostic_plots(scores_df, output_dir=self.base_dir / "output" / "reports" / "figures", title_prefix="Hierárquico MoE")
            if plots:
                console.print("\n[bold cyan]📊 Gráficos Diagnósticos Gerados com Sucesso:[/bold cyan]")
                for p_name, p_path in plots.items():
                    console.print(f"  • {p_name}: [underline]{p_path}[/underline]")

    def compare_experts(self) -> None:
        """Direct comparison between Feeder Experts vs Global Anchor Model."""
        console.print("\n[bold cyan]⚡ OPÇÃO 4: COMPARAR ESPECIALISTAS VS MODELO GLOBAL[/bold cyan]\n")

        if not self.input_file.exists():
            console.print("[bold red]Dataset de treinamento não encontrado.[/bold red]")
            return

        df = pl.read_parquet(self.input_file)
        hdf = pl.read_parquet(self.hierarchy_file) if self.hierarchy_file.exists() else None

        splitter = DatasetSplitter(random_state=self.random_state)
        split_res = splitter.split(df, hierarchy_df=hdf)

        feature_cols = [c for c in df.columns if c.startswith("x__") and df[c].dtype.is_numeric()]

        # Fit Global
        X_train = split_res.train_df.select(feature_cols).to_numpy()
        X_test = split_res.test_df.select(feature_cols).to_numpy()

        global_det = CompositeAnomalyDetector(contamination=self.contamination, random_state=self.random_state)
        global_det.fit(X_train)
        preds_global = global_det.predict_composite(X_test, raw_dataframe=split_res.test_df)

        comp_table = Table(title="Comparação de Desempenho no Conjunto de Teste", border_style="magenta")
        comp_table.add_column("Abordagem / Modelo", style="bold yellow")
        comp_table.add_column("Score Médio", justify="right", style="cyan")
        comp_table.add_column("Desvio Padrão", justify="right", style="blue")
        comp_table.add_column("Taxa de Anomalias", justify="right", style="red")

        comp_table.add_row(
            "Modelo Global Âncora",
            f"{preds_global['consensus_score'].mean():.4f}",
            f"{preds_global['consensus_score'].std():.4f}",
            f"{preds_global['is_anomaly'].sum() / max(1, len(X_test)):.1%}",
        )
        console.print(comp_table)

        # Generate diagnostic plots for compare_experts
        eval_df = split_res.test_df.with_columns([
            pl.Series("consensus_score", preds_global["consensus_score"]),
            pl.Series("is_anomaly", preds_global["is_anomaly"]),
            pl.Series("iso_score", preds_global["iso_score"]),
            pl.Series("lof_score", preds_global["lof_score"]),
            pl.Series("ecod_score", preds_global["ecod_score"]),
            pl.Series("pca_score", preds_global["pca_score"]),
        ])
        plots = generate_diagnostic_plots(eval_df, output_dir=self.base_dir / "output" / "reports" / "figures", title_prefix="Modelo Global Âncora")
        if plots:
            console.print("\n[bold cyan]📊 Gráficos Diagnósticos da Comparação Gerados com Sucesso:[/bold cyan]")
            for p_name, p_path in plots.items():
                console.print(f"  • {p_name}: [underline]{p_path}[/underline]")

    def run_health_audit(self) -> None:
        """Executes Data Lake health check."""
        console.print("\n[bold cyan]🩺 OPÇÃO 5: AUDITORIA DE SAÚDE DO DATA LAKE & TESTES[/bold cyan]\n")
        runner = DailyAutomationRunner(base_dir=self.base_dir)
        result = runner.run_daily_audit()

        status_style = "bold green" if result.get("global_status") == "HEALTHY" else "bold red"
        console.print(
            Panel(
                f"Status Global: [{status_style}]{result.get('global_status')}[/{status_style}]\n"
                f"Relatório gerado em: [underline]{self.base_dir / 'reports' / 'daily_health_report.md'}[/underline]",
                title="Resultado da Auditoria",
                border_style="cyan",
            )
        )

    def configure_settings(self) -> None:
        """Allows tuning global configuration parameters."""
        contam = FloatPrompt.ask("Taxa Padrão de Contaminação [0.001 - 0.50]", default=self.contamination)
        if contam <= 0.0 or contam > 0.5:
            safe_contam = float(np.clip(contam, 0.001, 0.5))
            console.print(
                f"[bold yellow]⚠️ Taxa de contaminação deve estar no intervalo (0.0, 0.5]. "
                f"Ajustando de {contam} para {safe_contam:.3f}[/bold yellow]"
            )
            contam = safe_contam
        self.contamination = contam
        self.random_state = IntPrompt.ask("Semente Aleatória (Random State)", default=self.random_state)
        console.print("[bold green]✔ Configurações salvas para a sessão atual.[/bold green]")

    def run_menu(self) -> None:
        """Main interactive loop."""
        self.render_header()
        show_environment_warning()

        while True:
            console.print("\n[bold white]Selecione uma opção do Menu Principal:[/bold white]")
            console.print("  [bold yellow][1][/bold yellow] 📥 Gerar Dataset (Pipeline de Extração CIS/GEO/MDM)")
            console.print("  [bold yellow][2][/bold yellow] 🤖 Treinar & Validar Modelos de IA (Composto / MoE / Baseline)")
            console.print("  [bold yellow][3][/bold yellow] 📊 Visualizar Estatísticas do Data Lake")
            console.print("  [bold yellow][4][/bold yellow] ⚡ Comparar Especialistas vs Modelo Global (No Teste)")
            console.print("  [bold yellow][5][/bold yellow] 🩺 Executar Auditoria de Saúde & Tests")
            console.print("  [bold yellow][6][/bold yellow] ⚙️ Configurações Globais")
            console.print("  [bold yellow][7][/bold yellow] 🚪 Sair")

            choice = Prompt.ask("\nDigite o número da opção desejada", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")

            if choice == "1":
                self.generate_dataset()
            elif choice == "2":
                self.train_models()
            elif choice == "3":
                self.display_stats()
            elif choice == "4":
                self.compare_experts()
            elif choice == "5":
                self.run_health_audit()
            elif choice == "6":
                self.configure_settings()
            elif choice == "7":
                console.print("\n[bold magenta]Encerrando a TUI. Até logo![/bold magenta]\n")
                break


def main_tui() -> None:
    tui = RichPipelineTUI()
    tui.run_menu()


if __name__ == "__main__":
    main_tui()
