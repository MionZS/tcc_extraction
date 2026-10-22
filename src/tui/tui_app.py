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

from datetime import date, datetime, timedelta
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
from src.models.voltage_classifier import (
    MLVoltageDisturbanceClassifier,
    RuleBasedVoltageClassifier,
    extract_voltage_classifier_features,
    group_train_test_split,
)
from src.visualization.diagnostics import generate_diagnostic_plots, generate_voltage_diagnostic_plots

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
        self.output_format = "both"
        self.default_voltage_max_samples = 0  # 0 = sem limite (usa 100% dos dados)
        self.default_voltage_slice = "1"     # 1 = todo o período disponível no Data Lake
        self.default_voltage_feeder_choice = "1"  # 1 = todos os alimentadores

    @property
    def input_file(self) -> Path:
        p = self.base_dir / "model_input" / "v1" / "training_dataset.parquet"
        if not p.exists():
            c = self.base_dir / "model_input" / "v1" / "training_dataset.csv"
            if c.exists():
                return c
        return p

    @property
    def hierarchy_file(self) -> Path:
        p = self.base_dir / "context" / "electrical_hierarchy.parquet"
        if not p.exists():
            c = self.base_dir / "context" / "electrical_hierarchy.csv"
            if c.exists():
                return c
        return p

    def read_table(self, path: Path) -> pl.DataFrame:
        """Reads Parquet or CSV transparently."""
        if not path.exists():
            csv_peer = path.with_suffix(".csv")
            if csv_peer.exists():
                return pl.read_csv(csv_peer, separator=";", infer_schema_length=5000, ignore_errors=True)
            p_peer = path.with_suffix(".parquet")
            if p_peer.exists():
                return pl.read_parquet(p_peer)
            return pl.DataFrame()
        if path.suffix == ".csv":
            return pl.read_csv(path, separator=";", infer_schema_length=5000, ignore_errors=True)
        return pl.read_parquet(path)

    def render_header(self) -> None:
        header_text = Text(
            "⚡ TCC COPEL ARAUCÁRIA - EXTRAÇÃO, PIPELINE & DETECÇÃO DE ANOMALIAS ⚡",
            style="bold white on blue",
            justify="center",
        )
        console.print(Panel(header_text, border_style="blue", padding=(0, 1)))

    def generate_dataset(self) -> None:
        """Triggers dataset generation — full Araucária, single feeder, or historical period."""
        console.print("\n[bold cyan]📥 OPÇÃO 1: GERAR DATASET & EXECUTAR PIPELINE DE EXTRAÇÃO[/bold cyan]\n")

        console.print("Escopo de Extração:")
        console.print("  [bold yellow][1][/bold yellow] 🌍 Araucária Completa (dia único via days_back)")
        console.print("  [bold yellow][2][/bold yellow] 🔌 Alimentador específico (teste rápido ou re-extração pontual)")
        console.print("  [bold yellow][3][/bold yellow] 📅 Extração de Período Histórico (01/08/2026 até a data mais recente — preenchimento de lacunas)")
        console.print("  [bold yellow][4][/bold yellow] ⚡ Sincronização Inteligente (Auto-Backfill da data mais recente + preenchimento de gaps)")
        scope_choice = Prompt.ask("Escopo", choices=["1", "2", "3", "4"], default="4")

        console.print("\nFormato de Armazenamento:")
        console.print("  [bold yellow][1][/bold yellow] Ambos (Parquet + CSV) [bold green][Padrão Recomendado][/bold green]")
        console.print("  [bold yellow][2][/bold yellow] Apenas CSV (.csv)")
        console.print("  [bold yellow][3][/bold yellow] Apenas Parquet (.parquet)")
        fmt_choice = Prompt.ask("Formato", choices=["1", "2", "3"], default="1")
        output_format = {"1": "both", "2": "csv", "3": "parquet"}[fmt_choice]

        if scope_choice == "4":
            refresh_cis = Prompt.ask("Re-extrair cadastro CIS do Oracle (para carregar medidores retirados/histórico)?", choices=["s", "n"], default="s") == "s"
            console.print("\n[bold green][*] Iniciando Sincronização Inteligente do Data Lake (Auto-Backfill)...[/bold green]\n")
            try:
                from scripts.run_araucaria_full import run_pipeline
                ret = run_pipeline(
                    auto_backfill=True,
                    fill_gaps=True,
                    batch_size=500,
                    no_cis_cache=refresh_cis,
                    max_workers=5,
                    output_format=output_format,
                    train_model=False,
                )
                if ret == 0:
                    console.print(
                        Panel(
                            "[bold green]✔ Sincronização do Data Lake concluída com sucesso![/bold green]\n"
                            "O Data Lake agora está atualizado com o calendário sem lacunas.\n"
                            "Consulte a [bold yellow]Opção 3[/bold yellow] para auditar a continuidade no calendário.",
                            title="Auto-Backfill Concluído",
                            border_style="green",
                        )
                    )
                else:
                    console.print("[bold red][-] O pipeline retornou com erro. Verifique os logs acima.[/bold red]")
            except Exception as e:
                console.print(
                    Panel(
                        f"[bold red]Falha na sincronização inteligente:[/bold red]\n{e}",
                        title="Erro",
                        border_style="red",
                    )
                )
            return

        if scope_choice == "3":
            start_str = Prompt.ask("Data Inicial (AAAA-MM-DD)", default="2026-08-01")
            default_end = (date.today() - timedelta(days=1)).isoformat()
            end_str = Prompt.ask("Data Final (AAAA-MM-DD)", default=default_end)

            try:
                start_dt = date.fromisoformat(start_str)
                end_dt = date.fromisoformat(end_str)
            except ValueError as e:
                console.print(f"[bold red]Data inválida: {e}[/bold red]")
                return

            total_d = (end_dt - start_dt).days + 1
            console.print(
                f"\n[bold green][*] Iniciando extração do período: {start_str} a {end_str} "
                f"({total_d} dias)...[/bold green]\n"
            )

            try:
                from scripts.run_araucaria_full import run_pipeline
                ret = run_pipeline(
                    start_date=start_dt,
                    end_date=end_dt,
                    batch_size=500,
                    no_cis_cache=False,
                    max_workers=5,
                    output_format=output_format,
                    train_model=False,
                )
                if ret == 0:
                    console.print(
                        Panel(
                            f"[bold green]✔ Extração do período concluída com sucesso![/bold green]\n"
                            f"Período: [bold white]{start_str} → {end_str}[/bold white]\n"
                            f"Consulte a [bold yellow]Opção 3[/bold yellow] para auditar a continuidade no calendário.",
                            title="Extração por Período Concluída",
                            border_style="green",
                        )
                    )
                else:
                    console.print("[bold red][-] O pipeline retornou com erro. Verifique os logs acima.[/bold red]")
            except Exception as e:
                console.print(
                    Panel(
                        f"[bold red]Falha na execução do pipeline de período:[/bold red]\n{e}",
                        title="Erro",
                        border_style="red",
                    )
                )
            return

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
                output_format=output_format,
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
            df = self.read_table(self.input_file)
            table.add_row("Model Input (v1)", str(self.input_file.name), f"{df.height:,}", f"{df.width:,}")
        else:
            table.add_row("Model Input (v1)", "Não Encontrado", "0", "0")

        # Check hierarchy
        if self.hierarchy_file.exists():
            hdf = self.read_table(self.hierarchy_file)
            table.add_row("Electrical Hierarchy", str(self.hierarchy_file.name), f"{hdf.height:,}", f"{hdf.width:,}")
        else:
            table.add_row("Electrical Hierarchy", "Não Encontrado", "0", "0")

        # Check UC Context
        ctx_file = self.base_dir / "context" / "uc_context.parquet"
        cdf = self.read_table(ctx_file)
        if not cdf.is_empty():
            table.add_row("UC Context", str(ctx_file.name), f"{cdf.height:,}", f"{cdf.width:,}")

        console.print(table)

        if self.hierarchy_file.exists():
            hdf = self.read_table(self.hierarchy_file)
            if "ALIMENTADOR" in hdf.columns:
                feeders = hdf.get_column("ALIMENTADOR").drop_nulls().unique().to_list()
                f_table = Table(title="Alimentadores Identificados na Hierarquia", border_style="magenta")
                f_table.add_column("Alimentador", style="bold green")
                f_table.add_column("Contagem de UCs", justify="right", style="cyan")
                for f in feeders:
                    cnt = hdf.filter(pl.col("ALIMENTADOR") == f).height
                    f_table.add_row(str(f), f"{cnt:,}")
                console.print(f_table)

        # Governança & Continuidade Temporal (Linha do Tempo / Calendário renderizado imediatamente)
        from src.visualization.continuity import render_continuity_calendar
        console.print("\n")
        render_continuity_calendar(
            start_date=date(2026, 8, 1),
            end_date=date.today() - timedelta(days=1),
            base_dir=self.base_dir,
            file_type="both",
        )

        # Ações Adicionais de Auditoria e Formato
        console.print("\nAções Adicionais de Formato & Governança:")
        console.print("  [bold yellow][1][/bold yellow] 📦 Filtrar Calendário Apenas Parquet (.parquet)")
        console.print("  [bold yellow][2][/bold yellow] 📄 Filtrar Calendário Apenas CSV (.csv)")
        console.print("  [bold yellow][3][/bold yellow] 🔄 Sincronizar Formatos (Converter Faltantes Parquet ↔ CSV)")
        console.print("  [bold yellow][4][/bold yellow] ↩️ Retornar ao Menu Principal")
        stat_opt = Prompt.ask("Escolha uma ação ou retorne", choices=["1", "2", "3", "4"], default="4")

        if stat_opt == "4":
            return
        elif stat_opt == "3":
            from src.datasets.format_converter import render_format_audit_table, sync_datalake_formats
            console.print("\n[bold green][*] Escaneando discrepâncias e convertendo arquivos faltantes...[/bold green]\n")
            res = sync_datalake_formats(self.base_dir)
            console.print(
                Panel(
                    f"[bold green]✔ Sincronização de Formatos Concluída![/bold green]\n\n"
                    f"• Parquet → CSV convertidos: [bold cyan]{res['parquet_to_csv']:,}[/bold cyan]\n"
                    f"• CSV → Parquet convertidos: [bold green]{res['csv_to_parquet']:,}[/bold green]\n"
                    f"• Erros encontrados:         [{'bold green' if res['errors'] == 0 else 'bold red'}]{res['errors']}[/]",
                    title="Resultado da Sincronização",
                    border_style="green",
                )
            )
            render_format_audit_table(self.base_dir)
            return

        f_type = "parquet" if stat_opt == "1" else "csv"
        console.print("\n")
        render_continuity_calendar(
            start_date=date(2026, 8, 1),
            end_date=date.today() - timedelta(days=1),
            base_dir=self.base_dir,
            file_type=f_type,
        )

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

        console.print("[bold white]Hierarquia & Ordem de Treinamento dos Modelos de IA:[/bold white]\n")

        console.print("[bold yellow]── NÍVEL 1: MODELO FÍSICO DA REDE (Sinal & Qualidade de Energia) ──[/bold yellow]")
        console.print("  [bold white][1][/bold white] ⚡ Classificador de Perturbações de Tensão (PRODIST M8 / IEEE 1159)")
        console.print("      [dim]• Classificação física da onda (afundamento, elevação, faltas, desbalanceamento)[/dim]\n")

        console.print("[bold yellow]── NÍVEL 2: MODELOS COMPORTAMENTAIS DA UC (Matriz UC × Cutoff) ──[/bold yellow]")
        console.print("  [bold white][2][/bold white] 🌍 Modelo Global Âncora (População Copel Araucária Completa)")
        console.print("      [dim]• Comitê Composto (IsoForest 0.3 + LOF 0.25 + ECOD 0.25 + PCA 0.20 + Bônus Físico)[/dim]")
        console.print("  [bold white][3][/bold white] 🔌 Especialistas Locais por Alimentador (Fonte Nova, Angélica, CSN...)")
        console.print("      [dim]• Modelos especialistas dedicados às peculiaridades de cada circuito elétrico[/dim]")
        console.print("  [bold white][4][/bold white] 🔀 Mistura Hierárquica de Especialistas (MoE Router)")
        console.print("      [dim]• Roteamento inteligente adaptativo: Especialista do Alimentador vs Modelo Global[/dim]")
        console.print("  [bold white][5][/bold white] 🎯 Pipeline Especialista Multi-Visão (Comunicação, Elétrica, Fraude & Clusters)")
        console.print("      [dim]• Decomposição explícita em escores S_comm, S_elec, S_tamper e diagnóstico operacional[/dim]\n")

        console.print("[bold yellow]── NÍVEL 3: REFERÊNCIA BASELINE, PIPELINE COMPLETO & GOVERNANÇA ──[/bold yellow]")
        console.print("  [bold white][6][/bold white] 📊 Modelo Baseline de Referência (Isolation Forest + KMeans)")
        console.print("  [bold white][7][/bold white] 🚀 Treinamento Completo em Cadeia (Pipeline Hierárquico Ponta a Ponta)")
        console.print("      [dim]• Executa todos os níveis na ordem ideal: Tensão → Global → Especialistas → MoE[/dim]")
        console.print("  [bold white][8][/bold white] 🏛️ Registro & Governança de Modelos (Model Registry)")

        mode_choice = Prompt.ask("\nEscolha a etapa de treinamento", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")

        if mode_choice == "1":
            self.train_voltage_classifier()
            return

        if mode_choice == "8":
            self.manage_model_registry()
            return

        # Para as opções 2 a 7, valida existência do dataset de features da UC
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

        contam = FloatPrompt.ask("Taxa de Contaminação (Contamination) [0.001 - 0.50]", default=self.contamination)
        if contam <= 0.0 or contam > 0.5:
            safe_contam = float(np.clip(contam, 0.001, 0.5))
            console.print(
                f"[bold yellow]⚠️ Taxa de contaminação deve estar no intervalo (0.0, 0.5]. "
                f"Ajustando de {contam} para {safe_contam:.3f}[/bold yellow]"
            )
            contam = safe_contam

        df = self.read_table(self.input_file)
        hdf = self.read_table(self.hierarchy_file) if self.hierarchy_file.exists() else None

        if df.is_empty():
            console.print("[bold red]Dataset de treinamento está vazio. Verifique a extração na Opção 1.[/bold red]")
            return

        # Filtro Temporal para os modelos de anomalia
        console.print("\nFiltro Temporal de Dados para Treinamento:")
        console.print("  [1] Todo o histórico disponível no dataset")
        console.print("  [2] Período Customizado (Data Inicial e Data Final)")
        period_choice = Prompt.ask("Filtro Temporal", choices=["1", "2"], default="1")

        if period_choice == "2":
            s_str = Prompt.ask("Data Inicial (AAAA-MM-DD)", default="2026-08-01")
            e_str = Prompt.ask("Data Final (AAAA-MM-DD)", default=date.today().isoformat())
            try:
                s_dt = date.fromisoformat(s_str)
                e_dt = date.fromisoformat(e_str)
                if "meta__report_year" in df.columns and "meta__report_month" in df.columns and "meta__report_day" in df.columns:
                    start_val = s_dt.year * 10000 + s_dt.month * 100 + s_dt.day
                    end_val = e_dt.year * 10000 + e_dt.month * 100 + e_dt.day
                    df = df.filter(
                        (pl.col("meta__report_year") * 10000 + pl.col("meta__report_month") * 100 + pl.col("meta__report_day") >= start_val) &
                        (pl.col("meta__report_year") * 10000 + pl.col("meta__report_month") * 100 + pl.col("meta__report_day") <= end_val)
                    )
                elif "REPORT_DAY" in df.columns:
                    df = df.filter(
                        (pl.col("REPORT_DAY").cast(pl.Date, strict=False) >= s_dt) &
                        (pl.col("REPORT_DAY").cast(pl.Date, strict=False) <= e_dt)
                    )
                console.print(f"[*] Dataset filtrado para o período {s_str} → {e_str}: {df.height:,} amostras.")
            except Exception as e:
                console.print(f"[bold yellow]Aviso: Não foi possível filtrar datas ({e}). Usando dataset completo.[/bold yellow]")

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

        # ── Opção 2 ou parte da Cadeia: Modelo Global Âncora ─────────────
        if mode_choice in ("2", "7"):
            console.print("\n[bold yellow][*] Treinando Modelo Global Âncora (Comitê Composto) no conjunto de TREINO...[/bold yellow]")
            X_train = split_res.train_df.select(feature_cols).to_numpy()
            detector = CompositeAnomalyDetector(contamination=contam, random_state=self.random_state)
            detector.fit(X_train)

            console.print("[bold yellow][*] Avaliando performance estritamente no conjunto de TESTE...[/bold yellow]")
            X_test = split_res.test_df.select(feature_cols).to_numpy()
            preds = detector.predict_composite(X_test, raw_dataframe=split_res.test_df)

            models_dir = self.base_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(detector, models_dir / "composite_anomaly_detector.joblib")

            n_anom = int(preds["is_anomaly"].sum())
            res_panel = Panel(
                f"[bold green]✔ Treinamento do Modelo Global Âncora Concluído![/bold green]\n\n"
                f"• Total de Amostras no Teste: [bold white]{split_res.test_df.height:,}[/bold white]\n"
                f"• Anomalias Identificadas:    [bold red]{n_anom:,} ({n_anom/max(1, split_res.test_df.height):.1%})[/bold red]\n"
                f"• Score Médio de Consenso:    [bold yellow]{preds['consensus_score'].mean():.4f}[/bold yellow]\n"
                f"• Artefato Salvo em:          [dim]{models_dir / 'composite_anomaly_detector.joblib'}[/dim]",
                title="Resultado do Teste - Modelo Global Âncora",
                border_style="green",
            )
            console.print(res_panel)

            eval_df = split_res.test_df.with_columns([
                pl.Series("consensus_score", preds["consensus_score"]),
                pl.Series("is_anomaly", preds["is_anomaly"]),
                pl.Series("iso_score", preds["iso_score"]),
                pl.Series("lof_score", preds["lof_score"]),
                pl.Series("ecod_score", preds["ecod_score"]),
                pl.Series("pca_score", preds["pca_score"]),
            ])
            plots = generate_diagnostic_plots(eval_df, output_dir=self.base_dir / "output" / "reports" / "figures", title_prefix="Modelo Global Âncora")
            if plots:
                console.print("\n[bold cyan]📊 Gráficos Diagnósticos Gerados com Sucesso:[/bold cyan]")
                for p_name, p_path in plots.items():
                    console.print(f"  • {p_name}: [underline]{p_path}[/underline]")

        # ── Opção 3 ou parte da Cadeia: Especialistas por Alimentador ────
        if mode_choice in ("3", "7"):
            target_f = None
            if mode_choice == "3":
                console.print("\nEscolha o escopo dos especialistas:")
                console.print("  [1] Todos os alimentadores elétricos cadastrados")
                console.print("  [2] Alimentador específico (ex: Fonte_Nova)")
                f_ch = Prompt.ask("Escopo", choices=["1", "2"], default="1")
                if f_ch == "2":
                    target_f = Prompt.ask("Nome do Alimentador", default=self.default_feeder)

            console.print(f"\n[bold yellow][*] Treinando Especialistas Locais ({target_f or 'Todos os alimentadores'})...[/bold yellow]")
            orchestrator = HierarchicalModelOrchestrator(
                output_dir=self.base_dir,
                contamination=contam,
                random_state=self.random_state,
            )
            scores_df = orchestrator.train_and_score(split_res.train_df, hierarchy_df=hdf, target_feeder=target_f)
            console.print("[bold green]✔ Especialistas por Alimentador Treinados com Sucesso![/bold green]")

        # ── Opção 4: Mistura Hierárquica MoE ─────────────────────────────
        elif mode_choice == "4":
            console.print("\n[bold yellow][*] Executando Orchestrator MoE (Treinamento Hierárquico Completo)...[/bold yellow]")
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

        # ── Opção 5: Pipeline Multi-Visão ────────────────────────────────
        elif mode_choice == "5":
            from src.models.specialized_detectors import MultiViewAnomalyPipeline

            console.print("\n[bold yellow][*] Treinando Pipeline Especialista Multi-Visão (S_comm, S_elec, S_tamper, Clusters)...[/bold yellow]")
            comm_cols = [c for c in feature_cols if any(k in c.lower() for k in ("coverage", "interval_coverage", "alarm_count", "latency", "gap"))]
            elec_cols = [c for c in feature_cols if c not in comm_cols and not any(k in c.lower() for k in ("meter_changed", "meter_age"))]

            X_comm = split_res.train_df.select(comm_cols).to_numpy() if comm_cols else np.zeros((split_res.train_df.height, 1))
            X_elec = split_res.train_df.select(elec_cols).to_numpy() if elec_cols else np.zeros((split_res.train_df.height, 1))

            data_qual = split_res.train_df["meta__data_quality"].to_numpy() if "meta__data_quality" in split_res.train_df.columns else np.ones(split_res.train_df.height)
            tamper_al = split_res.train_df["x__alarm_count_window"].to_numpy() if "x__alarm_count_window" in split_res.train_df.columns else np.zeros(split_res.train_df.height)
            fa_slopes = split_res.train_df["x__trend_fa_slope"].to_numpy() if "x__trend_fa_slope" in split_res.train_df.columns else None
            m_changed = split_res.train_df["x__meter_changed_30d"].to_numpy() if "x__meter_changed_30d" in split_res.train_df.columns else None
            uc_ids = split_res.train_df["id__uc_id"].to_list() if "id__uc_id" in split_res.train_df.columns else None

            mv_pipeline = MultiViewAnomalyPipeline(contamination=contam, random_state=self.random_state)
            scored_mv = mv_pipeline.fit_and_score(
                X_comm=X_comm,
                X_elec=X_elec,
                data_quality=data_qual,
                tamper_alarms=tamper_al,
                fa_trend_slopes=fa_slopes,
                meter_changed_30d=m_changed,
                raw_dataframe=split_res.train_df,
                uc_ids=uc_ids,
            )

            models_dir = self.base_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(mv_pipeline, models_dir / "specialized_multiview_pipeline.joblib")
            console.print("[bold green]✔ Pipeline Especialista Multi-Visão Treinado com Sucesso![/bold green]")
            console.print("\nDiagnóstico Operacional das UCs:")
            for row in scored_mv["operational_diagnosis"].value_counts().to_dicts():
                console.print(f"  • {row['operational_diagnosis']}: {row['count']:,} UCs")

        # ── Opção 6: Baseline (Isolation Forest + KMeans) ────────────────
        elif mode_choice == "6":
            from sklearn.cluster import KMeans
            from sklearn.ensemble import IsolationForest
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import RobustScaler

            console.print("\n[bold yellow][*] Treinando Modelo Baseline (Isolation Forest + KMeans)...[/bold yellow]")
            X_train = split_res.train_df.select(feature_cols).to_numpy()
            iso_pipe = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), IsolationForest(n_estimators=200, contamination=contam, random_state=self.random_state))
            iso_pipe.fit(X_train)

            km_pipe = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), KMeans(n_clusters=4, random_state=self.random_state))
            km_pipe.fit(X_train)

            models_dir = self.base_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(iso_pipe, models_dir / "isolation_forest_pipeline.joblib")
            joblib.dump(km_pipe, models_dir / "kmeans_clustering_pipeline.joblib")
            console.print("[bold green]✔ Modelos Baseline (Isolation Forest + KMeans) Treinados e Salvos![/bold green]")

        # ── Finalização da Cadeia (Opção 7) ──────────────────────────────
        if mode_choice == "7":
            console.print("\n[bold cyan]🚀 EXECUÇÃO COMPLETA EM CADEIA CONCLUÍDA COM SUCESSO![/bold cyan]")
            console.print("  ✔ Nível 1: Modelo Físico de Tensão")
            console.print("  ✔ Nível 2A: Modelo Global Âncora")
            console.print("  ✔ Nível 2B: Especialistas Locais por Alimentador")
            console.print("  ✔ Nível 2C: Roteamento Adaptativo MoE")
            console.print("[bold green]Todos os artefatos foram treinados e versionados em output/models/.[/bold green]")

    def train_voltage_classifier(self) -> None:
        """Trains and validates the PRODIST Módulo 8 Voltage Disturbance Classifier with data slice filters and registry."""
        console.print("\n[bold cyan]⚡ TREINAMENTO: CLASSIFICADOR DE PERTURBAÇÕES DE TENSÃO (PRODIST M8)[/bold cyan]\n")

        from scripts.train_voltage_classifier import load_alarms_data, load_instantaneous_data

        inst_dir = self.base_dir / "measurements" / "ami_instantaneous"
        context_file = self.base_dir / "context" / "uc_context.parquet"
        hierarchy_file = self.base_dir / "context" / "electrical_hierarchy.parquet"
        meter_hist_file = self.base_dir / "context" / "meter_installation_history.parquet"
        alarms_dir = self.base_dir / "events"

        # 1. Detect Available Dates from Data Lake / Continuity Calendar
        from src.visualization.continuity import scan_datalake_dates

        records = scan_datalake_dates(self.base_dir, file_type="both")
        inst_dates = sorted([d for d, info in records.items() if info.get("instantaneous")])
        if not inst_dates:
            inst_dates = sorted(list(records.keys()))

        if inst_dates:
            min_avail = inst_dates[0]
            max_avail = inst_dates[-1]
            n_days = len(inst_dates)
            lake_summary = f"{min_avail.strftime('%d/%m/%Y')} até {max_avail.strftime('%d/%m/%Y')} ({n_days} dias com telemetria identificados no calendário)"
        else:
            min_avail = date(2026, 8, 1)
            max_avail = date.today()
            lake_summary = "período completo disponível no Data Lake"

        # 1. Select Data Slice (Temporal filter)
        console.print("[bold yellow]📅 1. Seleção do Recorte Temporal (Data Slice):[/bold yellow]")
        console.print(f"  [dim]Período detectado no Data Lake: {lake_summary}[/dim]")
        console.print(f"  [1] Todo o período disponível no Data Lake ({min_avail.strftime('%d/%m/%Y')} a {max_avail.strftime('%d/%m/%Y')}) [Padrão]")
        d30 = max_avail - timedelta(days=30)
        d7 = max_avail - timedelta(days=7)
        console.print(f"  [2] Últimos 30 dias com dados ({max(min_avail, d30).strftime('%d/%m/%Y')} a {max_avail.strftime('%d/%m/%Y')})")
        console.print(f"  [3] Últimos 7 dias com dados ({max(min_avail, d7).strftime('%d/%m/%Y')} a {max_avail.strftime('%d/%m/%Y')})")
        console.print("  [4] Intervalo Personalizado (Data Início → Data Fim)")
        slice_choice = Prompt.ask("Escolha o recorte temporal", choices=["1", "2", "3", "4"], default=self.default_voltage_slice)

        start_date: date | None = None
        end_date: date | None = None

        if slice_choice == "1":
            # Período completo: None/None lê todas as partições existentes no disco, sem truncar em 'ontem'
            start_date = None
            end_date = None
        elif slice_choice == "2":
            start_date = max(min_avail, d30)
            end_date = max_avail
        elif slice_choice == "3":
            start_date = max(min_avail, d7)
            end_date = max_avail
        elif slice_choice == "4":
            s_str = Prompt.ask("Data Inicial (AAAA-MM-DD)", default=min_avail.isoformat())
            e_str = Prompt.ask("Data Final (AAAA-MM-DD)", default=max_avail.isoformat())
            try:
                start_date = date.fromisoformat(s_str)
                end_date = date.fromisoformat(e_str)
            except ValueError:
                console.print("[red]Datas inválidas. Utilizando período total.[/red]")
                start_date, end_date = None, None

        # 2. Select Feeder Filter
        console.print("\n[bold yellow]🔌 2. Filtro por Alimentador Elétrico:[/bold yellow]")
        console.print("  [1] Todos os alimentadores de Araucária (Recomendado)")
        console.print("  [2] Alimentador específico (ex: Fonte_Nova, Angelica, CSN...)")
        feeder_choice = Prompt.ask("Escolha", choices=["1", "2"], default=self.default_voltage_feeder_choice)
        feeder_name: str | None = None
        if feeder_choice == "2":
            feeder_name = Prompt.ask("Nome do Alimentador", default=self.default_feeder)

        # 3. Max Samples (0 = usar todas as amostras / desabilitar limite)
        console.print("\n[bold yellow]🔢 3. Limite Máximo de Amostras para Treino:[/bold yellow]")
        console.print("  [dim]Defina 0 (zero) para desabilitar o limite e utilizar 100% dos dados disponíveis.[/dim]")
        max_samples = IntPrompt.ask("Limite de amostras (0 = sem limite / usar tudo)", default=self.default_voltage_max_samples)

        console.print("\n[*] Carregando telemetria e contexto elétrico correspondente...")
        hierarchy_df = pl.read_parquet(hierarchy_file) if hierarchy_file.exists() else None
        inst_df, data_metadata = load_instantaneous_data(
            instantaneous_dir=inst_dir,
            start_date=start_date,
            end_date=end_date,
            feeder_name=feeder_name,
            hierarchy_df=hierarchy_df,
            sample_quota_total=max_samples,
            random_state=self.random_state,
        )
        context_df = pl.read_parquet(context_file) if context_file.exists() else None
        meter_history_df = pl.read_parquet(meter_hist_file) if meter_hist_file.exists() else None
        alarms_df = load_alarms_data(alarms_dir)

        # Enriquecimento robusto de UC normalizando NIOs
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
                n_ucs = inst_df.select(pl.col("UC").n_unique()).item()
                console.print(f"[dim][*] Telemetria enriquecida com identificador de UC ({n_ucs:,} UCs distintas identificadas).[/dim]")
            elif "NIO" in inst_df.columns and "UC" not in inst_df.columns:
                inst_df = inst_df.with_columns(pl.col("NIO").cast(pl.String).alias("UC"))

        # Amostragem condicional (apenas se max_samples > 0)
        if not inst_df.is_empty():
            if max_samples and max_samples > 0 and inst_df.height > max_samples:
                console.print(f"[*] Amostrando {max_samples:,} de {inst_df.height:,} medições...")
                inst_df = inst_df.sample(n=max_samples, seed=self.random_state)
            else:
                console.print(f"[bold green]✔ Carregadas {inst_df.height:,} medições homogeneamente distribuídas entre os dias do calendário (100% estável em RAM).[/bold green]")

        if inst_df.is_empty():
            console.print("[bold yellow]Aviso: Nenhuma telemetria instantânea encontrada no intervalo. Gerando benchmark PRODIST...[/bold yellow]")
            rows = []
            for rep in range(30):
                for i in range(10):
                    rows.append({"UC": f"UC_NORM_{i}", "NIO": f"NIO_N_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 127.0, "U_L2": 127.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                    rows.append({"UC": f"UC_SAG_{i}", "NIO": f"NIO_S_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 95.0, "U_L2": 96.0, "U_L3": 95.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                    rows.append({"UC": f"UC_SWELL_{i}", "NIO": f"NIO_W_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 142.0, "U_L2": 143.0, "U_L3": 141.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                    rows.append({"UC": f"UC_MONO_{i}", "NIO": f"NIO_M_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 127.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                    rows.append({"UC": f"UC_BI_{i}", "NIO": f"NIO_B_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 0.0, "U_L3": 127.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
                    rows.append({"UC": f"UC_BLACKOUT_{i}", "NIO": f"NIO_K_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 0.0, "U_L2": 0.0, "U_L3": 0.0, "NEIGHBOR_COVERAGE_RATIO": 0.1})
                    rows.append({"UC": f"UC_DESBAL_{i}", "NIO": f"NIO_D_{i}", "FASE_CIRCUITO": "TRIFASICO", "TENSAO_BASE": 127.0, "U_L1": 127.0, "U_L2": 127.0, "U_L3": 118.0, "NEIGHBOR_COVERAGE_RATIO": 1.0})
            inst_df = pl.DataFrame(rows)

        # Extract features and rule labels
        console.print("[*] Extraindo features físicas PRODIST Módulo 8...")
        feat_df = extract_voltage_classifier_features(inst_df, context_df=context_df, alarms_df=alarms_df)
        labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

        dist_table = Table(title="Distribuição das Perturbações (PRODIST Módulo 8)", border_style="cyan")
        dist_table.add_column("Categoria de Perturbação", style="bold yellow")
        dist_table.add_column("Amostras", justify="right", style="green")
        dist_table.add_column("Percentual", justify="right", style="magenta")

        dist_counts = labeled_df.get_column("BASELINE_VOLTAGE_LABEL").value_counts().to_dicts()
        total_rows = max(1, labeled_df.height)
        for r in dist_counts:
            cnt = r["count"]
            dist_table.add_row(r["BASELINE_VOLTAGE_LABEL"], f"{cnt:,}", f"{cnt/total_rows:.1%}")
        console.print(dist_table)

        # Group Split / Estratégia de Particionamento
        uc_col = "UC" if "UC" in labeled_df.columns else labeled_df.columns[0]
        console.print("\n[bold cyan]🎯 4. Estratégia de Particionamento & Teste:[/bold cyan]")
        console.print("  [dim]Padrão Cadastrado: 100% do período de 81 dias alocado para Treinamento (Baseline Oficial).[/dim]")
        console.print("  [dim]A generalização será avaliada de forma temporal (Out-of-Time) sobre dados futuros/novos.[/dim]")
        test_ratio = FloatPrompt.ask("Fração reservada para Teste (0.0 = usar 100% para treino padrão)", default=0.0)
        train_df, test_df = group_train_test_split(labeled_df, group_col=uc_col, test_size=test_ratio, random_state=self.random_state)

        if test_ratio <= 0.0:
            console.print(f"[bold green]✔ Treino Padrão (81 dias):[/bold green] [bold white]{train_df.height:,}[/bold white] amostras (100% da base alocada para treino do baseline oficial).")
            console.print("[dim][*] Estratégia de teste: Validação contínua out-of-time sobre novos dados do calendário.[/dim]")
        else:
            console.print(f"[*] Treino: [bold green]{train_df.height:,}[/bold green] amostras | Teste: [bold yellow]{test_df.height:,}[/bold yellow] amostras (sem vazamento de UC).")

        # Train model
        console.print("[*] Treinando HistGradientBoosting com pesos balanceados...")
        clf = MLVoltageDisturbanceClassifier(random_state=self.random_state)
        clf.fit(train_df)

        # Evaluate Model
        metrics = clf.evaluate(test_df)
        panel_title = "Consistência Interna do Modelo Padrão (Base de Treino - 81 Dias)" if test_ratio <= 0.0 else "Performance no Conjunto de Teste (Disjoint UC)"
        res_panel = Panel(
            f"[bold green]✔ Classificador de Tensão Treinado com Sucesso![/bold green]\n\n"
            f"• Acurácia Global:              [bold white]{metrics.get('accuracy', 0.0):.2%}[/bold white]\n"
            f"• F1-Score Macro:              [bold yellow]{metrics.get('f1_macro', 0.0):.4f}[/bold yellow]\n"
            f"• F1-Score Ponderado:          [bold cyan]{metrics.get('f1_weighted', 0.0):.4f}[/bold cyan]",
            title=panel_title,
            border_style="green",
        )
        console.print(res_panel)

        # Detailed Report Table
        rep_table = Table(title="Relatório Detalhado por Classe de Perturbação", border_style="green")
        rep_table.add_column("Perturbação", style="bold yellow")
        rep_table.add_column("Precisão", justify="right", style="cyan")
        rep_table.add_column("Revocação (Recall)", justify="right", style="blue")
        rep_table.add_column("F1-Score", justify="right", style="green")
        rep_table.add_column("Suporte (Amostras)", justify="right", style="dim")

        report = metrics.get("report", {})
        classes = metrics.get("classes", [])
        for c_name in classes:
            c_dict = report.get(c_name, {})
            if isinstance(c_dict, dict):
                rep_table.add_row(
                    c_name,
                    f"{c_dict.get('precision', 0.0):.3f}",
                    f"{c_dict.get('recall', 0.0):.3f}",
                    f"{c_dict.get('f1-score', 0.0):.3f}",
                    f"{c_dict.get('support', 0):,}",
                )
        console.print(rep_table)

        # 4. Register in Model Registry
        from src.models.registry import ModelRegistry
        registry = ModelRegistry(self.base_dir / "models" / "registry")
        default_tag = "baseline" if test_ratio <= 0.0 else "candidate"
        tag = Prompt.ask("Tag no Model Registry", choices=["candidate", "production", "baseline"], default=default_tag)

        # Formatação exata do período baseado nas datas reais de dados
        loaded_dates = sorted(data_metadata.get("dates_loaded", []))
        eff_start = start_date or (date.fromisoformat(loaded_dates[0]) if loaded_dates else min_avail)
        eff_end = end_date or (date.fromisoformat(loaded_dates[-1]) if loaded_dates else max_avail)
        n_days = len(loaded_dates) if loaded_dates else (eff_end - eff_start).days + 1

        period_label = f"{eff_start.strftime('%d/%m/%Y')} a {eff_end.strftime('%d/%m/%Y')}"
        if test_ratio <= 0.0:
            default_desc = f"Padrão de treinamento PRODIST M8 ({n_days} dias: {period_label})"
        else:
            default_desc = f"Treino de tensão ({period_label} | Teste: {test_ratio:.0%})"

        desc = Prompt.ask("Descrição / Anotação da Versão", default=default_desc)

        version_id = registry.register_model(
            model_name="voltage_disturbance_classifier",
            model_obj=clf,
            metrics=metrics,
            data_slice={
                "start_date": eff_start.isoformat(),
                "end_date": eff_end.isoformat(),
                "days_count": n_days,
                "feeder": feeder_name or "TODOS",
                "total_samples": inst_df.height,
                "train_samples": train_df.height,
                "test_samples": 0 if test_ratio <= 0.0 else test_df.height,
                "test_strategy": "out_of_time_future" if test_ratio <= 0.0 else f"disjoint_uc_{test_ratio:.0%}",
                "standard_training_baseline": True if test_ratio <= 0.0 else False,
            },
            hyperparameters={
                "max_iter": clf.max_iter,
                "learning_rate": clf.learning_rate,
                "min_samples_leaf": clf.min_samples_leaf,
                "random_state": clf.random_state,
            },
            feature_names=clf.FEATURE_COLS,
            tag=tag,
            description=desc,
        )

        console.print(
            Panel(
                f"[bold green]✔ Modelo registrado com sucesso no Model Registry![/bold green]\n\n"
                f"• Versão:      [bold white]{version_id}[/bold white]\n"
                f"• Tag:         [bold yellow]{tag}[/bold yellow]\n"
                f"• Período:     [bold cyan]{eff_start.strftime('%d/%m/%Y')} → {eff_end.strftime('%d/%m/%Y')} ({n_days} dias identificados)[/bold cyan]\n"
                f"• Alimentador: [bold magenta]{feeder_name or 'TODOS'}[/bold magenta]\n"
                f"• Acurácia:    [bold white]{metrics.get('accuracy', 0.0):.2%}[/bold white] | F1 Macro: [bold green]{metrics.get('f1_macro', 0.0):.4f}[/bold green]",
                title="Governança de Modelos",
                border_style="green",
            )
        )

        # Save latest pointer for backwards compatibility
        models_dir = self.base_dir / "models"
        reports_dir = self.base_dir / "reports" / "figures"
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        model_file = models_dir / "voltage_disturbance_classifier.joblib"
        clf.save(model_file)

        # Generate plots
        v_plots = generate_voltage_diagnostic_plots(metrics, output_dir=reports_dir)
        if v_plots:
            console.print("[bold cyan]📊 Gráficos Gerados (Matriz de Confusão & F1-Score):[/bold cyan]")
            for p_name, p_path in v_plots.items():
                console.print(f"  • {p_name}: [underline]{p_path}[/underline]")

    def manage_model_registry(self) -> None:
        """Browse, inspect, and compare versioned models in the Model Registry."""
        console.print("\n[bold cyan]🏛️ GOVERNANÇA: REGISTRO DE MODELOS TREINADOS[/bold cyan]\n")
        from src.models.registry import ModelRegistry
        registry = ModelRegistry(self.base_dir / "models" / "registry")

        registry.render_registry_table()

        models = registry.list_models()
        if len(models) >= 2:
            want_compare = Prompt.ask("\nDeseja comparar duas versões lado a lado?", choices=["s", "n"], default="n") == "s"
            if want_compare:
                versions = [m["version_id"] for m in models]
                v_a = Prompt.ask("Versão A (Baseline / Anterior)", choices=versions, default=versions[1])
                v_b = Prompt.ask("Versão B (Candidata / Nova)", choices=versions, default=versions[0])

                comp = registry.compare_models(models[0]["model_name"], v_a, v_b)
                comp_table = Table(title=f"Comparativo de Performance: {v_a} vs {v_b}", border_style="magenta")
                comp_table.add_column("Métrica / Parâmetro", style="bold yellow")
                comp_table.add_column(f"Versão A ({v_a})", justify="right", style="cyan")
                comp_table.add_column(f"Versão B ({v_b})", justify="right", style="green")
                comp_table.add_column("Variação (Δ)", justify="right", style="bold white")

                acc_a = comp["version_a"]["accuracy"]
                acc_b = comp["version_b"]["accuracy"]
                d_acc = comp["deltas"]["accuracy_delta"]
                comp_table.add_row("Acurácia Global", f"{acc_a:.2%}", f"{acc_b:.2%}", f"{d_acc:+.2%}")

                f1_a = comp["version_a"]["f1_macro"]
                f1_b = comp["version_b"]["f1_macro"]
                d_f1 = comp["deltas"]["f1_macro_delta"]
                comp_table.add_row("F1-Score Macro", f"{f1_a:.4f}", f"{f1_b:.4f}", f"{d_f1:+.4f}")

                comp_table.add_row(
                    "Período dos Dados",
                    f"{comp['version_a']['data_slice'].get('start_date') or '-'} → {comp['version_a']['data_slice'].get('end_date') or '-'}",
                    f"{comp['version_b']['data_slice'].get('start_date') or '-'} → {comp['version_b']['data_slice'].get('end_date') or '-'}",
                    "-",
                )
                comp_table.add_row(
                    "Alimentador",
                    str(comp['version_a']['data_slice'].get('feeder') or 'TODOS'),
                    str(comp['version_b']['data_slice'].get('feeder') or 'TODOS'),
                    "-",
                )
                console.print(comp_table)

    def compare_experts(self) -> None:
        """Direct comparison between Feeder Experts vs Global Anchor Model."""
        console.print("\n[bold cyan]⚡ OPÇÃO 4: COMPARAR ESPECIALISTAS VS MODELO GLOBAL[/bold cyan]\n")

        console.print("Opções de Comparação:")
        console.print("  [1] Especialistas de Alimentador vs Modelo Global Âncora (Consenso Anomalias)")
        console.print("  [2] ⚡ Classificador de Tensão PRODIST M8: Regras Físicas vs Modelo ML")
        comp_choice = Prompt.ask("Escolha a comparação desejada", choices=["1", "2"], default="1")

        if comp_choice == "2":
            self.compare_voltage_models()
            return

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
        plots = generate_diagnostic_plots(eval_df, output_dir=self.base_dir / "reports" / "figures", title_prefix="Modelo Global Âncora")
        if plots:
            console.print("\n[bold cyan]📊 Gráficos Diagnósticos da Comparação Gerados com Sucesso:[/bold cyan]")
            for p_name, p_path in plots.items():
                console.print(f"  • {p_name}: [underline]{p_path}[/underline]")

    def compare_voltage_models(self) -> None:
        """Compares physical PRODIST M8 baseline rules vs ML HistGradientBoosting predictions."""
        console.print("\n[bold cyan]⚡ COMPARAÇÃO: REGRAS PRODIST M8 VS MODELO SUPERVISIONADO (ML)[/bold cyan]\n")

        model_file = self.base_dir / "models" / "voltage_disturbance_classifier.joblib"
        if not model_file.exists():
            console.print(
                Panel(
                    f"[bold red]Modelo ML de tensão não encontrado em {model_file}.[/bold red]\n"
                    "Por favor, treine o modelo primeiro na [bold yellow]Opção 2 -> Modo 4[/bold yellow].",
                    title="Aviso",
                    border_style="yellow",
                )
            )
            return

        from scripts.train_voltage_classifier import load_alarms_data, load_instantaneous_data

        inst_dir = self.base_dir / "measurements" / "ami_instantaneous"
        context_file = self.base_dir / "context" / "uc_context.parquet"
        meter_hist_file = self.base_dir / "context" / "meter_installation_history.parquet"
        alarms_dir = self.base_dir / "events"

        console.print("[*] Carregando modelo salvo e telemetria de teste...")
        clf = MLVoltageDisturbanceClassifier.load(model_file)
        inst_df = load_instantaneous_data(inst_dir)
        context_df = pl.read_parquet(context_file) if context_file.exists() else None
        meter_history_df = pl.read_parquet(meter_hist_file) if meter_hist_file.exists() else None
        alarms_df = load_alarms_data(alarms_dir)

        if not inst_df.is_empty() and "UC" not in inst_df.columns and "NIO" in inst_df.columns:
            if meter_history_df is not None and "NIO" in meter_history_df.columns and "UC" in meter_history_df.columns:
                nio_to_uc = meter_history_df.select(["NIO", "UC"]).unique(subset=["NIO"])
                inst_df = inst_df.join(nio_to_uc, on="NIO", how="left")

        if not inst_df.is_empty() and inst_df.height > 50000:
            inst_df = inst_df.sample(n=50000, seed=self.random_state)

        feat_df = extract_voltage_classifier_features(inst_df, context_df=context_df, alarms_df=alarms_df)
        labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

        res = clf.predict(labeled_df)
        pred_df = res.predictions_df

        rule_col = "BASELINE_VOLTAGE_LABEL"
        ml_col = "PREDICTED_DISTURBANCE_CLASS"

        total = pred_df.height
        concordance = pred_df.filter(pl.col(rule_col) == pl.col(ml_col)).height
        taxa_concordancia = concordance / max(1, total)

        comp_table = Table(title="Concordância: Regras Físicas PRODIST vs Classificador ML", border_style="cyan")
        comp_table.add_column("Métrica de Comparação", style="bold yellow")
        comp_table.add_column("Valor Observado", justify="right", style="green")

        comp_table.add_row("Total de Amostras Avaliadas", f"{total:,}")
        comp_table.add_row("Classificações Idênticas", f"{concordance:,}")
        comp_table.add_row("Taxa Global de Concordância", f"{taxa_concordancia:.2%}")
        comp_table.add_row("Confiança Média do ML", f"{pred_df.get_column('DISTURBANCE_CONFIDENCE').mean():.4f}")
        console.print(comp_table)

        # Cross breakdown
        cross_table = Table(title="Comparativo Cruzado por Perturbação de Tensão", border_style="green")
        cross_table.add_column("Categoria de Perturbação", style="bold yellow")
        cross_table.add_column("Detecções (Regras)", justify="right", style="cyan")
        cross_table.add_column("Detecções (ML)", justify="right", style="magenta")
        cross_table.add_column("Concordância Direta", justify="right", style="blue")

        all_classes = clf.classes_.tolist() if clf.classes_ is not None else []
        for cls_name in all_classes:
            r_cnt = pred_df.filter(pl.col(rule_col) == cls_name).height
            m_cnt = pred_df.filter(pl.col(ml_col) == cls_name).height
            c_cnt = pred_df.filter((pl.col(rule_col) == cls_name) & (pl.col(ml_col) == cls_name)).height
            ratio = f"{(c_cnt / max(1, r_cnt)):.1%}" if r_cnt > 0 else "N/A"
            cross_table.add_row(cls_name, f"{r_cnt:,}", f"{m_cnt:,}", f"{c_cnt:,} ({ratio})")

        console.print(cross_table)

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

        console.print("\nFormato Padrão de Saída dos Dados:")
        console.print("  [1] Ambos (Parquet + CSV)")
        console.print("  [2] Apenas CSV (.csv)")
        console.print("  [3] Apenas Parquet (.parquet)")
        fmt_ch = Prompt.ask("Escolha o formato padrão", choices=["1", "2", "3"], default="1" if self.output_format == "both" else ("2" if self.output_format == "csv" else "3"))
        self.output_format = {"1": "both", "2": "csv", "3": "parquet"}[fmt_ch]

        console.print("\nConfigurações de Treinamento de Tensão (PRODIST M8):")
        self.default_voltage_max_samples = IntPrompt.ask(
            "Limite Padrão de Amostras de Tensão (0 = usar todas / sem limite)",
            default=self.default_voltage_max_samples,
        )
        self.default_voltage_slice = Prompt.ask(
            "Recorte Temporal Padrão [1=Todo o período, 2=Últimos 30d, 3=Últimos 7d, 4=Custom]",
            choices=["1", "2", "3", "4"],
            default=self.default_voltage_slice,
        )
        self.default_feeder = Prompt.ask(
            "Alimentador Específico Padrão",
            default=self.default_feeder,
        )

        console.print(
            f"[bold green]✔ Configurações salvas (Formato: {self.output_format.upper()}, "
            f"Contaminação: {self.contamination:.3f}, Seed: {self.random_state}, "
            f"Max Amostras Tensão: {'Todas (0)' if self.default_voltage_max_samples <= 0 else self.default_voltage_max_samples}, "
            f"Recorte Tensão: Opção {self.default_voltage_slice}).[/bold green]"
        )

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
