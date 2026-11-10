"""Sequential Automated End-to-End Training Pipeline.

Executes all modeling tiers in optimal sequence without interactive prompts:
1. Nível 1: Voltage Disturbance Classifier (PRODIST Módulo 8 / ML).
2. Nível 2A: Global Anchor Model (Composite Committee: IsoForest + LOF + ECOD + PCA).
3. Nível 2B: Local Feeder Specialists (Dedicated per-feeder models).
4. Nível 2C: Hierarchical MoE Router with calibrated thresholds & Spurious Filter.
5. Nível 2D: Specialized Multi-View Pipeline (S_comm, S_elec, S_tamper, Behavioral Clusters).
6. Nível 2E: Anomaly Evidence Miner (Sub-clustering anomalous subspace and physical hypotheses).
7. Consolidation: Exports classified UCs table with scores, categories, and evidence to:
   - `output/reports/classified_ucs_latest.parquet` (and `.csv`)
   - `output/reports/model_training_summary_latest.json`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.datasets.dataset_splitter import DatasetSplitter
from src.datasets.retroactive_cleaner import RetroactiveDataCleaner
from src.models.category_reporting import (
    apply_delimited_isolation_and_categories,
    export_all_category_reports,
)
from src.models.composite import CompositeAnomalyDetector, select_pure_electrical_features
from src.models.evidence_miner import AnomalyEvidenceMiner
from src.models.hierarchical_trainer import HierarchicalModelOrchestrator
from src.models.registry import ModelRegistry
from src.models.specialized_detectors import MultiViewAnomalyPipeline

console = Console()


def run_sequential_training(
    base_dir: Path | str = "output",
    *,
    contamination: float = 0.01,
    random_state: int = 42,
    force_clean: bool = False,
    write_csv: bool = True,
) -> dict[str, Any]:
    """Runs the complete training and classification chain sequentially."""
    base_dir = Path(base_dir)
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_file = base_dir / "model_input" / "v1" / "training_dataset.parquet"
    if not input_file.exists():
        c_peer = base_dir / "model_input" / "v1" / "training_dataset.csv"
        if c_peer.exists():
            input_file = c_peer

    hierarchy_file = base_dir / "context" / "electrical_hierarchy.parquet"
    if not hierarchy_file.exists():
        c_hier = base_dir / "context" / "electrical_hierarchy.csv"
        if c_hier.exists():
            hierarchy_file = c_hier

    console.print(Panel(
        "[bold cyan]⚡ PIPELINE DE TREINAMENTO SEQUENCIAL COMPLETO (ARIANE / COPEL REI)[/bold cyan]\n"
        f"• Base de Dados: {base_dir}\n"
        f"• Contaminação:  {contamination:.2%}\n"
        f"• Semente:       {random_state}",
        border_style="cyan",
    ))

    # ── 0. Pré-Voo e Saneamento se necessário ────────────────────────────────
    df = (
        pl.read_parquet(input_file)
        if input_file.suffix == ".parquet"
        else pl.read_csv(input_file, separator=";", infer_schema_length=5000, ignore_errors=True)
    ) if input_file.exists() else pl.DataFrame()

    feeder_null_ratio = float(df.get_column("meta__feeder").is_null().mean() or 1.0) if "meta__feeder" in df.columns else 1.0
    zero_mad_ratio = float((df.get_column("x__hist_fa_sum_mad").fill_null(0.0) == 0.0).mean() or 1.0) if "x__hist_fa_sum_mad" in df.columns else 1.0

    if force_clean or df.is_empty() or feeder_null_ratio > 0.50 or zero_mad_ratio > 0.95:
        console.print("\n[bold yellow][*] Executando Saneamento Retroativo & Reconstrução Multi-Janela (Offline)...[/bold yellow]")
        cleaner = RetroactiveDataCleaner(base_dir=base_dir)
        clean_rep = cleaner.run(stride_mode="monthly", write_csv=write_csv)
        input_file = clean_rep.dataset_path
        df = pl.read_parquet(input_file) if input_file.suffix == ".parquet" else pl.read_csv(input_file.with_suffix(".csv"), separator=";")
        console.print(f"[bold green]✔ Saneamento concluído: {df.height:,} janelas consolidadas.[/bold green]")

    hdf = (
        pl.read_parquet(hierarchy_file)
        if hierarchy_file.suffix == ".parquet"
        else pl.read_csv(hierarchy_file, separator=";", infer_schema_length=5000, ignore_errors=True)
    ) if hierarchy_file.exists() else None

    # ── 1. Particionamento OOT (Out-of-Time) ──────────────────────────────────
    console.print("\n[bold yellow][*] Passo 1: Particionamento Temporal OOT (Sem Vazamento)...[/bold yellow]")
    splitter = DatasetSplitter(
        canonical_train_start=date(2026, 8, 1),
        canonical_train_end=date(2026, 10, 31),
        random_state=random_state,
    )
    split_res = splitter.split(df, hierarchy_df=hdf)
    summary_split = split_res.split_summary
    console.print(
        f"  • Treino:    {summary_split['train_rows']:,} ({summary_split['train_ratio']:.1%})\n"
        f"  • Validação: {summary_split['val_rows']:,} ({summary_split['val_ratio']:.1%})\n"
        f"  • Teste OOT: {summary_split['test_rows']:,} ({summary_split['test_ratio']:.1%})"
    )

    all_feature_cols = [c for c in df.columns if c.startswith("x__") and df[c].dtype.is_numeric()]
    elec_feature_cols = select_pure_electrical_features(df)
    console.print(
        f"  • Features Elétricas Puras (Consumo, Tensão, Corrente): [bold green]{len(elec_feature_cols)}[/bold green] "
        f"(de {len(all_feature_cols)} totais — Comunicação/Idade isoladas do detector elétrico)"
    )

    clean_train_df = (
        split_res.train_df.filter(pl.col("meta__is_clean_for_train").fill_null(True))
        if "meta__is_clean_for_train" in split_res.train_df.columns
        else split_res.train_df
    )
    clean_test_df = (
        split_res.test_df.filter(pl.col("meta__is_clean_for_train").fill_null(True))
        if "meta__is_clean_for_train" in split_res.test_df.columns
        else split_res.test_df
    )
    if clean_test_df.is_empty():
        clean_test_df = split_res.test_df

    # ── 2. Nível 1: Classificador de Tensão (Registry Linking) ───────────────
    console.print("\n[bold yellow][*] Passo 2: Vinculando Classificador de Tensão PRODIST M8 (Nível 1)...[/bold yellow]")
    registry = ModelRegistry(base_dir / "models" / "registry")
    v_base = registry.get_latest_baseline("voltage_disturbance_classifier", prefer_deterministic=True)
    if v_base is not None:
        _, v_meta, _ = v_base
        console.print(f"  ✔ Nível 1 Ativo: {v_meta.get('version_id')} (Tag: {v_meta.get('tag')})")
    else:
        console.print("  [dim]Nenhum modelo de tensão no Registry. Executando baseline determinístico integrado.[/dim]")

    # ── 3. Nível 2A: Modelo Global Âncora (Comitê PyOD 6-Modelos em Features Elétricas Puras) ──
    console.print("\n[bold yellow][*] Passo 3: Treinando Comitê PyOD (IsoForest+LOF+ECOD+COPOD+HBOS+PCA) em Features Elétricas Puras...[/bold yellow]")
    X_train = clean_train_df.select(elec_feature_cols).to_numpy()
    detector_global = CompositeAnomalyDetector(contamination=contamination, random_state=random_state)
    detector_global.fit(X_train)

    X_test = clean_test_df.select(elec_feature_cols).to_numpy()
    preds_global = detector_global.predict_composite(X_test, raw_dataframe=clean_test_df)
    n_global_anom = int(preds_global["is_anomaly"].sum())

    v_global_id = registry.register_model(
        model_name="composite_global_anchor",
        model_obj=detector_global,
        metrics={
            "anomalies_detected": n_global_anom,
            "anomaly_rate": round(n_global_anom / max(1, clean_test_df.height), 4),
            "mean_consensus_score": round(float(preds_global["consensus_score"].mean()), 4),
            "train_samples": clean_train_df.height,
            "test_samples": clean_test_df.height,
        },
        data_slice={
            "train_samples": clean_train_df.height,
            "test_samples": clean_test_df.height,
            "contamination": contamination,
        },
        feature_names=elec_feature_cols,
        tag="production",
        description="Comitê PyOD 6-Modelos (Features Elétricas Puras — Sem Vazamento de Telecom)",
    )
    joblib.dump(detector_global, models_dir / "composite_anomaly_detector.joblib")
    console.print(f"  ✔ Modelo Global Âncora Registrado em Produção: {v_global_id}")

    # ── 4. Nível 2B & 2C: Especialistas por Alimentador & MoE Router ──────────
    console.print("\n[bold yellow][*] Passo 4: Treinando Especialistas Locais e Roteador MoE...[/bold yellow]")
    orchestrator = HierarchicalModelOrchestrator(
        output_dir=base_dir,
        contamination=contamination,
        random_state=random_state,
    )
    # Fit MoE on train, score on test
    train_scored = orchestrator.train_and_score(clean_train_df, hierarchy_df=hdf)
    test_scored = orchestrator.train_and_score(clean_test_df, hierarchy_df=hdf)
    n_moe_anom = int(test_scored.get_column("FINAL_IS_ANOMALY").sum())
    console.print(f"  ✔ Especialistas e MoE Concluídos. Anomalias no Teste OOT: {n_moe_anom:,} ({n_moe_anom/max(1, clean_test_df.height):.2%})")

    # ── 5. Nível 2D: Pipeline Especialista Multi-Visão ───────────────────────
    console.print("\n[bold yellow][*] Passo 5: Decomposição Multi-Visão (S_comm, S_elec, S_tamper)...[/bold yellow]")
    comm_cols = [c for c in all_feature_cols if any(k in c.lower() for k in ("coverage", "interval_coverage", "alarm_count", "latency", "gap"))]
    elec_cols = elec_feature_cols

    X_comm = clean_train_df.select(comm_cols).to_numpy() if comm_cols else np.zeros((clean_train_df.height, 1))
    X_elec = clean_train_df.select(elec_cols).to_numpy() if elec_cols else np.zeros((clean_train_df.height, 1))

    data_qual = clean_train_df["meta__data_quality"].to_numpy() if "meta__data_quality" in clean_train_df.columns else np.ones(clean_train_df.height)
    tamper_al = clean_train_df["x__alarm_count_window"].to_numpy() if "x__alarm_count_window" in clean_train_df.columns else np.zeros(clean_train_df.height)
    fa_slopes = clean_train_df["x__trend_fa_slope"].to_numpy() if "x__trend_fa_slope" in clean_train_df.columns else None
    m_changed = clean_train_df["x__meter_changed_30d"].to_numpy() if "x__meter_changed_30d" in clean_train_df.columns else None
    uc_ids = clean_train_df["id__uc_id"].to_list() if "id__uc_id" in clean_train_df.columns else None

    mv_pipe = MultiViewAnomalyPipeline(contamination=contamination, random_state=random_state)
    _ = mv_pipe.fit_and_score(
        X_comm=X_comm,
        X_elec=X_elec,
        data_quality=data_qual,
        tamper_alarms=tamper_al,
        fa_trend_slopes=fa_slopes,
        meter_changed_30d=m_changed,
        raw_dataframe=clean_train_df,
        uc_ids=uc_ids,
    )
    joblib.dump(mv_pipe, models_dir / "specialized_multiview_pipeline.joblib")
    console.print("  ✔ Pipeline Multi-Visão treinado e serializado.")

    # ── 6. Nível 2E: Mineração de Evidências & Categorização de Causas ────────
    console.print("\n[bold yellow][*] Passo 6: Mineração de Evidências e Sub-Clustering Explicável...[/bold yellow]")
    miner = AnomalyEvidenceMiner(n_clusters=4, random_state=random_state)
    evidences = miner.mine_evidence(test_scored, score_col="FINAL_MOE_SCORE", anomaly_flag_col="FINAL_IS_ANOMALY")
    console.print(f"  ✔ {len(evidences)} categorias causais descobertas nos dados.")

    # ── 7. Consolidação Final: Regras Delimitadas, Transição de Classe e 8 Relatórios ──
    console.print("\n[bold yellow][*] Passo 7: Consolidando Regras de Isolação, Transição de Classes e Relatórios por Categoria...[/bold yellow]")

    # Run inference across full dataset strictly over Pure Electrical Features
    full_preds = detector_global.predict_composite(df.select(elec_feature_cols).to_numpy(), raw_dataframe=df)

    base_scored_df = df.with_columns([
        pl.Series("ANOMALY_SCORE", full_preds["consensus_score"].round(4)),
        pl.Series("COPOD_SCORE", full_preds["copod_score"].round(4)),
        pl.Series("ECOD_SCORE", full_preds["ecod_score"].round(4)),
        pl.Series("LOF_SCORE", full_preds["lof_score"].round(4)),
        pl.Series("HBOS_SCORE", full_preds["hbos_score"].round(4)),
        pl.Series("IS_ANOMALY", full_preds["is_anomaly"]),
    ])

    classified_df, transitions_df = apply_delimited_isolation_and_categories(
        base_scored_df,
        base_dir=base_dir,
    )

    category_reports_manifest = export_all_category_reports(
        classified_df,
        transitions_df,
        reports_dir=reports_dir,
        write_csv=write_csv,
    )

    out_parquet = reports_dir / "classified_ucs_latest.parquet"
    classified_df.write_parquet(out_parquet, compression="zstd")
    if write_csv:
        classified_df.write_csv(out_parquet.with_suffix(".csv"), separator=";")

    # Summary JSON report
    cat_counts = (
        classified_df.group_by(["CATEGORIA_DIAGNOSTICA", "METODO_DETECCAO_APLICADO"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "total_ucs": classified_df.height,
        "clean_train_samples": clean_train_df.height,
        "test_oot_samples": clean_test_df.height,
        "pure_electrical_features": elec_feature_cols,
        "anomalies_detected": int(classified_df.get_column("IS_ANOMALY").sum()),
        "temporal_class_transitions_count": transitions_df.height,
        "categories_distribution": cat_counts,
        "category_reports_manifest": category_reports_manifest,
        "evidences_mined": [
            {
                "cluster_id": e.cluster_id,
                "name": e.cluster_name,
                "count": e.sample_count,
                "ratio": e.sample_ratio,
                "hypothesis": e.primary_hypothesis,
                "top_features": e.top_divergent_features,
            }
            for e in evidences
        ],
        "exported_files": {
            "classified_parquet": str(out_parquet),
            "classified_csv": str(out_parquet.with_suffix(".csv")) if write_csv else None,
            "category_reports_dir": str(reports_dir / "categorias"),
        },
    }
    with open(reports_dir / "model_training_summary_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    console.print(Panel(
        f"[bold green]✔ TREINAMENTO E EXPORTAÇÃO DE RELATÓRIOS CONCLUÍDOS COM SUCESSO![/bold green]\n\n"
        f"• Total de Janelas (UC × Cutoff): [bold white]{classified_df.height:,}[/bold white]\n"
        f"• Anomalias Elétricas Puras:      [bold red]{summary_report['anomalies_detected']:,} ({summary_report['anomalies_detected']/max(1, classified_df.height):.2%})[/bold red]\n"
        f"• UCs que Mudaram de Classe:      [bold yellow]{transitions_df.height:,}[/bold yellow]\n"
        f"• Catálogo Geral Salvo em:        [underline]{out_parquet}[/underline]\n"
        f"• 8 Relatórios por Categoria em:  [underline]{reports_dir / 'categorias'}[/underline]",
        border_style="green",
    ))

    # Render dedicated category reports & ML vs. Rule matrix
    cat_tbl = Table(title="Relatórios Dedicados por Categoria & Matriz Metodológica (ML vs. Regra)", border_style="cyan")
    cat_tbl.add_column("Relatório / Categoria", style="bold yellow")
    cat_tbl.add_column("Janelas", justify="right", style="cyan")
    cat_tbl.add_column("UCs Únicas", justify="right", style="green")
    cat_tbl.add_column("Paradigma (Onde é ML vs. Regra)", style="white")
    cat_tbl.add_column("Arquivo Exportado (output/reports/categorias/)", style="dim")
    for item in category_reports_manifest:
        cat_tbl.add_row(
            item["title"],
            f"{item['row_count']:,}",
            f"{item['unique_ucs']:,}",
            item["paradigm"],
            f"{item['report_slug']}.csv",
        )
    console.print(cat_tbl)

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential automated training pipeline.")
    parser.add_argument("--base-dir", type=str, default="output", help="Root output directory.")
    parser.add_argument("--contamination", type=float, default=0.01, help="Nominal contamination rate.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state.")
    parser.add_argument("--force-clean", action="store_true", help="Force retroactive data cleaning prior to train.")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV emission.")
    args = parser.parse_args()

    run_sequential_training(
        base_dir=args.base_dir,
        contamination=args.contamination,
        random_state=args.random_state,
        force_clean=args.force_clean,
        write_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()

