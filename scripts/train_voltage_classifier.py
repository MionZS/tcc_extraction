"""Train and Evaluate the Supervised Voltage Disturbance Classifier (PRODIST M8).

Loads instantaneous telemetry measurements, electrical hierarchy context, and alarm events,
filters by chosen date period and feeder, extracts PRODIST Módulo 8 features, labels baseline classes,
splits train/test strictly by UC to avoid data leakage, fits the HistGradientBoosting classifier,
exports evaluation reports, and registers the versioned model into the Model Registry.

Usage:
    # Full dataset
    python scripts/train_voltage_classifier.py

    # Specific date range & feeder
    python scripts/train_voltage_classifier.py --start-date 2026-08-01 --end-date 2026-10-18 --feeder Fonte_Nova
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

import polars as pl

from src.models.registry import ModelRegistry
from src.models.voltage_classifier import (
    MLVoltageDisturbanceClassifier,
    RuleBasedVoltageClassifier,
    extract_voltage_classifier_features,
    group_train_test_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PRODIST M8 Voltage Disturbance Classifier.")
    parser.add_argument(
        "--instantaneous-dir",
        type=Path,
        default=Path("output/measurements/ami_instantaneous"),
        help="Path to instantaneous telemetry parquet files.",
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
        help="Optional: Filter training data exclusively to a specific electrical feeder.",
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
        "--max-samples",
        type=int,
        default=200000,
        help="Maximum telemetry rows to sample for training (default: 200,000).",
    )
    parser.add_argument(
        "--alarms-dir",
        type=Path,
        default=Path("output/events"),
        help="Path to alarms directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of UCs reserved for disjoint test split (default: 0.25).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="candidate",
        help="Registry tag for model (candidate, production, baseline).",
    )
    parser.add_argument(
        "--version-name",
        type=str,
        default=None,
        help="Optional custom version string for the registry.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description or notes about this training experiment.",
    )
    return parser.parse_args()


def _extract_date_from_partition_path(path: Path) -> date | None:
    try:
        parts = path.parts
        y = next(int(p.split("=")[1]) for p in parts if p.startswith("report_year="))
        m = next(int(p.split("=")[1]) for p in parts if p.startswith("report_month="))
        d = next(int(p.split("=")[1]) for p in parts if p.startswith("report_day="))
        return date(y, m, d)
    except Exception:
        return None


def load_instantaneous_data(
    instantaneous_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    feeder_name: str | None = None,
    hierarchy_df: pl.DataFrame | None = None,
    max_files: int | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load instantaneous measurements filtered by date partition range and feeder."""
    metadata: dict[str, Any] = {
        "dates_loaded": [],
        "files_count": 0,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "feeder": feeder_name,
    }

    if not instantaneous_dir.exists():
        return pl.DataFrame(), metadata

    files = list(instantaneous_dir.rglob("*.parquet"))
    if not files:
        files = list(instantaneous_dir.rglob("*.csv"))
    if not files:
        return pl.DataFrame(), metadata

    # Filter files by partition dates
    candidate_files: list[tuple[date, Path]] = []
    for f in files:
        d = _extract_date_from_partition_path(f)
        if d is not None:
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            candidate_files.append((d, f))
        else:
            # Fallback if unpartitioned
            candidate_files.append((date(2000, 1, 1), f))

    candidate_files.sort(key=lambda x: x[0])
    if max_files:
        candidate_files = candidate_files[-max_files:]

    if not candidate_files:
        print(f"[-] Nenhuma partição encontrada no intervalo de datas especificado ({start_date} → {end_date}).")
        return pl.DataFrame(), metadata

    dates_found = sorted(list({d for d, _ in candidate_files if d != date(2000, 1, 1)}))
    print(f"[*] Carregando {len(candidate_files)} arquivo(s) de telemetria ({len(dates_found)} dia(s) identificados)...")

    dfs = []
    for _, f in candidate_files:
        try:
            if f.suffix == ".parquet":
                dfs.append(pl.read_parquet(f))
            elif f.suffix == ".csv":
                dfs.append(pl.read_csv(f, separator=";", infer_schema_length=5000, ignore_errors=True))
        except Exception as e:
            print(f"[-] Erro ao ler {f.name}: {e}")
            print(f"[-] Could not read {f.name}: {e}")

    if not dfs:
        return pl.DataFrame(), metadata

    combined = pl.concat(dfs, how="diagonal")

    # Filter by feeder if requested
    if feeder_name and hierarchy_df is not None and not hierarchy_df.is_empty():
        if "ALIMENTADOR" in hierarchy_df.columns:
            target_ucs = (
                hierarchy_df.filter(pl.col("ALIMENTADOR").str.to_uppercase() == feeder_name.upper())
                .get_column("UC")
                .drop_nulls()
                .unique()
                .to_list()
            )
            if target_ucs and "UC" in combined.columns:
                orig_cnt = combined.height
                combined = combined.filter(pl.col("UC").is_in(target_ucs))
                print(f"[*] Filtrado pelo alimentador '{feeder_name}': {combined.height:,} de {orig_cnt:,} medições.")

    metadata["dates_loaded"] = [d.isoformat() for d in dates_found]
    metadata["files_count"] = len(candidate_files)
    return combined, metadata


def load_alarms_data(alarms_dir: Path) -> pl.DataFrame | None:
    """Load alarms aggregate file if available."""
    if not alarms_dir.exists():
        return None

    alarm_files = list(alarms_dir.glob("alarms_aggregate_*.csv"))
    if not alarm_files:
        return None

    target_file = sorted(alarm_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"[*] Loading alarms from {target_file.name}...")
    try:
        return pl.read_csv(target_file, separator=";", truncate_ragged_lines=True, ignore_errors=True)
    except Exception as e:
        print(f"[-] Warning: Failed to load alarms CSV: {e}")
        return None


def main() -> None:
    args = parse_args()
    print("=" * 65)
    print("VOLTAGE DISTURBANCE CLASSIFIER — TRAINING & GOVERNANCE PIPELINE")
    print("=" * 65)

    # 1. Load Data
    hierarchy_df = pl.read_parquet(args.hierarchy_file) if args.hierarchy_file.exists() else None
    inst_df, data_metadata = load_instantaneous_data(
        instantaneous_dir=args.instantaneous_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        feeder_name=args.feeder,
        hierarchy_df=hierarchy_df,
    )
    context_df = pl.read_parquet(args.context_file) if args.context_file.exists() else None
    meter_history_df = pl.read_parquet(args.meter_history_file) if args.meter_history_file.exists() else None
    alarms_df = load_alarms_data(args.alarms_dir)

    # If UC is not in inst_df, enrich with UC from meter history
    if not inst_df.is_empty() and "UC" not in inst_df.columns and "NIO" in inst_df.columns:
        if meter_history_df is not None and "NIO" in meter_history_df.columns and "UC" in meter_history_df.columns:
            nio_to_uc = meter_history_df.select(["NIO", "UC"]).unique(subset=["NIO"])
            inst_df = inst_df.join(nio_to_uc, on="NIO", how="left")
            print("[*] Enriched instantaneous telemetry with UC from meter installation history.")

    # Sample rows if telemetry dataset is very large
    if not inst_df.is_empty() and inst_df.height > args.max_samples:
        print(f"[*] Telemetry volume ({inst_df.height:,} rows) exceeds max_samples ({args.max_samples:,}). Sampling evenly...")
        inst_df = inst_df.sample(n=args.max_samples, seed=args.random_state)

    if inst_df.is_empty():
        print("[-] Nenhuma telemetria instantânea encontrada no diretório/intervalo selecionado.")
        print("[*] Gerando benchmark sintético PRODIST M8 para validação da pipeline...")
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

    print(f"[+] Total de amostras carregadas para classificação de tensão: {inst_df.height:,}")

    # 2. Extract PRODIST Features
    print("[*] Extraindo features físicas PRODIST Módulo 8 e flags de falta...")
    feat_df = extract_voltage_classifier_features(
        instantaneous_df=inst_df,
        context_df=context_df,
        alarms_df=alarms_df,
    )

    # 3. Apply PRODIST Rule-Based Baseline
    print("[*] Aplicando regras determinísticas normativas PRODIST Módulo 8...")
    labeled_df = RuleBasedVoltageClassifier.classify(feat_df)

    dist_counts = labeled_df.get_column("BASELINE_VOLTAGE_LABEL").value_counts().to_dicts()
    print("[+] Distribuição das classes normativas:")
    for row in dist_counts:
        print(f"    - {row['BASELINE_VOLTAGE_LABEL']}: {row['count']:,}")

    # 4. Group Split by UC (No Leakage)
    uc_col = "UC" if "UC" in labeled_df.columns else labeled_df.columns[0]
    print(f"[*] Dividindo dataset agrupado por '{uc_col}' (test_size={args.test_size})...")
    train_df, test_df = group_train_test_split(
        labeled_df,
        group_col=uc_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"[+] Amostras Treino: {train_df.height:,} | Amostras Teste: {test_df.height:,}")

    # 5. Fit ML Classifier
    print("[*] Treinando MLVoltageDisturbanceClassifier (HistGradientBoosting com pesos balanceados)...")
    clf = MLVoltageDisturbanceClassifier(random_state=args.random_state)
    clf.fit(train_df)

    # 6. Evaluate Model on Disjoint Test Set
    print("\n[*] Avaliando acurácia no conjunto de teste isolado...")
    metrics = clf.evaluate(test_df)
    print(f"    * Acurácia Global: {metrics.get('accuracy', 0.0):.4f}")
    print(f"    * F1 Macro:        {metrics.get('f1_macro', 0.0):.4f}")
    print(f"    * F1 Ponderado:    {metrics.get('f1_weighted', 0.0):.4f}")

    print("\n[+] Relatório de Classificação por Categoria:")
    report = metrics.get("report", {})
    for cls_name, vals in report.items():
        if isinstance(vals, dict):
            p = vals.get("precision", 0.0)
            r = vals.get("recall", 0.0)
            f1 = vals.get("f1-score", 0.0)
            sup = vals.get("support", 0)
            print(f"    - {cls_name:25s} | Prec: {p:.3f} | Rec: {r:.3f} | F1: {f1:.3f} | Support: {sup}")

    # 7. Persist to Model Registry
    data_slice_info = {
        "start_date": args.start_date.isoformat() if args.start_date else None,
        "end_date": args.end_date.isoformat() if args.end_date else None,
        "dates_count": len(data_metadata.get("dates_loaded", [])),
        "feeder": args.feeder or "TODOS",
        "total_samples": inst_df.height,
        "train_samples": train_df.height,
        "test_samples": test_df.height,
    }

    registry = ModelRegistry(args.output_dir / "models" / "registry")
    version_id = registry.register_model(
        model_name="voltage_disturbance_classifier",
        model_obj=clf,
        metrics=metrics,
        data_slice=data_slice_info,
        hyperparameters={
            "max_iter": clf.max_iter,
            "learning_rate": clf.learning_rate,
            "min_samples_leaf": clf.min_samples_leaf,
            "random_state": clf.random_state,
        },
        feature_names=clf.FEATURE_COLS,
        tag=args.tag,
        description=args.description,
        custom_version=args.version_name,
    )

    # Also persist latest active model pointer for backwards compatibility
    models_dir = args.output_dir / "models"
    outputs_dir = args.output_dir / "outputs" / "voltage_classifier"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "voltage_disturbance_classifier.joblib"
    metrics_path = outputs_dir / "voltage_classification_report.json"

    clf.save(model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[✔] Modelo versionado registrado no Model Registry com ID: {version_id}")
    print(f"[+] Tag:                  {args.tag}")
    print(f"[+] Período dos Dados:    {data_slice_info['start_date'] or 'Inicio'} → {data_slice_info['end_date'] or 'Fim'}")
    print(f"[+] Alimentador:          {data_slice_info['feeder']}")
    print(f"[+] Arquivo do Modelo:    {model_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
