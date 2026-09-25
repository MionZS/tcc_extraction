"""Anomaly Detection, Clustering & Hierarchical Model Training Script.

Supports:
- Composite Multi-Technique Model (Isolation Forest, LOF, ECOD, Robust PCA, Physics Rules)
- Hierarchical Training (Feeder-specific, Municipality, and Global Models via MoE Router)
- Baseline Model (Isolation Forest + KMeans)

Usage:
    python scripts/train_anomaly_model.py --mode composite
    python scripts/train_anomaly_model.py --mode hierarchical --feeder Fonte_Nova
    python scripts/train_anomaly_model.py --mode baseline --contamination 0.05
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.datasets.dataset_splitter import DatasetSplitter
from src.models.composite import CompositeAnomalyDetector
from src.models.hierarchical_trainer import HierarchicalModelOrchestrator
from src.models.spurious_filter import filter_spurious_anomalies
from src.tui.tui_app import show_environment_warning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detection models on UC features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/model_input/v1/training_dataset.parquet"),
        help="Path to training_dataset.parquet",
    )
    parser.add_argument(
        "--hierarchy-input",
        type=Path,
        default=Path("output/context/electrical_hierarchy.parquet"),
        help="Path to electrical_hierarchy.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["composite", "hierarchical", "baseline"],
        default="composite",
        help="Training architecture: 'composite', 'hierarchical', or 'baseline'",
    )
    parser.add_argument(
        "--feeder",
        type=str,
        default=None,
        help="Optional specific feeder name to filter/train (e.g. Fonte_Nova)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected proportion of anomalies (default: 0.05)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of behavioral clusters for KMeans (default: 4)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    show_environment_warning()
    args = parse_args()

    if not args.input.exists():
        print(f"[-] Input dataset not found: {args.input}")
        print("    Run the pipeline first to generate model_input/training_dataset.parquet.")
        return

    print(f"[*] Loading training dataset from {args.input}...")
    df = pl.read_parquet(args.input)
    print(f"[+] Loaded {df.height:,} rows, {df.width:,} columns.")

    hierarchy_df = (
        pl.read_parquet(args.hierarchy_input)
        if args.hierarchy_input.exists()
        else None
    )

    feature_cols = [col for col in df.columns if col.startswith("x__") and df[col].dtype.is_numeric()]
    print(f"[+] Identified {len(feature_cols)} numeric model features.")

    if not feature_cols:
        raise ValueError("No numeric features found with 'x__' prefix in dataset.")

    # ── Mode 1: Hierarchical / MoE Training ──────────────────────────────
    if args.mode == "hierarchical":
        print(f"\n[*] Starting Hierarchical MoE Training (Feeder vs Regional vs Global)...")
        orchestrator = HierarchicalModelOrchestrator(
            output_dir=args.output_dir,
            contamination=args.contamination,
            random_state=args.random_state,
        )
        scores_df = orchestrator.train_and_score(
            df,
            hierarchy_df=hierarchy_df,
            target_feeder=args.feeder,
        )

        n_anom = int(scores_df["FINAL_IS_ANOMALY"].sum()) if "FINAL_IS_ANOMALY" in scores_df.columns else 0
        print("\n" + "=" * 50)
        print("HIERARCHICAL MoE TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total evaluated samples: {scores_df.height:,}")
        print(f"Final Genuine Anomalies: {n_anom:,} ({n_anom/scores_df.height:.1%})")
        if "ANOMALY_CATEGORY" in scores_df.columns:
            cats = scores_df["ANOMALY_CATEGORY"].value_counts().to_dicts()
            print("Category breakdown:")
            for c in cats:
                print(f"  - {c.get('ANOMALY_CATEGORY')}: {c.get('count'):,}")
        return

    # ── Mode 2: Composite Multi-Technique Committee ───────────────────────
    if args.mode == "composite":
        print(f"\n[*] Training Composite Multi-Technique Detector (IsoForest + LOF + ECOD + PCA + Rules)...")
        X = df.select(feature_cols).to_numpy()

        detector = CompositeAnomalyDetector(
            contamination=args.contamination,
            random_state=args.random_state,
        )
        detector.fit(X)

        preds = detector.predict_composite(X, raw_dataframe=df)

        scores_dir = args.output_dir / "outputs" / "composite_scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        models_dir = args.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(detector, models_dir / "composite_anomaly_detector.joblib")

        id_cols = [c for c in ("id__uc_id", "meta__cutoff_date", "meta__window_days") if c in df.columns]
        scores_df = df.select(id_cols) if id_cols else pl.DataFrame()
        scores_df = scores_df.with_columns([
            pl.Series("consensus_score", preds["consensus_score"]),
            pl.Series("iso_score", preds["iso_score"]),
            pl.Series("lof_score", preds["lof_score"]),
            pl.Series("ecod_score", preds["ecod_score"]),
            pl.Series("pca_score", preds["pca_score"]),
            pl.Series("rule_violation", preds["rule_violation_flag"]),
            pl.Series("is_anomaly", preds["is_anomaly"]),
        ])

        # Filter spurious noise
        scores_df = filter_spurious_anomalies(scores_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_p = scores_dir / f"composite_scores_{timestamp}.parquet"
        out_c = scores_dir / f"composite_scores_{timestamp}.csv"
        scores_df.write_parquet(out_p)
        scores_df.write_csv(out_c, separator=";")

        print(f"[+] Saved composite detector to {models_dir / 'composite_anomaly_detector.joblib'}")
        print(f"[+] Saved composite scores to {out_p.name}")
        return

    # ── Mode 3: Baseline (Isolation Forest + KMeans) ───────────────────────
    print(f"\n[*] Training Baseline Model (Isolation Forest + KMeans)...")
    X_train = df.select(feature_cols).to_numpy()
    iso_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        IsolationForest(
            n_estimators=200,
            contamination=args.contamination,
            random_state=args.random_state,
            n_jobs=-1,
        ),
    )
    iso_pipeline.fit(X_train)

    estimator = iso_pipeline.named_steps["isolationforest"]
    transformed_X = iso_pipeline.named_steps["robustscaler"].transform(
        iso_pipeline.named_steps["simpleimputer"].transform(X_train)
    )
    raw_scores = -estimator.score_samples(transformed_X)
    is_anomaly = estimator.predict(transformed_X) == -1

    kmeans_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init="auto"),
    )
    cluster_labels = kmeans_pipeline.fit_predict(X_train)

    scores_df = df.select([c for c in ("id__uc_id", "meta__cutoff_date", "meta__window_days") if c in df.columns])
    scores_df = scores_df.with_columns([
        pl.Series("anomaly_score", np.round(raw_scores, 4)),
        pl.Series("is_anomaly", is_anomaly),
        pl.Series("behavior_cluster", cluster_labels),
    ])

    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_pipeline, models_dir / "isolation_forest_pipeline.joblib")
    joblib.dump(kmeans_pipeline, models_dir / "kmeans_clustering_pipeline.joblib")

    scores_dir = args.output_dir / "outputs" / "anomaly_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_parquet = scores_dir / f"anomaly_scores_{timestamp}.parquet"
    scores_df.write_parquet(out_parquet)
    print(f"[+] Saved baseline anomaly scores to {out_parquet.name}")


if __name__ == "__main__":
    main()
