"""Baseline Anomaly Detection & Clustering Training Script.

Trains an unsupervised baseline using IsolationForest and KMeans
on the model_input training dataset (UC × cutoff_date).

Usage:
    python scripts/train_anomaly_model.py
    python scripts/train_anomaly_model.py --input output/model_input/v1/training_dataset.parquet --contamination 0.05
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detection model on UC features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/model_input/v1/training_dataset.parquet"),
        help="Path to training_dataset.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory",
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
    args = parse_args()

    if not args.input.exists():
        print(f"[-] Input dataset not found: {args.input}")
        print("    Run the pipeline first to generate model_input/training_dataset.parquet.")
        return

    print(f"[*] Loading training dataset from {args.input}...")
    df = pl.read_parquet(args.input)
    print(f"[+] Loaded {df.height:,} rows, {df.width:,} columns.")

    # ── Feature Selection ─────────────────────────────────────────────────
    feature_cols = [
        col for col in df.columns
        if col.startswith("x__")
    ]
    # Filter only numeric features for baseline estimators
    numeric_features = [
        col for col in feature_cols
        if df[col].dtype.is_numeric()
    ]

    print(f"[+] Identified {len(numeric_features)} numeric model features:")
    for f in numeric_features:
        print(f"    - {f}")

    if not numeric_features:
        raise ValueError("No numeric features found with 'x__' prefix in dataset.")

    # Select samples (use train split if present)
    if "meta__split" in df.columns:
        train_df = df.filter(pl.col("meta__split") == "train")
        if train_df.is_empty():
            train_df = df
    else:
        train_df = df

    X_train = train_df.select(numeric_features).to_numpy()

    # ── Model 1: Isolation Forest (Anomaly Detection) ─────────────────────
    print(f"\n[*] Training Isolation Forest (contamination={args.contamination})...")
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

    # Negative decision function: higher means more anomalous
    estimator = iso_pipeline.named_steps["isolationforest"]
    transformed_X = iso_pipeline.named_steps["robustscaler"].transform(
        iso_pipeline.named_steps["simpleimputer"].transform(X_train)
    )
    raw_scores = -estimator.score_samples(transformed_X)
    predictions = estimator.predict(transformed_X)
    is_anomaly = predictions == -1

    # ── Model 2: KMeans (Behavioral Profiling / Clustering) ───────────────
    print(f"[*] Training KMeans behavioral clustering (k={args.n_clusters})...")
    kmeans_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        KMeans(
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            n_init="auto",
        ),
    )
    cluster_labels = kmeans_pipeline.fit_predict(X_train)

    # ── Build Output Scores Table ─────────────────────────────────────────
    scores_df = train_df.select([
        c for c in ("id__uc_id", "meta__cutoff_date", "meta__window_days")
        if c in train_df.columns
    ])

    scores_df = scores_df.with_columns([
        pl.Series("anomaly_score", np.round(raw_scores, 4)),
        pl.Series("is_anomaly", is_anomaly),
        pl.Series("behavior_cluster", cluster_labels),
    ])

    # Rank percentile (0.0 = normal, 1.0 = most anomalous)
    ranks = (np.argsort(np.argsort(raw_scores)) + 1) / len(raw_scores)
    scores_df = scores_df.with_columns(
        pl.Series("rank_percentile", np.round(ranks, 4))
    ).sort("anomaly_score", descending=True)

    # ── Save Models & Scores ──────────────────────────────────────────────
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_pipeline, models_dir / "isolation_forest_pipeline.joblib")
    joblib.dump(kmeans_pipeline, models_dir / "kmeans_clustering_pipeline.joblib")
    print(f"[+] Saved trained pipelines to {models_dir}")

    scores_dir = args.output_dir / "outputs" / "anomaly_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_parquet = scores_dir / f"anomaly_scores_{timestamp}.parquet"
    out_csv = scores_dir / f"anomaly_scores_{timestamp}.csv"

    scores_df.write_parquet(out_parquet)
    scores_df.write_csv(out_csv, separator=";")
    print(f"[+] Saved anomaly scores to:\n    - {out_parquet}\n    - {out_csv}")

    # ── Summary Report ────────────────────────────────────────────────────
    n_anomalies = int(is_anomaly.sum())
    print("\n" + "=" * 50)
    print("TRAINING & SCORING SUMMARY")
    print("=" * 50)
    print(f"Total evaluated samples: {len(raw_scores):,}")
    print(f"Flagged anomalies:       {n_anomalies:,} ({n_anomalies/len(raw_scores):.1%})")
    print(f"Behavior clusters:       {args.n_clusters}")
    print("\nTop 5 Flagged UCs by Anomaly Score:")
    top5 = scores_df.head(5)
    for row in top5.iter_rows(named=True):
        print(
            f"  UC {row.get('id__uc_id')}: score={row.get('anomaly_score'):.4f}, "
            f"percentile={row.get('rank_percentile'):.2%}, cluster={row.get('behavior_cluster')}"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()

