"""Reproducible Experimental Infrastructure, Contracts, and Audits for TCC AMI."""

from src.experiments.contracts import (
    ExperimentConfig,
    InvariantReport,
    KPIContract,
    KPI_REGISTRY,
    RunManifest,
    load_experiment_config,
    validate_population_partition,
    validate_portfolio_queue_sets,
    validate_score_and_kpi_integrity,
    validate_temporal_no_leakage,
)

__all__ = [
    "ExperimentConfig",
    "InvariantReport",
    "KPIContract",
    "KPI_REGISTRY",
    "RunManifest",
    "load_experiment_config",
    "validate_population_partition",
    "validate_portfolio_queue_sets",
    "validate_score_and_kpi_integrity",
    "validate_temporal_no_leakage",
]
