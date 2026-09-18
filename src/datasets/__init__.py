"""Datasets module — normalized Parquet architecture for ARAUCARIA smart meter data.

Implements the layered architecture defined in docs/dataset_architecture_decision.md:

- ``schemas``  — PyArrow/Polars schema definitions for each table
- ``normalize`` — transform raw MDM JSON-per-day into atomic rows
- ``partitioning`` — Hive-style partitioning helpers
- ``writers``  — Parquet/CSV writers with manifest integration
"""
