"""Mixture of Experts (MoE) Gating Router.

Dynamically routes and weights predictions across:
1. Feeder-Level Expert (Local circuit topology & load characteristics)
2. Municipality/Regional Expert (City/Regional socioeconomic and climatic profile)
3. Global Copel Model (Broad distributional anchor and regularizer)

Calculates gating weights:
    w_feeder, w_mun, w_global  (sum to 1.0)
based on circuit statistical characteristics and data density.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl


class MoERouter:
    """Computes gating weights and routes consensus predictions across hierarchical experts."""

    def __init__(
        self,
        *,
        min_feeder_samples: int = 150,
        large_feeder_samples: int = 1000,
    ):
        self.min_feeder_samples = min_feeder_samples
        self.large_feeder_samples = large_feeder_samples

    def compute_gating_weights(
        self,
        *,
        feeder_sample_count: int,
        circuit_profile: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Determine gating weights for Feeder, Municipality, and Global models.

        Args:
            feeder_sample_count: Number of UCs available for this feeder.
            circuit_profile: Optional electrical profile of the circuit.

        Returns:
            Dict mapping model names to weights summing to 1.0.
        """
        # Baseline sizing
        if feeder_sample_count >= self.large_feeder_samples:
            w_feeder = 0.70
            w_mun = 0.20
            w_global = 0.10
        elif feeder_sample_count >= self.min_feeder_samples:
            # Linear interpolation between 0.35 and 0.70
            ratio = (feeder_sample_count - self.min_feeder_samples) / (
                self.large_feeder_samples - self.min_feeder_samples
            )
            w_feeder = 0.35 + 0.35 * ratio
            w_mun = 0.40 - 0.20 * ratio
            w_global = 0.25 - 0.15 * ratio
        else:
            # Sparse feeder: rely heavily on regional and global regularizers
            w_feeder = 0.15
            w_mun = 0.45
            w_global = 0.40

        # Adjust based on circuit profile if available
        if circuit_profile:
            # If high solar penetration or high unbalance, local feeder physics is more distinct
            gd_pen = float(circuit_profile.get("CIRCUIT_SOLAR_GD_PENETRATION", 0.0) or 0.0)
            if gd_pen > 0.05 and feeder_sample_count >= self.min_feeder_samples:
                w_feeder = min(0.85, w_feeder + 0.10)
                w_global = max(0.05, w_global - 0.05)
                w_mun = max(0.05, w_mun - 0.05)

        # Normalize to exactly 1.0
        total = w_feeder + w_mun + w_global
        return {
            "w_feeder": round(w_feeder / total, 3),
            "w_mun": round(w_mun / total, 3),
            "w_global": round(w_global / total, 3),
        }

    def combine_predictions(
        self,
        scores_feeder: np.ndarray | None,
        scores_mun: np.ndarray | None,
        scores_global: np.ndarray | None,
        weights: dict[str, float],
    ) -> np.ndarray:
        """Combine expert anomaly scores using gating weights.

        Handles missing experts gracefully by re-normalizing available weights.
        """
        w_f = weights.get("w_feeder", 0.33)
        w_m = weights.get("w_mun", 0.33)
        w_g = weights.get("w_global", 0.34)

        terms = []
        active_weights = []

        if scores_feeder is not None and len(scores_feeder) > 0:
            terms.append(scores_feeder * w_f)
            active_weights.append(w_f)
        if scores_mun is not None and len(scores_mun) > 0:
            terms.append(scores_mun * w_m)
            active_weights.append(w_m)
        if scores_global is not None and len(scores_global) > 0:
            terms.append(scores_global * w_g)
            active_weights.append(w_g)

        if not terms:
            raise ValueError("At least one expert score array must be provided.")

        total_weight = sum(active_weights)
        combined = sum(terms) / (total_weight if total_weight > 0 else 1.0)
        return np.round(np.clip(combined, 0.0, 1.0), 4)
