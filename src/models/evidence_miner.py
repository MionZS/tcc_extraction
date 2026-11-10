"""Unsupervised Anomaly Profiling, Sub-Clustering and Root-Cause Evidence Mining.

Extracts evidence from isolated anomalous meter populations:
1. Filters anomalous subsets from calibrated detector scores (S_composite, S_elec, S_tamper).
2. Performs sub-clustering (e.g. KMeans / Gaussian Mixture) on the anomalous subspace.
3. Quantifies feature divergence (Z-Score / Relative Lift) of each sub-cluster against the normal baseline.
4. Synthesizes explainable causal hypotheses (GD Clandestina, Medidor Travado, Desvio de Carga, Troca de Perfil).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class ClusterEvidence:
    """Diagnostic profile for an identified sub-cluster of anomalous meters."""

    cluster_id: int
    cluster_name: str
    sample_count: int
    sample_ratio: float
    top_divergent_features: list[dict[str, Any]]
    primary_hypothesis: str
    action_recommendation: str
    sample_ucs: list[str]


class AnomalyEvidenceMiner:
    """Mines explanatory signatures and root-cause evidence from anomalous smart meters."""

    def __init__(
        self,
        *,
        n_clusters: int = 4,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def mine_evidence(
        self,
        full_df: pl.DataFrame,
        *,
        score_col: str = "consensus_score",
        anomaly_flag_col: str = "is_anomaly",
        top_k_features: int = 4,
    ) -> list[ClusterEvidence]:
        """Sub-clusters anomalous meters and computes feature lifts against regular population."""
        if full_df.is_empty():
            return []

        # Identify numeric feature columns
        feature_cols = [
            c for c in full_df.columns
            if (c.startswith("x__") or c in ("HIST_RA_REVERSAL_RATIO_MAX", "HIST_CURRENT_IMBALANCE_MAX", "AVG_INTERVAL_COVERAGE"))
            and full_df[c].dtype.is_numeric()
        ]

        if not feature_cols:
            return []

        # Determine anomaly subset
        if anomaly_flag_col in full_df.columns:
            anom_df = full_df.filter(pl.col(anomaly_flag_col) == True)  # noqa: E712
            norm_df = full_df.filter(pl.col(anomaly_flag_col) == False)  # noqa: E712
        elif score_col in full_df.columns:
            cutoff = float(np.percentile(full_df[score_col].to_numpy(), 98.0))
            anom_df = full_df.filter(pl.col(score_col) >= cutoff)
            norm_df = full_df.filter(pl.col(score_col) < cutoff)
        else:
            return []

        if anom_df.height < 5:
            return []

        if norm_df.is_empty():
            norm_df = anom_df

        # Compute baseline medians and interquartile spans (IQR) from normal population
        norm_medians = {col: float(norm_df[col].drop_nulls().median() or 0.0) for col in feature_cols}
        norm_stds = {
            col: float(norm_df[col].drop_nulls().std() or 1.0)
            for col in feature_cols
        }
        # Safeguard against zero std
        norm_stds = {k: (v if v > 1e-4 else 1.0) for k, v in norm_stds.items()}

        # Vectorize anomalous subspace
        imputer = SimpleImputer(strategy="median")
        X_anom = imputer.fit_transform(anom_df.select(feature_cols).to_numpy())
        scaler = RobustScaler()
        X_anom_scaled = scaler.fit_transform(X_anom)

        k = min(self.n_clusters, max(2, anom_df.height // 10))
        clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = clusterer.fit_predict(X_anom_scaled)

        anom_with_cluster = anom_df.with_columns(pl.Series("sub_cluster_id", labels))
        total_anom = anom_df.height

        evidence_list: list[ClusterEvidence] = []
        uc_col = "id__uc_id" if "id__uc_id" in anom_df.columns else ("UC" if "UC" in anom_df.columns else None)

        for c_id in range(k):
            c_df = anom_with_cluster.filter(pl.col("sub_cluster_id") == c_id)
            c_count = c_df.height
            if c_count == 0:
                continue

            # Contrast cluster median vs normal median (Z-score lift)
            feature_lifts = []
            for col in feature_cols:
                c_med = float(c_df[col].drop_nulls().median() or 0.0)
                n_med = norm_medians[col]
                n_std = norm_stds[col]
                z_lift = (c_med - n_med) / n_std
                feature_lifts.append({
                    "feature": col,
                    "cluster_median": round(c_med, 4),
                    "normal_median": round(n_med, 4),
                    "z_lift": round(z_lift, 2),
                    "abs_lift": abs(z_lift),
                })

            feature_lifts.sort(key=lambda x: x["abs_lift"], reverse=True)
            top_div = feature_lifts[:top_k_features]

            # Formulate automatic physical/operational hypothesis
            top_f_names = [f["feature"].lower() for f in top_div]
            hyp, rec, c_name = self._formulate_hypothesis(top_div, c_df)

            sample_ucs = (
                c_df.get_column(uc_col).head(5).to_list()
                if uc_col and uc_col in c_df.columns
                else []
            )

            evidence_list.append(
                ClusterEvidence(
                    cluster_id=c_id,
                    cluster_name=c_name,
                    sample_count=c_count,
                    sample_ratio=c_count / max(1, total_anom),
                    top_divergent_features=top_div,
                    primary_hypothesis=hyp,
                    action_recommendation=rec,
                    sample_ucs=[str(u) for u in sample_ucs],
                )
            )

        evidence_list.sort(key=lambda x: x.sample_count, reverse=True)
        return evidence_list

    def _formulate_hypothesis(
        self,
        top_div: list[dict[str, Any]],
        cluster_df: pl.DataFrame,
    ) -> tuple[str, str, str]:
        """Translates top divergent feature signatures into explainable business categories."""
        feat_map = {f["feature"].lower(): f["z_lift"] for f in top_div}

        # Check for Reverse Active Energy injection (GD Clandestina / Prosumer)
        ra_keys = [k for k in feat_map if "reversal" in k or "ra_" in k]
        if ra_keys and any(feat_map[k] > 1.5 for k in ra_keys):
            return (
                "Injeção Ativa Reversa Elevada (Possível Microgeração / GD Solar sem Cadastro ou com Inversor Desregulado)",
                "Auditar cadastro CIS para presença de GD; vistoriar instalação em campo para conferir homologação de inversor.",
                "Cluster GD / Injeção Reversa",
            )

        # Check for Current Imbalance under load (Bypass / Tamper físico em uma das fases)
        c_imb_keys = [k for k in feat_map if "current_imbalance" in k or "c_imb" in k]
        if c_imb_keys and any(feat_map[k] > 1.5 for k in c_imb_keys):
            return (
                "Desequilíbrio Severo de Corrente sob Regime de Carga (Assinatura de Bypass de Fase ou Derivação Clandestina)",
                "Emitir Ordem de Serviço de Fiscalização Comercial para inspeção metrológica e ensaio de carga por alicate amperímetro.",
                "Cluster Assimetria de Carga / Bypass",
            )

        # Check for Severe Negative Slope with MAD drop (Desvio abrupto ou troca de inquilino)
        slope_keys = [k for k in feat_map if "slope" in k]
        mad_keys = [k for k in feat_map if "mad" in k]
        if slope_keys and any(feat_map[k] < -1.5 for k in slope_keys):
            return (
                "Degrau Descendente Brusco no Consumo Faturável (Queda Persistente pós-Intervenção ou Desocupação)",
                "Verificar data de troca de titularidade contratual no CIS; se não houver troca recente, priorizar inspeção de desvio.",
                "Cluster Degrau Descendente / Queda de Consumo",
            )

        # Check for Zero Consumption or Meter Stalled
        fa_keys = [k for k in feat_map if "fa_sum" in k or "fa_interval" in k]
        if fa_keys and any(feat_map[k] < -2.0 for k in fa_keys):
            return (
                "Consumo Quase Nulo com Tensão Ativa Normal (Possível Medidor Travado / Defeito de Bobina de Corrente)",
                "Fiscalizar calibração do elemento metrológico do medidor ou verificar corte administrativo não baixado no sistema.",
                "Cluster Consumo Zerado / Medidor Suspeito",
            )

        # Check for Degraded RF Telecom / Missing Packets
        cov_keys = [k for k in feat_map if "coverage" in k]
        if cov_keys and any(feat_map[k] < -1.5 for k in cov_keys):
            return (
                "Comunicação Severamente Degradada (Sombra de Rádio ou Falha em Concentrador RF Mesh)",
                "Encaminhar para a equipe de Automação/Telecomunicações para reposicionamento de repetidor ou troca de modem.",
                "Cluster Instabilidade Telecom / Mesh",
            )

        return (
            "Comportamento Multivariado Atípico Não-Linear (Divergência Eletrocomercial Conjunta)",
            "Auditar curvas de carga detalhadas de 5 minutos no visualizador de diagnóstico para análise individualizada.",
            "Cluster Anomalia Multivariada",
        )

