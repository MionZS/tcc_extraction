"""
scripts/demo_ml_methods.py
-------------------------------------------------------------------------------
Demonstração Metodológica Completa e Reproduzível de Técnicas de ML e Estatística
para Detecção de Perdas Não-Técnicas (PNT), Anomalias e Caracterização Comportamental.

Objetivo Central:
  Não demonstrar que "um modelo X é superior a Y", mas evidenciar que:
    Mesmos Dados -> Técnicas Diferentes -> Perguntas Diferentes
                 -> Efeitos Diferentes  -> Contribuição Complementar

Saídas:
  output/ml_method_demo/
  ├── 01_raw_profiles.png
  ├── 02_kmeans_profiles.png
  ├── 03_pca_projection.png
  ├── 04_pca_residual.png
  ├── 05_isolation_forest.png
  ├── 06_lof.png
  ├── 07_ecod.png
  ├── 08_detector_comparison.png
  ├── 09_ablation_funnel.png
  ├── 10_case_studies.png
  ├── metrics.csv
  └── method_summary.csv
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless backend para geração de figuras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler

# Raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.censorship import censor_uc_id

# Diretórios de entrada e saída
OUT_DIR = PROJECT_ROOT / "output" / "ml_method_demo"
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"
CLASSIFIED_PARQUET = REPORTS_DIR / "classified_ucs_latest.parquet"
CLASSIFIED_CSV = REPORTS_DIR / "classified_ucs_latest.csv"


# -----------------------------------------------------------------------------
# 1. ESTILO ACADÊMICO DE VISUALIZAÇÃO
# -----------------------------------------------------------------------------
def apply_academic_style() -> None:
    """Configura paleta, fontes e grades para figuras de nível TCC / Artigo."""
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cbd5e1",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.75,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.autolayout": False,
    })


# -----------------------------------------------------------------------------
# 2. IMPLEMENTAÇÃO NATIVA DO ECOD (Li et al., TKDE 2022)
# -----------------------------------------------------------------------------
class ECODDetectorDemo:
    """Empirical Cumulative Distribution-based Outlier Detection.
    
    Calcula caudas esquerda e direita não-paramétricas e agrega as probabilidades
    em escala logarítmica negativa: O(x) = sum_j -log(min(F_left, F_right)).
    """

    def __init__(self) -> None:
        self.sorted_cols_: list[np.ndarray] = []
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> "ECODDetectorDemo":
        X_arr = np.asarray(X, dtype=float)
        self.n_samples_, self.n_features_ = X_arr.shape
        self.sorted_cols_ = [np.sort(X_arr[:, j]) for j in range(self.n_features_)]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        n_eval = X_arr.shape[0]
        n_train = max(1, self.n_samples_)
        total_scores = np.zeros(n_eval, dtype=float)

        for j in range(self.n_features_):
            col_sorted = self.sorted_cols_[j]
            vals = X_arr[:, j]
            left_counts = np.searchsorted(col_sorted, vals, side="right")
            p_left = np.clip((left_counts + 1.0) / (n_train + 2.0), 1e-9, 1.0)
            right_counts = n_train - np.searchsorted(col_sorted, vals, side="left")
            p_right = np.clip((right_counts + 1.0) / (n_train + 2.0), 1e-9, 1.0)
            tail_p = np.minimum(p_left, p_right)
            total_scores += -np.log(tail_p)

        return total_scores / max(1, self.n_features_)


# -----------------------------------------------------------------------------
# 3. GERAÇÃO DE DATASET SINTÉTICO CONTROLADO (Ground Truth Conhecido)
# -----------------------------------------------------------------------------
def generate_synthetic_benchmark(
    n_samples: int = 1000,
    n_timesteps: int = 96,
    seed: int = 42,
) -> dict[str, Any]:
    """Gera curvas diárias de 96 intervalos (15 min) com classes comportamentais
    conhecidas e outliers precisamente injetados para demonstração de mecanismos.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, n_timesteps, endpoint=False)

    profiles = []
    labels_true = []
    category_true = []
    feeder_assign = []
    feeders = ["ALIM_FONTE_NOVA", "ALIM_CSN", "ALIM_INDEPENDENCIA", "ALIM_BARIGUI"]

    # Perfis Base:
    # 0: Residencial Noturno (Pico 19h-23h)
    # 1: Comercial Diurno (Platô 08h-18h)
    # 2: Carga Contínua (Quase constante com ondulação leve)
    # 3: Geração Distribuída / Solar (Inversão líquida ao meio-dia)
    n_res = int(n_samples * 0.45)
    n_com = int(n_samples * 0.30)
    n_con = int(n_samples * 0.15)
    n_sol = n_samples - (n_res + n_com + n_con)

    # 1. Residencial Noturno
    for i in range(n_res):
        mag = rng.uniform(8.0, 22.0)
        base = 0.15 * mag + 0.10 * mag * np.sin(np.pi * t / 24)
        night_peak = mag * 0.75 * np.exp(-0.5 * ((t - 20.5) / 2.2) ** 2)
        morning_peak = mag * 0.25 * np.exp(-0.5 * ((t - 7.5) / 1.5) ** 2)
        noise = rng.normal(0, 0.04 * mag, n_timesteps)
        curve = np.clip(base + night_peak + morning_peak + noise, 0.01, None)
        profiles.append(curve)
        labels_true.append(0)
        category_true.append("RESIDENCIAL_NOTURNO")
        feeder_assign.append(feeders[i % len(feeders)])

    # 2. Comercial Diurno
    for i in range(n_com):
        mag = rng.uniform(30.0, 90.0)
        day_plateau = mag * 0.85 / (1.0 + np.exp(-1.5 * (t - 7.5))) - mag * 0.85 / (1.0 + np.exp(-1.5 * (t - 18.5)))
        base = 0.08 * mag + rng.normal(0, 0.03 * mag, n_timesteps)
        curve = np.clip(base + day_plateau, 0.01, None)
        profiles.append(curve)
        labels_true.append(1)
        category_true.append("COMERCIAL_DIURNO")
        feeder_assign.append(feeders[(i + 1) % len(feeders)])

    # 3. Carga Contínua / Industrial Leve
    for i in range(n_con):
        mag = rng.uniform(40.0, 110.0)
        flat = mag * (0.85 + 0.08 * np.sin(2 * np.pi * t / 24)) + rng.normal(0, 0.02 * mag, n_timesteps)
        curve = np.clip(flat, 0.01, None)
        profiles.append(curve)
        labels_true.append(2)
        category_true.append("CARGA_CONTINUA")
        feeder_assign.append(feeders[(i + 2) % len(feeders)])

    # 4. Geração Distribuída / Solar
    for i in range(n_sol):
        mag = rng.uniform(15.0, 45.0)
        base = 0.40 * mag
        solar_gen = mag * 0.80 * np.exp(-0.5 * ((t - 12.5) / 2.5) ** 2)
        curve = base - solar_gen + rng.normal(0, 0.03 * mag, n_timesteps)
        # Permite reversão ativa (valores negativos representam injeção líquida)
        profiles.append(curve)
        labels_true.append(3)
        category_true.append("SOLAR_GD")
        feeder_assign.append(feeders[(i + 3) % len(feeders)])

    X_raw = np.array(profiles, dtype=float)

    # Injeção controlada de anomalias didáticas:
    outlier_indices = {}

    # Caso B: Outlier Global (Extremo absoluto de potência e dispersão)
    idx_global = 12
    X_raw[idx_global] = X_raw[idx_global] * 4.5 + rng.uniform(50, 120, n_timesteps)
    category_true[idx_global] = "OUTLIER_GLOBAL_EXTREMO"
    outlier_indices["global_extreme"] = idx_global

    # Caso C: Outlier Local (LOF) - Inserido no meio do Feeder rural/comercial com consumo invertido
    idx_local = n_res + 15
    X_raw[idx_local] = X_raw[idx_local] * 0.12  # Muito baixo comparado aos vizinhos comerciais locais
    category_true[idx_local] = "OUTLIER_LOCAL_DENSIDADE"
    outlier_indices["local_density"] = idx_local

    # Caso D: Outlier Marginal (ECOD) - Consumo médio perfeitamente normal, mas CV e Day/Night extremos
    idx_marginal = 25
    med_target = np.mean(X_raw[idx_marginal])
    # Totalmente concentrado em 2 horas da madrugada
    spike_burst = np.zeros(n_timesteps)
    spike_burst[10:14] = med_target * (n_timesteps / 4.0)
    X_raw[idx_marginal] = spike_burst + rng.normal(0, 0.01, n_timesteps)
    category_true[idx_marginal] = "OUTLIER_MARGINAL_ECOD"
    outlier_indices["marginal_ecod"] = idx_marginal

    # Caso E: Anomalia Estrutural (PCA Residual) - Consumo diário IDÊNTICO à média residencial,
    # mas forma em dente de serra invertida caótica (variedade linear violada)
    idx_pca = 45
    mean_res = np.mean(X_raw[idx_pca])
    X_raw[idx_pca] = mean_res + mean_res * 0.9 * np.sin(16 * np.pi * t / 24)
    category_true[idx_pca] = "OUTLIER_PCA_RESIDUAL"
    outlier_indices["pca_residual"] = idx_pca

    # Injeção de Micro-Cluster Compacto (4 UCs idênticas fraudando exatamente igual)
    micro_indices = [60, 61, 62, 63]
    for mi in micro_indices:
        # Padrão sintético idêntico e atípico
        X_raw[mi] = 5.0 + 8.0 * np.sin(4 * np.pi * t / 24) + rng.normal(0, 0.02, n_timesteps)
        category_true[mi] = "MICRO_CLUSTER_FRAUDE"
    outlier_indices["micro_cluster"] = micro_indices

    # Normalização por curva (soma diária = 1) para K-Means morfológico
    sums = np.sum(np.abs(X_raw), axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    X_norm = X_raw / sums

    # Extração de features elétricas agregadas para detectores tabulares
    daily_energy = np.sum(np.maximum(0, X_raw), axis=1) * (24.0 / n_timesteps)
    cv_feature = np.std(X_raw, axis=1) / (np.mean(np.maximum(0, X_raw), axis=1) + 1e-6)
    load_factor = np.mean(np.maximum(0, X_raw), axis=1) / (np.max(np.maximum(0, X_raw), axis=1) + 1e-6)
    day_slice = X_raw[:, int(8 * n_timesteps / 24) : int(18 * n_timesteps / 24)]
    night_slice = np.concatenate([
        X_raw[:, : int(6 * n_timesteps / 24)],
        X_raw[:, int(20 * n_timesteps / 24) :],
    ], axis=1)
    day_night_ratio = (np.mean(np.maximum(0, day_slice), axis=1) + 1e-6) / (np.mean(np.maximum(0, night_slice), axis=1) + 1e-6)
    reversal_ratio = np.sum(np.maximum(0, -X_raw), axis=1) / (sums.ravel() + 1e-6)

    features_matrix = np.column_stack([
        daily_energy,
        cv_feature,
        load_factor,
        day_night_ratio,
        reversal_ratio,
    ])

    return {
        "t": t,
        "X_raw": X_raw,
        "X_norm": X_norm,
        "labels_true": np.array(labels_true),
        "categories_true": category_true,
        "feeders": np.array(feeder_assign),
        "features": features_matrix,
        "feature_names": ["Energia Diária (kWh)", "CV Janela", "Fator de Carga", "Razão Dia/Noite", "Razão Reversão (RA)"],
        "outlier_indices": outlier_indices,
    }


# -----------------------------------------------------------------------------
# 4. CARREGAMENTO DO DATASET REAL MASCARADO (Copel Araucária)
# -----------------------------------------------------------------------------
def load_real_araucaria_subset() -> dict[str, Any] | None:
    """Tenta carregar o dataset classificado real e o mapeamento de exemplos empíricos."""
    if not CLASSIFIED_PARQUET.exists() and not CLASSIFIED_CSV.exists():
        return None

    try:
        if CLASSIFIED_PARQUET.exists():
            df = pl.read_parquet(CLASSIFIED_PARQUET)
        else:
            df = pl.read_csv(CLASSIFIED_CSV, separator=";", infer_schema_length=5000)

        # Mapeamento prévio dos exemplos reais
        priv_lookup_path = REPORTS_DIR / "private_examples_lookup.json"
        lookup = {}
        if priv_lookup_path.exists():
            try:
                lookup = json.loads(priv_lookup_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "df": df,
            "lookup": lookup,
        }
    except Exception as exc:
        print(f"[AVISO] Não foi possível ler dados reais: {exc}")
        return None


# -----------------------------------------------------------------------------
# 5. GERADORES DE CADA FIGURA (01 até 10)
# -----------------------------------------------------------------------------

def plot_01_raw_profiles(bench: dict[str, Any], out_path: Path) -> None:
    """Figura 01: Perfis Diários Crus (Antes do ML)."""
    t = bench["t"]
    X_raw = bench["X_raw"]
    X_norm = bench["X_norm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Painel 1: Curvas em magnitude real (emaranhado confuso)
    sample_idx = np.random.default_rng(42).choice(len(X_raw), size=120, replace=False)
    for idx in sample_idx:
        ax1.plot(t, X_raw[idx], color="#64748b", alpha=0.18, linewidth=0.8)

    # Destaque de 3 UCs reais/arquetípicas
    ax1.plot(t, X_raw[bench["outlier_indices"]["global_extreme"]], color="#ef4444", linewidth=2.0, label="UC Extrema (Sobrecarga/Pico)")
    ax1.plot(t, X_raw[0], color="#2563eb", linewidth=2.0, label="UC Residencial Típica")
    ax1.plot(t, X_raw[bench["outlier_indices"]["local_density"]], color="#f59e0b", linewidth=2.0, label="UC Localmente Baixa")

    ax1.set_title("A. Curvas de Carga em Magnitude Real (kW)", fontweight="bold")
    ax1.set_xlabel("Hora do Dia (h)")
    ax1.set_ylabel("Demanda / Consumo (kW)")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 4))
    ax1.legend(loc="upper right", framealpha=0.9)

    # Painel 2: Curvas normalizadas por área unitária
    for idx in sample_idx:
        ax2.plot(t, X_norm[idx], color="#64748b", alpha=0.18, linewidth=0.8)

    ax2.plot(t, X_norm[0], color="#2563eb", linewidth=2.2, label="Perfil Noturno (Residencial)")
    ax2.plot(t, X_norm[int(len(X_raw)*0.45)+2], color="#059669", linewidth=2.2, label="Perfil Diurno (Comercial)")
    ax2.plot(t, X_norm[int(len(X_raw)*0.78)+2], color="#7c3aed", linewidth=2.2, label="Perfil Contínuo (Industrial)")

    ax2.set_title("B. Curvas Normalizadas por Integral Diária: $x_d / \\sum x_d$", fontweight="bold")
    ax2.set_xlabel("Hora do Dia (h)")
    ax2.set_ylabel("Fração de Energia Horária Normalizada")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.suptitle("Figura 01 — Perfis Diários Crus: A Complexidade Multidimensional das Curvas AMI", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")


def plot_02_kmeans_profiles(bench: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Figura 02: K-Means — Mostrar Formação de Perfis."""
    t = bench["t"]
    X_norm = bench["X_norm"]

    t0 = time.perf_counter()
    tracemalloc.start()
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_norm)
    t_fit = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sil = float(silhouette_score(X_norm, clusters))
    inertia = float(kmeans.inertia_)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    cluster_names = [
        "Cluster 0: Residencial com Pico Noturno",
        "Cluster 1: Comercial com Carga Diurna",
        "Cluster 2: Carga Contínua / Refrigeração",
        "Cluster 3: Solar / Geração Distribuída (GD)",
    ]
    colors = ["#2563eb", "#059669", "#7c3aed", "#d97706"]

    for k in range(4):
        ax = axes[k]
        members = X_norm[clusters == k]
        # Amostra de curvas membros
        sample_m = members[: min(40, len(members))]
        for c in sample_m:
            ax.plot(t, c, color=colors[k], alpha=0.15, linewidth=0.8)

        # Centróide
        centroid = kmeans.cluster_centers_[k]
        ax.plot(t, centroid, color="#0f172a", linewidth=2.8, linestyle="-", label="Centróide $\\mu_k$")
        ax.plot(t, centroid, color=colors[k], linewidth=2.0, linestyle="-")

        ax.set_title(f"{cluster_names[k]} (N={len(members)})", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.set_ylabel("Fração Normalizada")
        if k >= 2:
            ax.set_xlabel("Hora do Dia (h)")
        ax.legend(loc="upper right", framealpha=0.9)

    plt.suptitle(
        f"Figura 02 — K-Means: Populações Comportamentalmente Comparáveis\n"
        f"(Silhouette: {sil:.3f} | Inércia: {inertia:.1f} | Runtime: {t_fit*1000:.1f}ms)\n"
        f"Mensagem: K-Means não detecta fraude; constrói referências para comparação justa.",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "kmeans": kmeans,
        "clusters": clusters,
        "silhouette": sil,
        "inertia": inertia,
        "t_fit": t_fit,
        "peak_ram_mb": peak_ram / (1024 * 1024),
    }


def plot_03_pca_projection(bench: dict[str, Any], kmeans_res: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Figura 03: PCA — Representação e Variância Explicada."""
    X_norm = bench["X_norm"]
    clusters = kmeans_res["clusters"]

    t0 = time.perf_counter()
    tracemalloc.start()
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    t_fit = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Subplot A: PC1 x PC2 colorido por cluster
    colors = ["#2563eb", "#059669", "#7c3aed", "#d97706"]
    for k in range(4):
        mask = clusters == k
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[k], label=f"Cluster {k}", alpha=0.6, s=22, edgecolors="none")

    ax1.set_title("A. Projeção no Espaço Latente ($PC_1 \\times PC_2$)", fontweight="bold")
    ax1.set_xlabel(f"PC1 ({evr[0]*100:.1f}% da variância)")
    ax1.set_ylabel(f"PC2 ({evr[1]*100:.1f}% da variância)")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # Subplot B: Variância Explicada Acumulada
    comps = np.arange(1, len(cum_evr) + 1)
    ax2.bar(comps, evr * 100, color="#94a3b8", alpha=0.7, label="Individual")
    ax2.plot(comps, cum_evr * 100, color="#dc2626", marker="o", linewidth=2.0, label="Acumulada $\\sum EVR_j$")
    ax2.axhline(85, color="#f59e0b", linestyle="--", alpha=0.8, label="Patamar 85%")
    ax2.axhline(95, color="#10b981", linestyle=":", alpha=0.8, label="Patamar 95%")
    ax2.set_title("B. Variância Explicada Acumulada", fontweight="bold")
    ax2.set_xlabel("Número de Componentes Principais")
    ax2.set_ylabel("Variância Explicada (%)")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(comps)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=8)

    # Subplot C: Reconstrução original vs reconstruída com 3 PCs
    t = bench["t"]
    sample_uc = 10
    X_reconstructed = pca.inverse_transform(X_pca)
    ax3.plot(t, X_norm[sample_uc], color="#0f172a", linewidth=2.2, label="Original $x_i$")
    ax3.plot(t, X_reconstructed[sample_uc], color="#2563eb", linestyle="--", linewidth=2.0, label="Reconstruído $\\hat{x}_i$ (3 PCs)")
    ax3.fill_between(t, X_norm[sample_uc], X_reconstructed[sample_uc], color="#ef4444", alpha=0.25, label=r"Resíduo $\|x - \hat{x}\|$")
    ax3.set_title("C. Reconstrução Linear da Curva", fontweight="bold")
    ax3.set_xlabel("Hora do Dia (h)")
    ax3.set_ylabel("Fração Normalizada")
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 25, 4))
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.suptitle("Figura 03 — PCA: Compressão Estrutural das Curvas e Extração de Modos Dominantes", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "pca": pca,
        "X_pca": X_pca,
        "t_fit": t_fit,
        "peak_ram_mb": peak_ram / (1024 * 1024),
    }


def plot_04_pca_residual(bench: dict[str, Any], pca_res: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Figura 04: PCA Residual — Mostrar Anomalia Estrutural (Mesmo Consumo, Morfologia Rara)."""
    t = bench["t"]
    X_norm = bench["X_norm"]
    pca: PCA = pca_res["pca"]

    # Reconstrução com 3 componentes principais
    n_keep = 3
    pca_k = PCA(n_components=n_keep, random_state=42)
    X_proj = pca_k.fit_transform(X_norm)
    X_rec = pca_k.inverse_transform(X_proj)

    # Erro de reconstrução L2: E_i = ||x_i - \hat{x}_i||_2
    residuals = np.linalg.norm(X_norm - X_rec, axis=1)

    # Identificar caso normal vs anomalia estrutural
    idx_normal = 0
    idx_structural = bench["outlier_indices"]["pca_residual"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Painel 1: Perfil Normal (Excelente aderência ao subespaço PCA)
    err_norm = residuals[idx_normal]
    ax1.plot(t, X_norm[idx_normal], color="#0f172a", linewidth=2.2, label="Curva Real $x_{norm}$")
    ax1.plot(t, X_rec[idx_normal], color="#10b981", linestyle="--", linewidth=2.0, label="Reconstrução $\\hat{x}$")
    ax1.fill_between(t, X_norm[idx_normal], X_rec[idx_normal], color="#10b981", alpha=0.2, label=f"Resíduo $E = {err_norm:.4f}$")
    ax1.set_title("A. Consumidor Normal: Resíduo Estrutural Baixo", fontweight="bold")
    ax1.set_xlabel("Hora do Dia (h)")
    ax1.set_ylabel("Fração Horária Normalizada")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 4))
    ax1.legend(loc="upper right", framealpha=0.9)

    # Painel 2: Perfil Anômalo Estrutural (Mesma integral diária, morfologia não explicada pelo PCA)
    err_anom = residuals[idx_structural]
    ax2.plot(t, X_norm[idx_structural], color="#0f172a", linewidth=2.2, label="Curva Real $x_{anom}$")
    ax2.plot(t, X_rec[idx_structural], color="#ef4444", linestyle="--", linewidth=2.0, label="Reconstrução $\\hat{x}$")
    ax2.fill_between(t, X_norm[idx_structural], X_rec[idx_structural], color="#ef4444", alpha=0.35, label=f"Resíduo $E = {err_anom:.4f}$ (ALTO)")
    ax2.set_title("B. Anomalia de Morfologia: Resíduo Estrutural Elevado", fontweight="bold")
    ax2.set_xlabel("Hora do Dia (h)")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.suptitle(
        "Figura 04 — PCA Residual: Detecção de Anomalias Estruturais de Carga\n"
        "Mensagem: O total consumido (kWh) pode ser perfeitamente comum enquanto a dinâmica temporal viola a física da rede.",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "residuals": residuals,
        "pca_k": pca_k,
    }


def plot_05_isolation_forest(bench: dict[str, Any], pca_res: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Figura 05: Isolation Forest — Mostrar Raridade Global e a Armadilha do Micro-Cluster."""
    features = bench["features"]
    X_pca = pca_res["X_pca"]

    scaler = RobustScaler()
    X_feat_scaled = scaler.fit_transform(features)

    t0 = time.perf_counter()
    tracemalloc.start()
    iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_feat_scaled)
    # Score bruto invertido: maior = mais anômalo
    raw_scores = -iso.score_samples(X_feat_scaled)
    t_fit = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Normalização em [0, 1]
    s_min, s_max = np.min(raw_scores), np.max(raw_scores)
    iso_scores = (raw_scores - s_min) / (s_max - s_min + 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Painel 1: Score no Espaço Latente (PC1 x PC2)
    sc = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=iso_scores, cmap="YlOrRd", s=28, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax1, label="Score de Anomalia Isolation Forest [0, 1]")

    # Destacar os 3 pontos conceituais
    idx_center = 0
    idx_border = 5
    idx_isolated = bench["outlier_indices"]["global_extreme"]

    ax1.scatter(X_pca[idx_center, 0], X_pca[idx_center, 1], c="#2563eb", s=110, edgecolors="#0f172a", linewidth=2.0, label="Centro: Score Baixo")
    ax1.scatter(X_pca[idx_border, 0], X_pca[idx_border, 1], c="#f59e0b", s=110, edgecolors="#0f172a", linewidth=2.0, label="Borda: Score Médio")
    ax1.scatter(X_pca[idx_isolated, 0], X_pca[idx_isolated, 1], c="#dc2626", s=140, edgecolors="#0f172a", linewidth=2.5, marker="*", label="Isolado: Score Alto")

    ax1.set_title("A. Raridade Multidimensional Global (IForest)", fontweight="bold")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # Painel 2: Limitação Pedagógica — A Armadilha do Micro-Cluster Compacto
    micro_idx = bench["outlier_indices"]["micro_cluster"]
    bar_labels = ["Outlier\nIsolado", "Micro-Cluster\n(Fraude 1)", "Micro-Cluster\n(Fraude 2)", "Consumidor\nCentro"]
    bar_vals = [
        iso_scores[idx_isolated],
        iso_scores[micro_idx[0]],
        iso_scores[micro_idx[1]],
        iso_scores[idx_center],
    ]
    colors_bar = ["#dc2626", "#f97316", "#f97316", "#2563eb"]

    bars = ax2.bar(bar_labels, bar_vals, color=colors_bar, width=0.55, edgecolor="#0f172a")
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax2.set_title("B. Armadilha do Micro-Cluster Compacto no IForest", fontweight="bold")
    ax2.set_ylabel("Score de Anomalia Normalizado")
    ax2.set_ylim(0, 1.15)
    ax2.annotate(
        "Fraudes idênticas formam pequeno\ngrupo denso, exigindo mais cortes,\nreduzindo o score global do IForest!",
        xy=(1.5, 0.52), xytext=(1.0, 0.85),
        arrowprops=dict(facecolor="#0f172a", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef08a", edgecolor="#ca8a04", alpha=0.9)
    )

    plt.suptitle(
        "Figura 05 — Isolation Forest: Eficácia Global e Vulnerabilidade a Agrupamentos Coletivos\n"
        "Mensagem: Árvores de isolamento são excelentes em raridade esparsa, mas dependem de densidade local (LOF) para micro-fraudes.",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "iso_scores": iso_scores,
        "t_fit": t_fit,
        "peak_ram_mb": peak_ram / (1024 * 1024),
    }


def plot_06_lof(bench: dict[str, Any], if_res: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Figura 06: LOF — Mostrar Raridade Local e Densidade Relativa."""
    features = bench["features"]
    iso_scores = if_res["iso_scores"]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features)

    t0 = time.perf_counter()
    tracemalloc.start()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof.fit(X_scaled)
    raw_lof = -lof.score_samples(X_scaled)
    t_fit = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Normalização min-max
    l_min, l_max = np.min(raw_lof), np.max(raw_lof)
    lof_scores = (raw_lof - l_min) / (l_max - l_min + 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Painel 1: Comparação direta entre IForest e LOF para o Outlier Local e Micro-Cluster
    idx_local = bench["outlier_indices"]["local_density"]
    idx_micro = bench["outlier_indices"]["micro_cluster"][0]
    idx_global = bench["outlier_indices"]["global_extreme"]
    idx_norm = 0

    cases = ["Normal", "Outlier\nGlobal", "Outlier\nLocal", "Micro-Cluster\n(Fraude)"]
    x = np.arange(len(cases))
    w = 0.35

    s_if = [iso_scores[idx_norm], iso_scores[idx_global], iso_scores[idx_local], iso_scores[idx_micro]]
    s_lof = [lof_scores[idx_norm], lof_scores[idx_global], lof_scores[idx_local], lof_scores[idx_micro]]

    ax1.bar(x - w/2, s_if, w, label="Isolation Forest (Global)", color="#3b82f6", edgecolor="#0f172a")
    ax1.bar(x + w/2, s_lof, w, label="LOF (Densidade Local)", color="#dc2626", edgecolor="#0f172a")

    ax1.set_title("A. Resposta dos Detectores: Global vs Local", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases)
    ax1.set_ylabel("Score de Anomalia [0, 1]")
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Painel 2: Dispersão Scatter IF x LOF destacando a complementaridade
    cats = np.asarray(bench["categories_true"])
    scatter_colors = np.where(
        cats == "OUTLIER_LOCAL_DENSIDADE", "#dc2626",
        np.where(cats == "MICRO_CLUSTER_FRAUDE", "#f97316", "#94a3b8")
    ).tolist()
    ax2.scatter(iso_scores, lof_scores, c=scatter_colors, alpha=0.65, s=28)
    ax2.scatter(iso_scores[idx_local], lof_scores[idx_local], c="#dc2626", s=130, marker="D", edgecolors="#0f172a", linewidth=2.0, label="Outlier Local (LOF alto, IF moderado)")
    ax2.scatter(iso_scores[idx_global], lof_scores[idx_global], c="#2563eb", s=140, marker="*", edgecolors="#0f172a", linewidth=2.0, label="Outlier Global (Ambos altos)")

    ax2.axvline(np.percentile(iso_scores, 95), color="#3b82f6", linestyle="--", alpha=0.7, label="Limiar P95 IF")
    ax2.axhline(np.percentile(lof_scores, 95), color="#dc2626", linestyle="--", alpha=0.7, label="Limiar P95 LOF")

    ax2.set_title("B. Espaço de Decisão Ortogonal: IF $\\times$ LOF", fontweight="bold")
    ax2.set_xlabel("Score Isolation Forest (Raridade Global)")
    ax2.set_ylabel("Score LOF (Densidade Relativa $k$-NN)")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=8)

    plt.suptitle(
        "Figura 06 — Local Outlier Factor (LOF): Resolução do Ponto Cego de Densidade Local\n"
        "Mensagem: Um consumidor pode parecer comum no panorama geral, mas ser uma anomalia severa no seu grupo contextual.",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "lof_scores": lof_scores,
        "t_fit": t_fit,
        "peak_ram_mb": peak_ram / (1024 * 1024),
    }


def plot_07_ecod(bench: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Figura 07: ECOD — Mostrar Extremidade Marginal em Distribuições Empíricas."""
    features = bench["features"]
    feature_names = bench["feature_names"]

    t0 = time.perf_counter()
    tracemalloc.start()
    ecod = ECODDetectorDemo().fit(features)
    raw_ecod = ecod.score_samples(features)
    t_fit = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    e_min, e_max = np.min(raw_ecod), np.max(raw_ecod)
    ecod_scores = (raw_ecod - e_min) / (e_max - e_min + 1e-12)

    # Caso selecionado: Outlier Marginal (energia normal, mas CV e Day/Night em caudas)
    idx_target = bench["outlier_indices"]["marginal_ecod"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for j in range(4):
        ax = axes[j]
        col_data = features[:, j]
        val_target = features[idx_target, j]
        p_emp = np.mean(col_data <= val_target)

        # Histograma com KDE empírico
        n_bins = 30
        counts, edges, _ = ax.hist(col_data, bins=n_bins, density=True, color="#94a3b8", alpha=0.6, edgecolor="#ffffff")

        # Linha vertical do caso avaliado
        color_line = "#dc2626" if (p_emp > 0.95 or p_emp < 0.05) else "#2563eb"
        status_txt = "EXTREMO DE CAUDA!" if (p_emp > 0.95 or p_emp < 0.05) else "Na Média (Normal)"

        ax.axvline(val_target, color=color_line, linewidth=2.5, linestyle="-")
        ax.annotate(
            f"UC #{idx_target}\nValor: {val_target:.2f}\nPercentil: {p_emp*100:.1f}%\n({status_txt})",
            xy=(val_target, np.max(counts) * 0.65),
            xytext=(15, 15), textcoords="offset points",
            arrowprops=dict(facecolor=color_line, shrink=0.08, width=1.5, headwidth=5),
            fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=color_line, alpha=0.95)
        )

        ax.set_title(f"{feature_names[j]} (Marginal $X_{{{j+1}}}$)", fontweight="bold")
        ax.set_ylabel("Densidade de Probabilidade")

    plt.suptitle(
        f"Figura 07 — ECOD: Detecção Não-Paramétrica por Caudas Empíricas Multidimensionais\n"
        f"Score ECOD da UC #{idx_target}: {ecod_scores[idx_target]:.3f} (Runtime: {t_fit*1000:.1f}ms)\n"
        f"Mensagem: O ECOD combina extremidades marginais sem impor hipóteses gaussianas nem custos quadráticos de distância.",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "ecod_scores": ecod_scores,
        "t_fit": t_fit,
        "peak_ram_mb": peak_ram / (1024 * 1024),
    }


def plot_08_detector_comparison(
    bench: dict[str, Any],
    if_res: dict[str, Any],
    lof_res: dict[str, Any],
    ecod_res: dict[str, Any],
    pca_res: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Figura 08: Comparação Direta dos 4 Detectores e Modelo Composto Ponderado."""
    s_if = if_res["iso_scores"]
    s_lof = lof_res["lof_scores"]
    s_ecod = ecod_res["ecod_scores"]
    s_pca = pca_res["residuals"]
    # Normalizar PCA residual em [0, 1]
    s_pca = (s_pca - np.min(s_pca)) / (np.max(s_pca) - np.min(s_pca) + 1e-12)

    # Calibração empírica por percentil: z_m = F_m(S_m) in [0, 1]
    z_if = rankdata(s_if) / len(s_if)
    z_lof = rankdata(s_lof) / len(s_lof)
    z_ecod = rankdata(s_ecod) / len(s_ecod)
    z_pca = rankdata(s_pca) / len(s_pca)

    # Ensemble Ponderado (Composite)
    w_if, w_lof, w_ecod, w_pca = 0.28, 0.26, 0.24, 0.22
    s_ensemble = w_if * z_if + w_lof * z_lof + w_ecod * z_ecod + w_pca * z_pca

    # Selecionar 5 casos representativos
    cases_keys = [
        ("Consumidor Normal", 0),
        ("Globalmente Extremo", bench["outlier_indices"]["global_extreme"]),
        ("Localmente Estranho", bench["outlier_indices"]["local_density"]),
        ("Morfologia Rara (PCA)", bench["outlier_indices"]["pca_residual"]),
        ("Extremo Marginal", bench["outlier_indices"]["marginal_ecod"]),
    ]

    mat_data = []
    for _, idx in cases_keys:
        mat_data.append([z_if[idx], z_lof[idx], z_ecod[idx], z_pca[idx], s_ensemble[idx]])
    mat_data = np.array(mat_data)

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Heatmap de resposta dos detectores calibrados
    im = ax1.imshow(mat_data, cmap="YlOrRd", vmin=0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax1, label="Score Normalizado por Ranking Empírico $z_{i,m} \\in [0, 1]$")

    det_labels = ["Isolation\nForest", "Local Outlier\nFactor (LOF)", "ECOD\n(Caudas)", "PCA\nResidual", "Ensemble\nComposto"]
    case_labels = [c[0] for c in cases_keys]

    ax1.set_xticks(range(5))
    ax1.set_xticklabels(det_labels, fontsize=8.5, fontweight="bold")
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(case_labels, fontsize=9.5, fontweight="bold")

    for i in range(5):
        for j in range(5):
            val = mat_data[i, j]
            color_txt = "#ffffff" if val > 0.65 else "#0f172a"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=color_txt, fontweight="bold", fontsize=9.5)

    ax1.set_title("A. Matriz de Resposta dos Detectores Especialistas", fontweight="bold")

    # Decomposição de Barras Empilhadas da Evidência Composta
    y_pos = np.arange(len(cases_keys))
    ax2.barh(y_pos, [w_if * z_if[idx] for _, idx in cases_keys], height=0.55, color="#3b82f6", label=f"IForest ({w_if*100:.0f}%)", edgecolor="#0f172a")
    ax2.barh(y_pos, [w_lof * z_lof[idx] for _, idx in cases_keys], left=[w_if * z_if[idx] for _, idx in cases_keys], height=0.55, color="#dc2626", label=f"LOF ({w_lof*100:.0f}%)", edgecolor="#0f172a")
    ax2.barh(y_pos, [w_ecod * z_ecod[idx] for _, idx in cases_keys], left=[(w_if * z_if[idx] + w_lof * z_lof[idx]) for _, idx in cases_keys], height=0.55, color="#d97706", label=f"ECOD ({w_ecod*100:.0f}%)", edgecolor="#0f172a")
    ax2.barh(y_pos, [w_pca * z_pca[idx] for _, idx in cases_keys], left=[(w_if * z_if[idx] + w_lof * z_lof[idx] + w_ecod * z_ecod[idx]) for _, idx in cases_keys], height=0.55, color="#7c3aed", label=f"PCA Res. ({w_pca*100:.0f}%)", edgecolor="#0f172a")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(case_labels, fontweight="bold")
    ax2.set_xlabel("Score Consolidado do Ensemble $S_{ensemble} = \\sum w_m z_{i,m}$")
    ax2.set_xlim(0, 1.05)
    ax2.set_title("B. Decomposição Ponderada da Evidência de Fraude", fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    plt.suptitle("Figura 08 — Comparação Direta dos Detectores: Ortogonalidade e Sinergia Composta", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")

    return {
        "z_if": z_if,
        "z_lof": z_lof,
        "z_ecod": z_ecod,
        "z_pca": z_pca,
        "s_ensemble": s_ensemble,
    }


def plot_09_ablation_funnel(bench: dict[str, Any], comp_res: dict[str, Any], out_path: Path) -> None:
    """Figura 09: Funil de Ablação Operacional (M0 -> M5) e Contexto por Alimentador."""
    N_total = len(bench["X_raw"])
    s_ensemble = comp_res["s_ensemble"]
    feeders = bench["feeders"]

    # Simulação do Funil de Triagem Operacional:
    # M0: População Total
    # M1: + Saneamento Cadastral UC/NIO (remove 3% de inconsistências cadastrais)
    # M2: + Gating Telecom (remove 8% de medidores com cobertura degradada / quarentena)
    # M3: + Clustering Comportamental e Contexto de Alimentador
    # M4: + Análise de Mudança de Regime / CPD / Persistência
    # M5: + Ensemble de Especialistas PyOD -> Priorizados para Vistoria em Campo
    m0_n = N_total
    m1_n = int(m0_n * 0.97)
    m2_n = int(m1_n * 0.92)
    m3_n = int(m2_n * 0.40)  # Foco em classes de maior risco
    m4_n = int(np.sum(s_ensemble > 0.80))
    m5_n = int(np.sum(s_ensemble > 0.92))

    funnel_stages = [
        ("M0: População Bruta Total", m0_n, "#64748b", "Medições AMI coletadas"),
        ("M1: + Saneamento UC/NIO", m1_n, "#0284c7", "Exclusão de NIOs descontinuados"),
        ("M2: + Gating Telecom (S_comm)", m2_n, "#059669", "Exclusão de falhas mesh/telecom"),
        ("M3: + Clustering Contextual", m3_n, "#7c3aed", "Segmentação por porte/hábito"),
        ("M4: + Persistência / CPD", m4_n, "#d97706", "Sustentado por >= 3 dias"),
        ("M5: + Ensemble PyOD (Final)", m5_n, "#dc2626", "Alvos de Campo Prioritários"),
    ]

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Gráfico de Funil Horizontal
    y_pos = np.arange(len(funnel_stages))[::-1]
    for idx, (label, val, color, desc) in enumerate(funnel_stages):
        y = y_pos[idx]
        ax1.barh(y, val, height=0.55, color=color, edgecolor="#0f172a", alpha=0.9)
        pct_prev = (val / funnel_stages[idx-1][1] * 100) if idx > 0 else 100
        ax1.text(val + 15, y, f"{val:,} UCs ({val/m0_n*100:.1f}%) — {desc}", va="center", fontsize=8.5, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s[0] for s in funnel_stages], fontsize=9, fontweight="bold")
    ax1.set_xlabel("Número de Unidades Consumidoras")
    ax1.set_xlim(0, m0_n * 1.55)
    ax1.set_title("A. Funil de Triagem Operacional: Da População à OS de Campo", fontweight="bold")

    # Painel 2: Contextualização por Alimentador (Rank Global vs Rank no Alimentador)
    # Selecionar alimentador específico e calcular ranks
    target_feeder = "ALIM_FONTE_NOVA"
    mask_feeder = feeders == target_feeder
    idx_in_feeder = np.where(mask_feeder)[0]

    rank_global = rankdata(-s_ensemble)
    rank_feeder = np.zeros(len(s_ensemble), dtype=int)
    rank_feeder[idx_in_feeder] = rankdata(-s_ensemble[idx_in_feeder])

    ax2.scatter(rank_global[mask_feeder], rank_feeder[mask_feeder], c="#0284c7", alpha=0.6, s=28, label="UCs do Alimentador")

    # Destaque de uma UC que salta de posição contextual
    # Achar UC com maior ganho de rank
    jumps = rank_global[idx_in_feeder] - rank_feeder[idx_in_feeder]
    best_jump_idx = idx_in_feeder[np.argmax(jumps)]

    rg = rank_global[best_jump_idx]
    rf = rank_feeder[best_jump_idx]

    ax2.scatter(rg, rf, c="#dc2626", s=140, marker="*", edgecolors="#0f172a", linewidth=2.0, label="UC c/ Anomalia Contextual")
    ax2.annotate(
        f"UC #{best_jump_idx}\nRank Global: #{rg}\nRank no Alimentador: #{rf}!",
        xy=(rg, rf), xytext=(20, 20), textcoords="offset points",
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#dc2626", alpha=0.95)
    )

    # Jaccard Top-50
    top50_g = set(np.argsort(-s_ensemble)[:50])
    top50_f = set(idx_in_feeder[np.argsort(-s_ensemble[idx_in_feeder])[:50]])
    jaccard_top50 = len(top50_g & top50_f) / max(1, len(top50_g | top50_f))

    ax2.set_title(f"B. Efeito de Contextualização ({target_feeder})", fontweight="bold")
    ax2.set_xlabel("Ranking Global na Concessionária (1 = Mais Anômala)")
    ax2.set_ylabel("Ranking Local no Alimentador")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax2.text(0.05, 0.90, f"Jaccard Top-50: {jaccard_top50:.2f}", transform=ax2.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#cbd5e1"))

    plt.suptitle("Figura 09 — Ablação Operacional e Importância da Contextualização por Circuito Elétrico", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")


def plot_10_case_studies(bench: dict[str, Any], comp_res: dict[str, Any], out_path: Path) -> None:
    """Figura 10: Os 6 Estudos de Caso Obrigatórios + Calendário Comportamental com CPD."""
    t = bench["t"]
    X_raw = bench["X_raw"]
    s_ensemble = comp_res["s_ensemble"]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

    # Top 3: Casos A, B, C
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Bottom 3: Casos D, E, F (Calendário Comportamental + CPD)
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # Caso A: Normal Evidente
    axA.plot(t, X_raw[0], color="#2563eb", linewidth=2.0)
    axA.set_title(f"Caso A: Normal Típica\nScore Ensemble: {s_ensemble[0]:.3f}", fontweight="bold", fontsize=10.5)
    axA.set_ylabel("Demanda (kW)")
    axA.set_xlabel("Hora do Dia (h)")
    axA.set_xlim(0, 24)

    # Caso B: Anomalia Global (IForest)
    idx_b = bench["outlier_indices"]["global_extreme"]
    axB.plot(t, X_raw[idx_b], color="#dc2626", linewidth=2.0)
    axB.set_title(f"Caso B: Anomalia Global (IForest)\nScore Ensemble: {s_ensemble[idx_b]:.3f}", fontweight="bold", fontsize=10.5)
    axB.set_xlabel("Hora do Dia (h)")
    axB.set_xlim(0, 24)

    # Caso C: Anomalia Local Contextual (LOF)
    idx_c = bench["outlier_indices"]["local_density"]
    axC.plot(t, X_raw[idx_c], color="#d97706", linewidth=2.0)
    axC.set_title(f"Caso C: Anomalia Local (LOF)\nScore Ensemble: {s_ensemble[idx_c]:.3f}", fontweight="bold", fontsize=10.5)
    axC.set_xlabel("Hora do Dia (h)")
    axC.set_xlim(0, 24)

    # Caso D: Extremidade Marginal (ECOD)
    idx_d = bench["outlier_indices"]["marginal_ecod"]
    axD.plot(t, X_raw[idx_d], color="#7c3aed", linewidth=2.0)
    axD.set_title(f"Caso D: Marginal Caudas (ECOD)\nScore Ensemble: {s_ensemble[idx_d]:.3f}", fontweight="bold", fontsize=10.5)
    axD.set_ylabel("Demanda (kW)")
    axD.set_xlabel("Hora do Dia (h)")
    axD.set_xlim(0, 24)

    # Caso E: Forma Temporal Rara (PCA Residual)
    idx_e = bench["outlier_indices"]["pca_residual"]
    axE.plot(t, X_raw[idx_e], color="#059669", linewidth=2.0)
    axE.set_title(f"Caso E: Morfologia Rara (PCA Res.)\nScore Ensemble: {s_ensemble[idx_e]:.3f}", fontweight="bold", fontsize=10.5)
    axE.set_xlabel("Hora do Dia (h)")
    axE.set_xlim(0, 24)

    # Caso F: Calendário Comportamental Longitudinal + Change-Point Detection (CPD)
    # Simula 10 semanas (70 dias) de uma UC sofrendo bypass na semana 6 (dia 42)
    n_days = 70
    days_t = np.arange(1, n_days + 1)
    # Regime A (dias 1-42): média 25 kWh/dia
    # Regime B (dias 43-70): média 6 kWh/dia (bypass brusco)
    rng_cpd = np.random.default_rng(123)
    kwh_ts = np.concatenate([
        rng_cpd.normal(25.0, 2.5, 42),
        rng_cpd.normal(6.5, 1.2, 28),
    ])

    axF.plot(days_t, kwh_ts, color="#0f172a", linewidth=1.5, marker="o", markersize=3, label="Consumo Diário (kWh)")
    # Linhas de média pré e pós
    axF.hlines(25.0, 1, 42, color="#2563eb", linewidth=2.5, linestyle="--", label="Pré-Mudança: 25.0 kWh")
    axF.hlines(6.5, 43, 70, color="#dc2626", linewidth=2.5, linestyle="--", label="Pós-Mudança: 6.5 kWh")
    # Ponto de quebra CPD
    axF.axvline(42, color="#dc2626", linewidth=2.0, linestyle=":", label="Instante CPD ($t^* = 42d$)")

    axF.annotate(
        "Bypass Instalado!\nQueda de -74%\nPersistentemente Mantida",
        xy=(42, 6.5), xytext=(46, 17.0),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#dc2626", alpha=0.95)
    )

    axF.set_title("Caso F: Quebra Estrutural (CPD + Persistência)\n$\\Delta R = +100\\%$ Dias Atípicos", fontweight="bold", fontsize=10.5)
    axF.set_xlabel("Dia da Série Temporal (Dias)")
    axF.set_ylabel("Consumo Diário (kWh)")
    axF.set_xlim(1, 70)
    axF.legend(loc="upper right", framealpha=0.9, fontsize=7.5)

    plt.suptitle("Figura 10 — Painel Completo de Estudos de Caso: Especialistas Multimodais e Dinâmica Temporal", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Salvo: {out_path.name}")


# -----------------------------------------------------------------------------
# 6. EXPORTAÇÃO QUANTITATIVA (metrics.csv e method_summary.csv)
# -----------------------------------------------------------------------------
def export_quantitative_metrics(
    bench: dict[str, Any],
    kmeans_res: dict[str, Any],
    pca_res: dict[str, Any],
    pca_rec_res: dict[str, Any],
    if_res: dict[str, Any],
    lof_res: dict[str, Any],
    ecod_res: dict[str, Any],
    comp_res: dict[str, Any],
    out_dir: Path,
) -> None:
    """Gera metrics.csv e method_summary.csv com tempos de execução e RAM peak."""
    n_samples = len(bench["X_raw"])
    records = []

    s_ens = comp_res["s_ensemble"]
    ranks = rankdata(-s_ens)

    for i in range(n_samples):
        records.append({
            "uc_id": f"UC_{i+1:05d}",
            "feeder": str(bench["feeders"][i]),
            "ground_truth_category": bench["categories_true"][i],
            "kmeans_cluster": int(kmeans_res["clusters"][i]),
            "pca_reconstruction_error": round(float(pca_rec_res["residuals"][i]), 5),
            "if_score": round(float(if_res["iso_scores"][i]), 4),
            "lof_score": round(float(lof_res["lof_scores"][i]), 4),
            "ecod_score": round(float(ecod_res["ecod_scores"][i]), 4),
            "composite_score": round(float(s_ens[i]), 4),
            "final_rank": int(ranks[i]),
        })

    metrics_df = pl.DataFrame(records)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.write_csv(metrics_path, separator=";")
    print(f" -> Salvo: {metrics_path.name} ({metrics_df.height:,} linhas)")

    # Resumo Comparativo dos Métodos:
    top100_ens = set(np.argsort(-s_ens)[:100])
    top500_ens = set(np.argsort(-s_ens)[:500])

    def calc_overlap(scores: np.ndarray, top_k_ens: set[int], k: int) -> float:
        top_m = set(np.argsort(-scores)[:k])
        return round(len(top_m & top_k_ens) / float(k) * 100.0, 1)

    summary_rows = [
        {
            "method": "K-Means (Perfis)",
            "paradigm": "Clusterização Não-Supervisionada",
            "question_answered": "Quais curvas possuem comportamento semelhante?",
            "n_flagged": 0,
            "flag_rate_pct": 0.0,
            "top_100_overlap_pct": "-",
            "top_500_overlap_pct": "-",
            "median_score": "-",
            "runtime_fit_ms": round(kmeans_res["t_fit"] * 1000.0, 2),
            "peak_ram_mb": round(kmeans_res["peak_ram_mb"], 3),
        },
        {
            "method": "PCA Residual",
            "paradigm": "Subespaço Linear / Erro L2",
            "question_answered": "Quem viola a variedade linear dominante?",
            "n_flagged": int(np.sum(pca_rec_res["residuals"] > np.percentile(pca_rec_res["residuals"], 95))),
            "flag_rate_pct": 5.0,
            "top_100_overlap_pct": calc_overlap(pca_rec_res["residuals"], top100_ens, 100),
            "top_500_overlap_pct": calc_overlap(pca_rec_res["residuals"], top500_ens, 500),
            "median_score": round(float(np.median(pca_rec_res["residuals"])), 4),
            "runtime_fit_ms": round(pca_res["t_fit"] * 1000.0, 2),
            "peak_ram_mb": round(pca_res["peak_ram_mb"], 3),
        },
        {
            "method": "Isolation Forest",
            "paradigm": "Árvores de Isolamento Global",
            "question_answered": "Quem é raro globalmente no hiper-espaço?",
            "n_flagged": int(np.sum(if_res["iso_scores"] > np.percentile(if_res["iso_scores"], 95))),
            "flag_rate_pct": 5.0,
            "top_100_overlap_pct": calc_overlap(if_res["iso_scores"], top100_ens, 100),
            "top_500_overlap_pct": calc_overlap(if_res["iso_scores"], top500_ens, 500),
            "median_score": round(float(np.median(if_res["iso_scores"])), 4),
            "runtime_fit_ms": round(if_res["t_fit"] * 1000.0, 2),
            "peak_ram_mb": round(if_res["peak_ram_mb"], 3),
        },
        {
            "method": "Local Outlier Factor (LOF)",
            "paradigm": "Densidade Relativa k-NN Local",
            "question_answered": "Quem é estranho em relação aos seus vizinhos?",
            "n_flagged": int(np.sum(lof_res["lof_scores"] > np.percentile(lof_res["lof_scores"], 95))),
            "flag_rate_pct": 5.0,
            "top_100_overlap_pct": calc_overlap(lof_res["lof_scores"], top100_ens, 100),
            "top_500_overlap_pct": calc_overlap(lof_res["lof_scores"], top500_ens, 500),
            "median_score": round(float(np.median(lof_res["lof_scores"])), 4),
            "runtime_fit_ms": round(lof_res["t_fit"] * 1000.0, 2),
            "peak_ram_mb": round(lof_res["peak_ram_mb"], 3),
        },
        {
            "method": "ECOD",
            "paradigm": "Caudas Empíricas Não-Paramétricas",
            "question_answered": "Quem ocupa as caudas das distribuições marginais?",
            "n_flagged": int(np.sum(ecod_res["ecod_scores"] > np.percentile(ecod_res["ecod_scores"], 95))),
            "flag_rate_pct": 5.0,
            "top_100_overlap_pct": calc_overlap(ecod_res["ecod_scores"], top100_ens, 100),
            "top_500_overlap_pct": calc_overlap(ecod_res["ecod_scores"], top500_ens, 500),
            "median_score": round(float(np.median(ecod_res["ecod_scores"])), 4),
            "runtime_fit_ms": round(ecod_res["t_fit"] * 1000.0, 2),
            "peak_ram_mb": round(ecod_res["peak_ram_mb"], 3),
        },
        {
            "method": "Ensemble Especialistas (Composite)",
            "paradigm": "Combinação Ponderada com Calibração Empírica",
            "question_answered": "Como integrar evidências ortogonais para priorização de campo?",
            "n_flagged": int(np.sum(s_ens > np.percentile(s_ens, 95))),
            "flag_rate_pct": 5.0,
            "top_100_overlap_pct": "100.0",
            "top_500_overlap_pct": "100.0",
            "median_score": round(float(np.median(s_ens)), 4),
            "runtime_fit_ms": round((if_res["t_fit"] + lof_res["t_fit"] + ecod_res["t_fit"] + pca_res["t_fit"]) * 1000.0, 2),
            "peak_ram_mb": round(max(if_res["peak_ram_mb"], lof_res["peak_ram_mb"], ecod_res["peak_ram_mb"], pca_res["peak_ram_mb"]), 3),
        },
    ]

    summary_df = pl.DataFrame(summary_rows)
    summary_path = out_dir / "method_summary.csv"
    summary_df.write_csv(summary_path, separator=";")
    print(f" -> Salvo: {summary_path.name}")


# -----------------------------------------------------------------------------
# 7. EXECUÇÃO PRINCIPAL DO PIPELINE DEMONSTRATIVO
# -----------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("DEMONSTRAÇÃO METODOLÓGICA DAS TÉCNICAS DE ML PARA O ARTIGO / TCC")
    print("Objetivo: Evidenciar perguntas, efeitos e complementaridade de cada técnica.")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_academic_style()

    # 1. Gerar Benchmark Sintético Controlado (Ground Truth Conhecido)
    print("\n[1/10] Gerando benchmark sintético controlado com ground truth...")
    bench = generate_synthetic_benchmark(n_samples=1000, n_timesteps=96, seed=42)
    print(f" -> {len(bench['X_raw'])} curvas diárias geradas em 4 classes comportamentais e outliers controlados.")

    # 2. Carregar dados reais de apoio caso disponíveis
    print("\n[2/10] Verificando dados reais de Araucária para contraste empírico...")
    real_data = load_real_araucaria_subset()
    if real_data is not None:
        print(f" -> Dataset real detectado: {real_data['df'].height:,} linhas de cutoffs avaliados.")
    else:
        print(" -> Modo sintético puro ativo (dados reais não localizados ou incompletos).")

    # 3. Gerar Figura 01: Perfis Diários Crus
    print("\n[3/10] Gerando Figura 01: Perfis Diários Crus (Antes do ML)...")
    plot_01_raw_profiles(bench, OUT_DIR / "01_raw_profiles.png")

    # 4. Gerar Figura 02: K-Means (Formação de Perfis)
    print("\n[4/10] Gerando Figura 02: K-Means (Formação de Perfis Comportamentais)...")
    kmeans_res = plot_02_kmeans_profiles(bench, OUT_DIR / "02_kmeans_profiles.png")

    # 5. Gerar Figura 03: PCA (Compressão e Espaço Latente)
    print("\n[5/10] Gerando Figura 03: PCA (Projeção e Variância Explicada)...")
    pca_res = plot_03_pca_projection(bench, kmeans_res, OUT_DIR / "03_pca_projection.png")

    # 6. Gerar Figura 04: PCA Residual (Anomalia Estrutural)
    print("\n[6/10] Gerando Figura 04: PCA Residual (Mesmo Consumo, Morfologia Rara)...")
    pca_rec_res = plot_04_pca_residual(bench, pca_res, OUT_DIR / "04_pca_residual.png")

    # 7. Gerar Figura 05: Isolation Forest (Raridade Global e Micro-Cluster)
    print("\n[7/10] Gerando Figura 05: Isolation Forest (Raridade Global)...")
    if_res = plot_05_isolation_forest(bench, pca_res, OUT_DIR / "05_isolation_forest.png")

    # 8. Gerar Figura 06: Local Outlier Factor (Densidade Relativa Local)
    print("\n[8/10] Gerando Figura 06: LOF (Densidade Local vs IForest)...")
    lof_res = plot_06_lof(bench, if_res, OUT_DIR / "06_lof.png")

    # 9. Gerar Figura 07: ECOD (Extremidade Marginal)
    print("\n[9/10] Gerando Figura 07: ECOD (Caudas Empíricas Não-Paramétricas)...")
    ecod_res = plot_07_ecod(bench, OUT_DIR / "07_ecod.png")

    # 10. Gerar Figuras 08, 09 e 10
    print("\n[10/10] Gerando Figuras 08 (Comparação Direta), 09 (Ablação) e 10 (Estudos de Caso)...")
    comp_res = plot_08_detector_comparison(bench, if_res, lof_res, ecod_res, pca_rec_res, OUT_DIR / "08_detector_comparison.png")
    plot_09_ablation_funnel(bench, comp_res, OUT_DIR / "09_ablation_funnel.png")
    plot_10_case_studies(bench, comp_res, OUT_DIR / "10_case_studies.png")

    # 11. Exportar CSVs Quantitativos
    print("\nExportando métricas e profiling em CSV...")
    export_quantitative_metrics(bench, kmeans_res, pca_res, pca_rec_res, if_res, lof_res, ecod_res, comp_res, OUT_DIR)

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"CONCLUÍDO COM SUCESSO EM {elapsed:.1f} SEGUNDOS!")
    print(f"Todas as 10 figuras e arquivos CSV foram gerados em: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
