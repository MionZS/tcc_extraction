"""Pure, testable EDA primitives for the AMI dataset (Araucária / Copel).

Everything here is dataset-agnostic Polars + matplotlib code:

* **Completeness** of the mass memory (memória de massa) per meter, 0-100 %.
* **Consumption outliers** per consumption band (meters with similar average
  consumption are compared against each other) and per meter (own history).
* **Communication delay** heatmap / CDF built from the ORCA delay histograms
  produced by the ``delay_analysis`` project.

All thresholds are explicit parameters so the notebook can expose them as
controls and the article can quote the exact values used.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

# ── Discovery / loading ─────────────────────────────────────────────────────

DAILY_FEATURE_GLOBS = (
    "output/features/uc_day_features/**/data.parquet",
    "data/features/uc_day_features/**/data.parquet",
)


def discover_daily_features(project_root: Path | str, globs: Sequence[str] = DAILY_FEATURE_GLOBS) -> list[Path]:
    """Return every daily-feature parquet partition found under ``project_root``."""
    root = Path(project_root)
    found: list[Path] = []
    for pattern in globs:
        found.extend(sorted(root.glob(pattern)))
    return found


def load_daily_features(files: Iterable[Path | str]) -> pl.DataFrame:
    """Concatenate daily-feature partitions into one frame (diagonal, tolerant to schema drift)."""
    frames = []
    for f in files:
        try:
            frames.append(pl.read_parquet(f))
        except Exception:
            continue
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# ── 1. Mass-memory completeness ─────────────────────────────────────────────


def meter_completeness(daily: pl.DataFrame, *, coverage_col: str = "INTERVAL_COVERAGE", key: str = "UC") -> pl.DataFrame:
    """Completeness of the mass memory per meter/UC, in [0, 1].

    ``completeness_calendar`` charges missing days as zero: the denominator is the
    number of calendar days between the meter's first observation and the last day
    of the dataset (so newly installed meters are not penalised for the past).
    ``completeness_present`` only averages days that have a row.
    """
    if daily.is_empty() or coverage_col not in daily.columns:
        return pl.DataFrame(
            schema={key: pl.String, "n_days_present": pl.Int64, "n_days_calendar": pl.Int64,
                    "completeness_present": pl.Float64, "completeness_calendar": pl.Float64,
                    "full_days_share": pl.Float64}
        )

    last_day = daily["REPORT_DAY"].max()
    per_day = (
        daily.group_by([key, "REPORT_DAY"])
        .agg(pl.col(coverage_col).max().clip(0.0, 1.0).alias("cov"))
    )
    return (
        per_day.group_by(key)
        .agg(
            pl.len().alias("n_days_present"),
            pl.col("cov").sum().alias("_cov_sum"),
            pl.col("cov").mean().alias("completeness_present"),
            (pl.col("cov") >= 0.95).mean().alias("full_days_share"),
            pl.col("REPORT_DAY").min().alias("_first_day"),
        )
        .with_columns(
            ((pl.lit(last_day) - pl.col("_first_day")).dt.total_days() + 1).alias("n_days_calendar")
        )
        .with_columns((pl.col("_cov_sum") / pl.col("n_days_calendar")).clip(0.0, 1.0).alias("completeness_calendar"))
        .drop(["_cov_sum", "_first_day"])
        .sort(key)
    )


def completeness_histogram(comp: pl.DataFrame, *, column: str = "completeness_calendar", step_pct: int = 10) -> pl.DataFrame:
    """Number of meters per completeness bin (0-100 %). Last bin is closed at 100 %."""
    edges = list(range(0, 100 + step_pct, step_pct))
    pct = (comp[column] * 100.0).to_numpy() if not comp.is_empty() else np.array([])
    counts, _ = np.histogram(pct, bins=edges)
    labels = [f"{a}-{b}%" for a, b in zip(edges[:-1], edges[1:])]
    total = int(counts.sum())
    return pl.DataFrame(
        {
            "bin": labels,
            "bin_start": edges[:-1],
            "meters": counts.astype(int).tolist(),
            "share": [(c / total if total else 0.0) for c in counts],
        }
    ).with_columns(pl.col("meters").cum_sum().alias("cumulative_meters"))


# ── 2. Consumption outliers ─────────────────────────────────────────────────


def consumption_bands(
    daily: pl.DataFrame,
    *,
    n_bands: int = 5,
    value_col: str = "FA_INTERVAL_SUM",
    coverage_col: str = "INTERVAL_COVERAGE",
    min_coverage: float = 0.8,
    key: str = "UC",
) -> pl.DataFrame:
    """Assign each meter to a consumption band using quantiles of its median daily energy.

    Only reliable days (coverage >= ``min_coverage``) enter the median so partial
    days do not drag a meter into a lower band.
    """
    reliable = daily.filter(pl.col(coverage_col) >= min_coverage) if coverage_col in daily.columns else daily
    med = reliable.group_by(key).agg(
        pl.col(value_col).median().alias("median_daily_kwh"),
        pl.len().alias("n_reliable_days"),
    )
    if med.is_empty():
        return med.with_columns(pl.lit(None, dtype=pl.String).alias("band"))

    qs = np.linspace(0, 1, n_bands + 1)
    edges = np.quantile(med["median_daily_kwh"].to_numpy(), qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([edges[0] - 1e-9, edges[0] + 1e-9])
    idx = np.clip(np.searchsorted(edges, med["median_daily_kwh"].to_numpy(), side="right") - 1, 0, len(edges) - 2)
    labels = [f"B{i + 1}: {edges[i]:.1f}-{edges[i + 1]:.1f} kWh/dia" for i in range(len(edges) - 1)]
    return med.with_columns(
        pl.Series("band", [labels[i] for i in idx]),
        pl.Series("band_idx", idx.astype(int)),
    )


def consumption_outliers(
    daily: pl.DataFrame,
    bands: pl.DataFrame,
    *,
    value_col: str = "FA_INTERVAL_SUM",
    coverage_col: str = "INTERVAL_COVERAGE",
    min_coverage: float = 0.8,
    z_threshold: float = 3.5,
    tukey_k: float = 1.5,
    key: str = "UC",
) -> pl.DataFrame:
    """Flag daily consumption values that fall outside the expected pattern.

    Two complementary definitions (both on reliable days only):

    * ``out_band``  - value outside Tukey fences of the *band peers* (log1p scale):
      compares a meter-day against all meter-days of meters with similar mean consumption.
    * ``out_self``  - modified z-score vs the meter's *own* history
      (median/MAD, with a scale floor so near-constant meters do not explode).
    """
    if daily.is_empty():
        return pl.DataFrame()

    d = (
        daily.filter(pl.col(coverage_col) >= min_coverage)
        .select([key, "REPORT_DAY", value_col])
        .drop_nulls()
        .join(bands.select([key, "band", "band_idx", "median_daily_kwh"]), on=key, how="inner")
        .with_columns(pl.col(value_col).clip(lower_bound=0.0).log1p().alias("_lv"))
    )
    if d.is_empty():
        return d

    fences = d.group_by("band").agg(
        pl.col("_lv").quantile(0.25).alias("_q1"),
        pl.col("_lv").quantile(0.75).alias("_q3"),
    ).with_columns(
        (pl.col("_q3") - pl.col("_q1")).alias("_iqr")
    ).with_columns(
        (pl.col("_q1") - tukey_k * pl.col("_iqr")).alias("band_lo"),
        (pl.col("_q3") + tukey_k * pl.col("_iqr")).alias("band_hi"),
    ).select(["band", "band_lo", "band_hi"])

    d = d.join(fences, on="band", how="left").with_columns(
        pl.col(value_col).median().over(key).alias("_med"),
    ).with_columns(
        (pl.col(value_col) - pl.col("_med")).abs().median().over(key).alias("_mad"),
    ).with_columns(
        pl.max_horizontal(pl.col("_mad") * 1.4826, pl.col("_med").abs() * 0.10, pl.lit(0.05)).alias("_scale"),
    ).with_columns(
        ((pl.col(value_col) - pl.col("_med")) / pl.col("_scale")).alias("z_self"),
        ((pl.col("_lv") < pl.col("band_lo")) | (pl.col("_lv") > pl.col("band_hi"))).alias("out_band"),
    ).with_columns(
        (pl.col("z_self").abs() > z_threshold).alias("out_self"),
        (pl.col("_lv") > pl.col("band_hi")).alias("out_band_high"),
        (pl.col("_lv") < pl.col("band_lo")).alias("out_band_low"),
    )
    return d.drop(["_lv", "_med", "_mad", "_scale"])


def outlier_summary_by_band(outliers: pl.DataFrame, *, key: str = "UC") -> pl.DataFrame:
    """Per band: meters, measurements, share of outlier measurements, meters with >=1 outlier."""
    if outliers.is_empty():
        return pl.DataFrame()
    per_meter = outliers.group_by([key, "band", "band_idx"]).agg(
        pl.len().alias("n_days"),
        pl.col("out_band").sum().alias("n_out_band"),
        pl.col("out_self").sum().alias("n_out_self"),
    )
    return (
        per_meter.group_by(["band", "band_idx"])
        .agg(
            pl.len().alias("meters"),
            pl.col("n_days").sum().alias("measurements"),
            pl.col("n_out_band").sum().alias("out_band_measurements"),
            pl.col("n_out_self").sum().alias("out_self_measurements"),
            (pl.col("n_out_band") > 0).sum().alias("meters_with_out_band"),
            (pl.col("n_out_self") > 0).sum().alias("meters_with_out_self"),
        )
        .with_columns(
            (pl.col("out_band_measurements") / pl.col("measurements")).alias("out_band_rate"),
            (pl.col("out_self_measurements") / pl.col("measurements")).alias("out_self_rate"),
            (pl.col("meters_with_out_band") / pl.col("meters")).alias("meters_with_out_band_share"),
            (pl.col("meters_with_out_self") / pl.col("meters")).alias("meters_with_out_self_share"),
        )
        .sort("band_idx")
    )


def outlier_meter_distribution(outliers: pl.DataFrame, *, key: str = "UC") -> pl.DataFrame:
    """Per meter outlier rates (for histograms of 'how many outliers does a meter have')."""
    if outliers.is_empty():
        return pl.DataFrame()
    return outliers.group_by([key, "band"]).agg(
        pl.len().alias("n_days"),
        pl.col("out_band").mean().alias("out_band_rate"),
        pl.col("out_self").mean().alias("out_self_rate"),
    )


# ── 3. Communication delay (ORCA histograms from delay_analysis) ────────────

REGULAR_DELAY_CATEGORIES = ("0-60m", "60-240m", "240-1440m", "1440-10080m")


def discover_delay_histograms(delay_dir: Path | str, source: str) -> list[Path]:
    """Real (non-simulated) daily histogram parquets: ``<delay_dir>/<SOURCE>/<SOURCE>_<YYYY-MM-DD>.parquet``."""
    src_dir = Path(delay_dir) / source
    if not src_dir.exists():
        return []
    return sorted(
        f for f in src_dir.glob(f"{source}_*.parquet")
        if not f.name.startswith("SIM_") and "consolidado" not in f.stem.lower()
    )


def load_delay_histograms(files: Iterable[Path | str]) -> pl.DataFrame:
    """Load and concatenate daily delay histograms."""
    frames = []
    for f in files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        if "dt_referencia" in df.columns:
            df = df.with_columns(pl.col("dt_referencia").cast(pl.String).str.slice(0, 10))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def delay_arrival_table(hist: pl.DataFrame, *, include_specials: bool = False) -> pl.DataFrame:
    """Readings per delay bin plus cumulative share ("how many arrive within X minutes")."""
    if hist.is_empty():
        return pl.DataFrame()
    base = hist if include_specials else hist.filter(pl.col("bin_category").is_in(REGULAR_DELAY_CATEGORIES))
    out = (
        base.group_by(["bin_label", "bin_start", "bin_end"])
        .agg(pl.col("contagem").sum().alias("readings"))
        .sort("bin_start")
    )
    total = out["readings"].sum()
    return out.with_columns(
        (pl.col("readings") / total).alias("share"),
        (pl.col("readings").cum_sum() / total).alias("cumulative_share"),
    )


def delay_heatmap_matrix(hist: pl.DataFrame, *, agg_minutes: int = 60, log10: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
    """Delay-bin × time-of-day density matrix (same construction as the delay_analysis notebook)."""
    if hist.is_empty():
        return np.zeros((0, 0)), [], []
    base = hist.filter(pl.col("bin_category").is_in(REGULAR_DELAY_CATEGORIES))
    base = base.with_columns(
        (((pl.col("window_idx") * 5) // agg_minutes) * agg_minutes).alias("_start")
    ).with_columns(
        pl.format("{}:{}", (pl.col("_start") // 60).cast(pl.String).str.zfill(2), (pl.col("_start") % 60).cast(pl.String).str.zfill(2)).alias("_win")
    )
    grid = base.group_by(["_win", "bin_label", "bin_start", "bin_width"]).agg(pl.col("contagem").sum().alias("cnt"))
    grid = grid.with_columns((pl.col("cnt") / pl.col("bin_width")).alias("density"))
    bins = grid.select(["bin_label", "bin_start"]).unique().sort("bin_start")["bin_label"].to_list()
    wins = sorted(grid["_win"].unique().to_list())
    bi = {b: i for i, b in enumerate(bins)}
    wi = {w: i for i, w in enumerate(wins)}
    z = np.zeros((len(bins), len(wins)))
    for b, w, v in grid.select(["bin_label", "_win", "density"]).iter_rows():
        z[bi[b], wi[w]] += v
    if log10:
        z = np.log10(np.maximum(z, 1e-4))
    return z, wins, bins


def daily_delay_summary(hist: pl.DataFrame) -> pl.DataFrame:
    """Per day: readings, weighted mean delay, and share within 1 h / 4 h / 24 h."""
    if hist.is_empty():
        return pl.DataFrame()
    reg = hist.filter(pl.col("bin_category").is_in(REGULAR_DELAY_CATEGORIES))
    return (
        reg.group_by("dt_referencia")
        .agg(
            pl.col("contagem").sum().alias("readings"),
            (pl.col("soma_delay").sum() / pl.col("contagem").sum()).alias("mean_delay_min"),
            pl.col("contagem").filter(pl.col("bin_category") == "0-60m").sum().alias("_le1h"),
            pl.col("contagem").filter(pl.col("bin_category").is_in(["0-60m", "60-240m"])).sum().alias("_le4h"),
            pl.col("contagem").filter(pl.col("bin_category").is_in(["0-60m", "60-240m", "240-1440m"])).sum().alias("_le24h"),
        )
        .with_columns(
            (pl.col("_le1h") / pl.col("readings")).alias("share_le_1h"),
            (pl.col("_le4h") / pl.col("readings")).alias("share_le_4h"),
            (pl.col("_le24h") / pl.col("readings")).alias("share_le_24h"),
        )
        .drop(["_le1h", "_le4h", "_le24h"])
        .sort("dt_referencia")
    )


# ── Plot helpers (matplotlib; no extra dependency) ──────────────────────────


def _plt():
    import matplotlib

    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    return plt


def plot_completeness(hist: pl.DataFrame, *, title: str = "Completude da memória de massa por medidor"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(hist["bin"].to_list(), hist["meters"].to_list(), color="#2563eb", edgecolor="white")
    total = max(1, int(hist["meters"].sum()))
    for rect, n in zip(bars, hist["meters"].to_list()):
        if n:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{n:,}\n{n / total:.1%}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Completude (0–100 %)")
    ax.set_ylabel("Nº de medidores")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_outlier_bands(summary: pl.DataFrame, *, title: str = "Medições de consumo fora do padrão por faixa de consumo médio"):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [b.split(":")[0] for b in summary["band"].to_list()]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, summary["out_band_rate"].to_numpy() * 100, 0.4, label="vs pares da faixa (Tukey)", color="#dc2626")
    axes[0].bar(x + 0.2, summary["out_self_rate"].to_numpy() * 100, 0.4, label="vs histórico próprio (MAD)", color="#f59e0b")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("% das medições diárias")
    axes[0].set_title("Taxa de medições outlier")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x - 0.2, summary["meters_with_out_band_share"].to_numpy() * 100, 0.4, color="#dc2626")
    axes[1].bar(x + 0.2, summary["meters_with_out_self_share"].to_numpy() * 100, 0.4, color="#f59e0b")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("% dos medidores da faixa")
    axes[1].set_title("Medidores com ≥ 1 medição outlier")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_delay_heatmap(z: np.ndarray, wins: list[str], bins: list[str], *, title: str, log10: bool = True):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="turbo")
    step = max(1, len(wins) // 24)
    ax.set_xticks(range(0, len(wins), step), [wins[i] for i in range(0, len(wins), step)], rotation=60, fontsize=7)
    ystep = max(1, len(bins) // 25)
    ax.set_yticks(range(0, len(bins), ystep), [bins[i] for i in range(0, len(bins), ystep)], fontsize=7)
    ax.set_xlabel("Janela de medição (hora do dia)")
    ax.set_ylabel("Faixa de atraso até a chegada (min)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log10(densidade)" if log10 else "densidade (leituras/min)")
    fig.tight_layout()
    return fig


def plot_delay_arrival(table: pl.DataFrame, *, title: str = "Chegada acumulada das leituras"):
    plt = _plt()
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = np.arange(table.height)
    ax1.bar(x, table["share"].to_numpy() * 100, color="#2563eb")
    ax1.set_ylabel("% das leituras na faixa")
    ax1.set_xticks(x, table["bin_label"].to_list(), rotation=70, fontsize=7)
    ax2 = ax1.twinx()
    ax2.plot(x, table["cumulative_share"].to_numpy() * 100, color="#dc2626", marker="o", ms=3)
    ax2.set_ylabel("% acumulado")
    ax2.set_ylim(0, 101)
    ax1.set_title(title)
    fig.tight_layout()
    return fig
