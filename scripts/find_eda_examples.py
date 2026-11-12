"""
scripts/find_eda_examples.py
-------------------------------------------------------------------------------
Localiza exemplos empíricos REAIS no dataset classificado + Data Lake e gera
gráficos LGPD-compliant (com censura de UC via src.utils.censorship):
  1. docs/figures/eda/exemplos/comparacao_duas_ucs_mesmo_consumo.png
     (Duas UCs de consumo mediano semelhante, porém com MAD e perfil real distintos)
  2. docs/figures/eda/exemplos/acompanhamento_semanal_outlier.png
     (Série temporal REAL de 7 dias / 168h em ami_interval, ou diária em
      uc_day_features caso os brutos horários não estejam presentes, destacando
      baseline mediana, banda MAD ±3.5σ, degrau descendente CUSUM, picos e gaps
      reais de cobertura telecom — SEM dados sintéticos)
  3. docs/figures/eda/exemplos/examples_metadata.json (somente IDs censurados)
  4. output/reports/private_examples_lookup.json (de-para interno para o pesquisador)
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda.uc_timeseries import (
    load_uc_24h_profile,
    load_uc_daily_timeseries,
    load_uc_hourly_timeseries,
    resolve_col,
)
from src.utils.censorship import censor_uc_id

OUT_DIR = PROJECT_ROOT / "docs" / "figures" / "eda" / "exemplos"
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"
PARQUET_PATH = REPORTS_DIR / "classified_ucs_latest.parquet"
CSV_PATH = REPORTS_DIR / "classified_ucs_latest.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "ucs_interesse.yaml"


def _apply_dark_academic_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cbd5e1",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.85,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
    })


def _load_interest_config() -> dict[str, Any]:
    """Read optional config/ucs_interesse.yaml without failing if PyYAML is absent."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_classified_df() -> pl.DataFrame:
    if PARQUET_PATH.exists():
        print(f"Lendo dataset Parquet: {PARQUET_PATH}")
        return pl.read_parquet(str(PARQUET_PATH))
    if CSV_PATH.exists():
        print(f"Lendo dataset CSV (;): {CSV_PATH}")
        return pl.read_csv(str(CSV_PATH), separator=";", infer_schema_length=5000)
    raise FileNotFoundError(
        f"Nenhum arquivo classificado encontrado em {PARQUET_PATH} ou {CSV_PATH}."
    )


def _normalize_schema(df: pl.DataFrame) -> dict[str, str | None]:
    return {
        "uc": resolve_col(df, ["id__uc_id", "UC", "uc_id"]),
        "cutoff": resolve_col(df, ["meta__cutoff_date", "CUTOFF_DATE"]),
        "cov": resolve_col(df, ["x__avg_interval_coverage", "AVG_INTERVAL_COVERAGE", "meta__reliable_day_ratio", "score_comunicacao"]),
        "median": resolve_col(df, ["x__hist_fa_sum_median", "HIST_FA_SUM_MEDIAN", "fa_sum_median"]),
        "mad": resolve_col(df, ["x__hist_fa_sum_mad", "HIST_FA_SUM_MAD", "fa_sum_mad"]),
        "slope": resolve_col(df, ["x__trend_fa_slope", "TREND_FA_SLOPE"]),
        "cv": resolve_col(df, ["x__fa_cv_window", "FA_CV_WINDOW"]),
        "lf": resolve_col(df, ["x__hist_load_factor_mean", "HIST_LOAD_FACTOR_MEAN"]),
        "ci": resolve_col(df, ["x__hist_current_imbalance_max", "HIST_CURRENT_IMBALANCE_MAX"]),
        "score": resolve_col(df, ["ANOMALY_SCORE", "anomaly_score"]),
        "is_anom": resolve_col(df, ["IS_ANOMALY", "is_anomaly"]),
        "cat": resolve_col(df, ["CATEGORIA_DIAGNOSTICA", "categoria_diagnostica"]),
        "sub": resolve_col(df, ["SUBTIPO_DIAGNOSTICO", "subtipo_diagnostico"]),
        "feeder": resolve_col(df, ["meta__feeder", "ALIMENTADOR", "FEEDER"]),
        "classe": resolve_col(df, ["x__classe", "CLASSE"]),
    }


def _row_summary(row: dict[str, Any], cols: dict[str, str | None]) -> dict[str, Any]:
    uc_raw = str(row[cols["uc"]]) if cols["uc"] else "N/A"
    return {
        "uc_raw": uc_raw,
        "uc_censored": censor_uc_id(uc_raw),
        "cutoff": str(row.get(cols["cutoff"], ""))[:10] if cols["cutoff"] else "",
        "coverage": round(float(row.get(cols["cov"]) or 0.0), 4) if cols["cov"] else None,
        "median_kwh_day": round(float(row.get(cols["median"]) or 0.0), 3) if cols["median"] else 0.0,
        "mad_kwh_day": round(float(row.get(cols["mad"]) or 0.0), 3) if cols["mad"] else 0.0,
        "slope": round(float(row.get(cols["slope"]) or 0.0), 4) if cols["slope"] else None,
        "cv_window": round(float(row.get(cols["cv"]) or 0.0), 4) if cols["cv"] else None,
        "load_factor": round(float(row.get(cols["lf"]) or 0.0), 3) if cols["lf"] else None,
        "current_imbalance": round(float(row.get(cols["ci"]) or 0.0), 3) if cols["ci"] else None,
        "anomaly_score": round(float(row.get(cols["score"]) or 0.0), 4) if cols["score"] else None,
        "category": str(row.get(cols["cat"], "")) if cols["cat"] else "",
        "subtype": str(row.get(cols["sub"], "")) if cols["sub"] else "",
        "feeder": str(row.get(cols["feeder"], "")) if cols["feeder"] else "",
        "classe": str(row.get(cols["classe"], "")) if cols["classe"] else "",
    }


def find_contrasting_uc_pair(
    df: pl.DataFrame,
    cols: dict[str, str | None],
    manual_ucs: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Find two clean UCs with nearly equal daily median consumption and very different MAD."""
    uc_col, med_col, mad_col = cols["uc"], cols["median"], cols["mad"]
    cov_col = cols["cov"]
    if not uc_col or not med_col or not mad_col:
        return None

    # Deduplicate to latest cutoff per UC if multiple cutoffs exist
    work = df
    if cols["cutoff"]:
        work = work.sort(cols["cutoff"]).unique(subset=[uc_col], keep="last")

    # Honor manual UC override from config/ucs_interesse.yaml if valid
    if manual_ucs and len(manual_ucs) >= 2:
        sub = work.filter(pl.col(uc_col).cast(pl.String).is_in([str(u) for u in manual_ucs[:2]]))
        if sub.height >= 2:
            rows = sub.to_dicts()
            return _row_summary(rows[0], cols), _row_summary(rows[1], cols)

    cond = (
        (pl.col(med_col).cast(pl.Float64, strict=False) >= 8.0)
        & (pl.col(med_col).cast(pl.Float64, strict=False) <= 35.0)
        & (pl.col(mad_col).cast(pl.Float64, strict=False) > 0.05)
    )
    if cov_col:
        cond = cond & (pl.col(cov_col).cast(pl.Float64, strict=False) >= 0.70)

    df_valid = work.filter(cond)
    if df_valid.height < 20:
        df_valid = work.filter(
            (pl.col(med_col).cast(pl.Float64, strict=False) >= 3.0)
            & (pl.col(mad_col).cast(pl.Float64, strict=False) > 0.01)
        )
    if df_valid.height < 2:
        return None

    # Focus on typical residential/commercial band 10-25 kWh/day when populated
    preferred = df_valid.filter(
        (pl.col(med_col).cast(pl.Float64, strict=False) >= 10.0)
        & (pl.col(med_col).cast(pl.Float64, strict=False) <= 25.0)
    )
    pool = preferred if preferred.height >= 40 else df_valid

    stable_pool = pool.sort(mad_col).head(80).to_dicts()
    volatile_pool = pool.sort(mad_col, descending=True).head(80).to_dicts()

    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_score = -1.0

    for a in stable_pool[:30]:
        med_a = float(a[med_col])
        mad_a = max(float(a[mad_col]), 0.05)
        for b in volatile_pool[:30]:
            if str(a[uc_col]) == str(b[uc_col]):
                continue
            med_b = float(b[med_col])
            rel_diff = abs(med_a - med_b) / max(med_a, 1e-6)
            if rel_diff > 0.08:
                continue
            mad_b = float(b[mad_col])
            ratio = mad_b / mad_a
            # Small bonus if both actually have real 24h interval partitions
            if ratio > best_score:
                best_score = ratio
                best_pair = (_row_summary(a, cols), _row_summary(b, cols))

    if best_pair is not None:
        return best_pair

    # Fallback: closest medians among top and bottom quartiles
    a = stable_pool[0]
    b = min(volatile_pool, key=lambda r: abs(float(r[med_col]) - float(a[med_col])))
    return _row_summary(a, cols), _row_summary(b, cols)


def find_outlier_with_drop_and_spike(
    df: pl.DataFrame,
    cols: dict[str, str | None],
    manual_uc: str | None = None,
) -> dict[str, Any] | None:
    """Find a real anomaly UC exhibiting both sustained drop and strong dispersion/spikes."""
    uc_col = cols["uc"]
    if not uc_col:
        return None

    work = df
    if cols["cutoff"]:
        work = work.sort(cols["cutoff"]).unique(subset=[uc_col], keep="last")

    if manual_uc:
        sub = work.filter(pl.col(uc_col).cast(pl.String) == str(manual_uc))
        if sub.height > 0:
            return _row_summary(sub.to_dicts()[0], cols)

    if cols["is_anom"]:
        try:
            anoms = work.filter(
                pl.col(cols["is_anom"]).cast(pl.String).str.to_lowercase().is_in(["true", "1"])
            )
        except Exception:
            anoms = work.filter(pl.col(cols["is_anom"]) == True)
        if anoms.height == 0:
            anoms = work
    else:
        anoms = work

    # Exclude low-coverage telecom quarantine so the anomaly reflects real consumption behavior
    if cols["cov"]:
        try:
            clean_anoms = anoms.filter(pl.col(cols["cov"]).cast(pl.Float64) >= 0.70)
            if clean_anoms.height > 0:
                anoms = clean_anoms
        except Exception:
            pass

    # Prioritize drop/multivariate/bypass categories
    target_cats = {
        "DEGRAU_DESCENDENTE_CONSUMO",
        "QUEDA_TENDENCIA_CONSUMO_SLOPE_NEGATIVO",
        "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT",
        "DESEQUILIBRIO_CARGA_OU_BYPASS",
        "ANOMALIA_MULTIVARIADA",
    }
    if cols["cat"]:
        cat_match = anoms.filter(pl.col(cols["cat"]).cast(pl.String).is_in(list(target_cats)))
        if cat_match.height > 0:
            anoms = cat_match

    # Prefer negative slope AND high MAD/CV so both drop and spike signals exist
    candidates: pl.DataFrame = anoms
    if cols["slope"] and cols["mad"]:
        both = anoms.filter((pl.col(cols["slope"]) < -0.05) & (pl.col(cols["mad"]) > 0.3))
        if both.height > 0:
            candidates = both

    sort_col = cols["score"] or cols["mad"] or uc_col
    top_rows = candidates.sort(sort_col, descending=True).head(25).to_dicts()

    # Check Data Lake partitions to prefer a candidate that actually has real telemetry on disk
    for row in top_rows:
        uc_id = str(row[uc_col])
        ts = load_uc_daily_timeseries(uc_id, base_dir=PROJECT_ROOT / "output", days=35)
        if ts.get("status") == "ok" and len(ts.get("points", [])) >= 7:
            return _row_summary(row, cols)

    if top_rows:
        return _row_summary(top_rows[0], cols)
    return None


def find_top_outlier_candidates(
    df: pl.DataFrame,
    cols: dict[str, str | None],
    top_n: int = 30,
) -> list[dict[str, Any]]:
    """Return ranked list of top candidate UCs across diagnostic anomaly categories."""
    uc_col = cols["uc"]
    if not uc_col:
        return []
    work = df
    if cols["cutoff"]:
        work = work.sort(cols["cutoff"]).unique(subset=[uc_col], keep="last")
    if cols["is_anom"]:
        try:
            anoms = work.filter(
                pl.col(cols["is_anom"]).cast(pl.String).str.to_lowercase().is_in(["true", "1"])
            )
        except Exception:
            anoms = work.filter(pl.col(cols["is_anom"]) == True)
        if anoms.height == 0:
            anoms = work
    else:
        anoms = work

    if cols["cov"]:
        try:
            clean = anoms.filter(pl.col(cols["cov"]).cast(pl.Float64) >= 0.70)
            if clean.height > 0:
                anoms = clean
        except Exception:
            pass

    sort_col = cols["score"] or cols["mad"] or uc_col
    top_rows = anoms.sort(sort_col, descending=True).head(top_n).to_dicts()
    return [_row_summary(r, cols) for r in top_rows]


def plot_comparison_pair(
    uc_a: dict[str, Any],
    uc_b: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Plot real statistical metrics + real 24h (or 30d) comparison between two UCs."""
    _apply_dark_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)

    label_a = f"UC {uc_a['uc_censored']}\n(Estável)"
    label_b = f"UC {uc_b['uc_censored']}\n(Irregular / Alta Dispersão)"

    # Left panel: Real Window Statistics
    metrics_names = ["Mediana Diária\n(kWh/dia)", "MAD\n(kWh/dia)"]
    vals_a = [uc_a["median_kwh_day"], uc_a["mad_kwh_day"]]
    vals_b = [uc_b["median_kwh_day"], uc_b["mad_kwh_day"]]

    if uc_a.get("cv_window") is not None and uc_b.get("cv_window") is not None:
        metrics_names.append("CV Janela\n(×10)")
        vals_a.append(uc_a["cv_window"] * 10.0)
        vals_b.append(uc_b["cv_window"] * 10.0)

    x = np.arange(len(metrics_names))
    w = 0.34
    bars1 = ax1.bar(x - w / 2, vals_a, width=w, label=f"UC {uc_a['uc_censored']}", color="#2563eb", alpha=0.88)
    bars2 = ax1.bar(x + w / 2, vals_b, width=w, label=f"UC {uc_b['uc_censored']}", color="#ea580c", alpha=0.88)

    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax1.annotate(
            f"{h:.2f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.set_title("Mesma Mediana de Faturamento, Dispersão (MAD) Oposta")
    ax1.set_ylabel("Magnitude")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Right panel: Real 24h profile from ami_interval, or fallback to real daily series
    prof_a = load_uc_24h_profile(uc_a["uc_raw"], base_dir=PROJECT_ROOT / "output")
    prof_b = load_uc_24h_profile(uc_b["uc_raw"], base_dir=PROJECT_ROOT / "output")
    curve_mode = "24h_ami_interval"

    if prof_a["status"] == "ok" and prof_b["status"] == "ok":
        hours = [p["hour"] for p in prof_a["profile_24h"]]
        med_a = [p["median"] for p in prof_a["profile_24h"]]
        p10_a = [p["p10"] for p in prof_a["profile_24h"]]
        p90_a = [p["p90"] for p in prof_a["profile_24h"]]

        med_b = [p["median"] for p in prof_b["profile_24h"]]
        p10_b = [p["p10"] for p in prof_b["profile_24h"]]
        p90_b = [p["p90"] for p in prof_b["profile_24h"]]

        ax2.fill_between(hours, p10_a, p90_a, color="#2563eb", alpha=0.14, label=f"{uc_a['uc_censored']} (P10–P90)")
        ax2.plot(hours, med_a, color="#2563eb", lw=2.3, marker="o", ms=3.5, label=f"{uc_a['uc_censored']} Mediana Horária")

        ax2.fill_between(hours, p10_b, p90_b, color="#ea580c", alpha=0.14, label=f"{uc_b['uc_censored']} (P10–P90)")
        ax2.plot(hours, med_b, color="#ea580c", lw=2.3, marker="s", ms=3.5, label=f"{uc_b['uc_censored']} Mediana Horária")

        ax2.set_xticks(range(0, 24, 3))
        ax2.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)])
        ax2.set_xlabel("Hora do Dia (Dados Reais — Memória de Massa AMI)")
        ax2.set_ylabel("Consumo Ativo Horário (kWh/h)")
        ax2.set_title("Curva de Carga Horária Real (Mediana & Banda P10–P90)")
        ax2.legend(loc="upper left", fontsize=8)
    else:
        # Real daily telemetry fallback from uc_day_features
        curve_mode = "daily_uc_day_features"
        day_a = load_uc_daily_timeseries(uc_a["uc_raw"], base_dir=PROJECT_ROOT / "output", days=30)
        day_b = load_uc_daily_timeseries(uc_b["uc_raw"], base_dir=PROJECT_ROOT / "output", days=30)
        if day_a.get("points"):
            ax2.plot(
                range(len(day_a["points"])),
                [p["value"] for p in day_a["points"]],
                color="#2563eb",
                lw=2.0,
                marker="o",
                ms=3,
                label=f"UC {uc_a['uc_censored']} (Diário Real)",
            )
        if day_b.get("points"):
            ax2.plot(
                range(len(day_b["points"])),
                [p["value"] for p in day_b["points"]],
                color="#ea580c",
                lw=2.0,
                marker="s",
                ms=3,
                label=f"UC {uc_b['uc_censored']} (Diário Real)",
            )
        ax2.set_xlabel("Dias na Janela Analítica (uc_day_features)")
        ax2.set_ylabel("Consumo Diário FA_INTERVAL_SUM (kWh/dia)")
        ax2.set_title("Trajetória Diária Real na Janela de 30 Dias")
        ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        f"Comparação de Perfil de Carga: UCs com Consumo Mediano Equivalente "
        f"({uc_a['uc_censored']} vs {uc_b['uc_censored']})",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return {"mode": curve_mode}


def plot_weekly_comparison_pair(
    uc_a: dict[str, Any],
    uc_b: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """
    Plot 7-day (168-hour) real telemetry comparison between the stable and volatile UCs:
      - Top panel: Both 168h curves overlaid on the exact same scale, contrasting the
        continuous ~0.85 kWh/h flat line vs the 0–7 kWh/h diurnal pulsing mountains.
      - Bottom panel: Volatile UC with its MAD operational band and detected demand spikes.
    """
    _apply_dark_academic_style()
    ts_a = load_uc_hourly_timeseries(
        uc_a["uc_raw"],
        base_dir=PROJECT_ROOT / "output",
        cutoff_date=uc_a.get("cutoff") or None,
        week_days=7,
        baseline_days=28,
    )
    ts_b = load_uc_hourly_timeseries(
        uc_b["uc_raw"],
        base_dir=PROJECT_ROOT / "output",
        cutoff_date=uc_b.get("cutoff") or None,
        week_days=7,
        baseline_days=28,
    )

    pts_a = ts_a.get("points", [])
    pts_b = ts_b.get("points", [])
    if not pts_a or not pts_b:
        return {"status": "unavailable", "message": "Sem dados horários suficientes para ambas as UCs."}

    n_pts = min(len(pts_a), len(pts_b))
    xs = np.arange(n_pts)
    vals_a = np.array([p["value"] for p in pts_a[:n_pts]], dtype=np.float64)
    vals_b = np.array([p["value"] for p in pts_b[:n_pts]], dtype=np.float64)
    labels = [p.get("label", str(i)) for i, p in enumerate(pts_b[:n_pts])]
    gran = ts_b.get("granularity", "hourly")
    unit = "kWh/h" if gran == "hourly" else "kWh/dia"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.8), dpi=160, sharex=True)

    # ----------------------------------------------------
    # Painel 1: As duas curvas sobrepostas na MESMA escala
    # ----------------------------------------------------
    ax1.plot(
        xs,
        vals_a,
        color="#2563eb",
        lw=2.3,
        label=f"UC {uc_a['uc_censored']} (Estável: Mediana={uc_a['median_kwh_day']:.1f} kWh/dia | MAD={uc_a['mad_kwh_day']:.2f})",
        zorder=4,
    )
    ax1.plot(
        xs,
        vals_b,
        color="#ea580c",
        lw=1.9,
        label=f"UC {uc_b['uc_censored']} (Volátil: Mediana={uc_b['median_kwh_day']:.1f} kWh/dia | MAD={uc_b['mad_kwh_day']:.2f})",
        zorder=3,
    )

    # Alternar sombreamento sutil por dia (a cada 24 horas)
    days_step = 24 if gran == "hourly" else 1
    total_days = n_pts // days_step
    for d in range(total_days):
        if d % 2 == 1:
            ax1.axvspan(d * days_step, min((d + 1) * days_step, n_pts - 1), color="#f1f5f9", alpha=0.7, zorder=1)
            ax2.axvspan(d * days_step, min((d + 1) * days_step, n_pts - 1), color="#f1f5f9", alpha=0.7, zorder=1)
        # Linha divisória entre dias
        if d > 0:
            ax1.axvline(d * days_step, color="#cbd5e1", ls=":", lw=1.1, zorder=2)
            ax2.axvline(d * days_step, color="#cbd5e1", ls=":", lw=1.1, zorder=2)

    ax1.set_ylabel(f"Energia Ativa ({unit})")
    ax1.set_title(
        f"Acompanhamento Semanal Comparativo (168h): Carga Base Plana vs Regime Comercial com Picos\n"
        f"Ambas com consumo mediano equivalente (~{uc_a['median_kwh_day']:.1f} kWh/dia), mas dinâmicas de rede opostas",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    # ----------------------------------------------------
    # Painel 2: Detalhe da UC Volátil com Banda MAD e Picos
    # ----------------------------------------------------
    if gran == "hourly" and "baseline_hod" in pts_b[0]:
        base_b = np.array([p["baseline_hod"] for p in pts_b[:n_pts]], dtype=np.float64)
        upper_b = np.array([p["upper_hod"] for p in pts_b[:n_pts]], dtype=np.float64)
        lower_b = np.array([p["lower_hod"] for p in pts_b[:n_pts]], dtype=np.float64)
        ax2.fill_between(xs, lower_b, upper_b, color="#ea580c", alpha=0.14, label="Banda Operacional MAD (±3.5σ por hora)")
        ax2.plot(xs, base_b, color="#c2410c", ls="--", lw=1.3, label="Baseline Mediana Histórica da UC Volátil")
    else:
        med_b = float(ts_b.get("baseline_median", uc_b["median_kwh_day"]))
        up_b = float(ts_b.get("upper_band", med_b + 3.5 * float(uc_b["mad_kwh_day"])))
        lo_b = float(ts_b.get("lower_band", max(0.0, med_b - 3.5 * float(uc_b["mad_kwh_day"]))))
        ax2.axhspan(lo_b, up_b, color="#ea580c", alpha=0.12, label="Banda Operacional MAD (±3.5σ)")
        ax2.axhline(med_b, color="#c2410c", ls="--", lw=1.3, label=f"Baseline Mediana ({med_b:.2f} {unit})")

    ax2.plot(xs, vals_b, color="#9a3412", lw=1.8, label=f"Consumo Real UC {uc_b['uc_censored']}")

    # Destacar picos da UC volátil
    spikes_b = [i for i, p in enumerate(pts_b[:n_pts]) if p.get("flag") == "PICO_OUTLIER"]
    if spikes_b:
        ax2.scatter(
            spikes_b,
            vals_b[spikes_b],
            color="#dc2626",
            s=55,
            marker="^",
            zorder=5,
            label="Pico de Demanda Anômalo (Z > +3.5σ)",
        )

    ax2.set_ylabel(f"Energia Ativa ({unit})")
    ax2.set_xlabel("Linha do Tempo Semanal (168 Horas Contínuas)")
    ax2.set_title(
        f"UC Volátil {uc_b['uc_censored']} em Detalhe: Ciclo Diurno, Variação Intra-semanal e Picos",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    # Marcações no eixo X com rótulos de data/hora legíveis
    tick_step = max(1, n_pts // 14)
    ax2.set_xticks(xs[::tick_step])
    ax2.set_xticklabels(labels[::tick_step], rotation=22, ha="right", fontsize=8.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    return {
        "status": "ok",
        "n_hours": n_pts,
        "uc_a_censored": uc_a["uc_censored"],
        "uc_b_censored": uc_b["uc_censored"],
        "max_kwh_a": round(float(np.max(vals_a)), 3),
        "max_kwh_b": round(float(np.max(vals_b)), 3),
        "granularity": gran,
    }


def plot_real_outlier_timeline(
    uc_info: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """
    Plot REAL telemetry from Data Lake (168h hourly via ami_interval when present,
    falling back to real daily series via uc_day_features) showing:
      - Measured consumption curve
      - Baseline median + ±3.5 sigma_MAD band
      - Highlighted CUSUM sustained drop region & step date
      - Highlighted Z-Score MAD > 3.5 spikes
      - Shaded telecom gap periods (coverage < 70%)
    """
    _apply_dark_academic_style()
    ts = load_uc_hourly_timeseries(
        uc_info["uc_raw"],
        base_dir=PROJECT_ROOT / "output",
        cutoff_date=uc_info.get("cutoff") or None,
        week_days=7,
        baseline_days=28,
    )

    if ts.get("status") != "ok" or not ts.get("points"):
        raise RuntimeError(
            f"Sem série temporal real disponível para a UC {uc_info['uc_censored']} "
            f"em output/measurements/ami_interval ou output/features/uc_day_features."
        )

    pts = ts["points"]
    xs = np.arange(len(pts))
    vals = np.array([p["value"] for p in pts], dtype=np.float64)
    labels = [p["label"] for p in pts]
    flags = [p["flag"] for p in pts]
    gran = ts["granularity"]

    fig, (ax_top, ax_cov) = plt.subplots(
        2, 1, figsize=(13.5, 6.2), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3.3, 1.0]}
    )

    # Top panel: Consumption + baseline + MAD band
    if gran == "hourly" and "baseline_hod" in pts[0]:
        base_line = np.array([p["baseline_hod"] for p in pts], dtype=np.float64)
        upper_line = np.array([p["upper_hod"] for p in pts], dtype=np.float64)
        lower_line = np.array([p["lower_hod"] for p in pts], dtype=np.float64)
        ax_top.fill_between(xs, lower_line, upper_line, color="#0284c7", alpha=0.13, label="Banda Operacional MAD (±3.5σ por hora)")
        ax_top.plot(xs, base_line, color="#0369a1", ls="--", lw=1.3, label="Baseline Mediana Histórica (Hora do Dia)")
        unit_lbl = "kWh/h"
        sub_lbl = f"Série Real Horária (7 dias / {len(pts)}h — ami_interval)"
    else:
        med = float(ts["baseline_median"])
        up = float(ts["upper_band"])
        lo = float(ts["lower_band"])
        ax_top.axhspan(lo, up, color="#0284c7", alpha=0.12, label="Banda Operacional MAD (±3.5σ)")
        ax_top.axhline(med, color="#0369a1", ls="--", lw=1.4, label=f"Baseline Mediana ({med:.2f} kWh/dia)")
        unit_lbl = "kWh/dia"
        sub_lbl = f"Série Real Diária ({len(pts)} dias — uc_day_features)"

    ax_top.plot(xs, vals, color="#0f172a", lw=1.7, label="Consumo Real Medido (FA)")

    # Shade Telecom Gaps (coverage < 0.70)
    gap_labeled = False
    for i, f in enumerate(flags):
        if f == "GAP_TELECOM":
            ax_top.axvspan(i - 0.5, i + 0.5, color="#94a3b8", alpha=0.25, label="Gap Telecom (Cobertura < 70%)" if not gap_labeled else None)
            gap_labeled = True

    # Highlight Step Change (Sustained Drop)
    step = ts.get("step_change")
    if step and step.get("direction") == "QUEDA_DEGRAU":
        s_idx = int(step["index"])
        ax_top.axvspan(s_idx, len(xs) - 1, color="#f59e0b", alpha=0.13, label=f"Regime Pós-Degrau ({step['delta_pct']:+.1f}%)")
        ax_top.axvline(s_idx, color="#d97706", ls="-.", lw=1.8, label=f"Alarme CUSUM: Início da Queda ({labels[s_idx]})")

    # Highlight Spikes & Drops
    spike_xs = [i for i, f in enumerate(flags) if f == "PICO_OUTLIER"]
    if spike_xs:
        ax_top.scatter(
            spike_xs, vals[spike_xs], color="#dc2626", s=65, zorder=5, marker="^",
            label="Pico Anômalo (Z-Score MAD > +3.5)"
        )
        peak_idx = max(spike_xs, key=lambda idx: vals[idx])
        ax_top.annotate(
            f"Pico: {vals[peak_idx]:.2f} {unit_lbl}",
            xy=(peak_idx, vals[peak_idx]),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#dc2626", alpha=0.9),
        )

    drop_xs = [i for i, f in enumerate(flags) if f == "QUEDA_OUTLIER"]
    if drop_xs:
        ax_top.scatter(
            drop_xs, vals[drop_xs], color="#d97706", s=50, zorder=5, marker="v",
            label="Queda Pontual (Z-Score MAD < -3.5)"
        )

    ax_top.set_ylabel(f"Energia Ativa ({unit_lbl})")
    cat_txt = uc_info.get("category") or "ANOMALIA"
    score_txt = f"Score={uc_info['anomaly_score']:.3f}" if uc_info.get("anomaly_score") is not None else ""
    ax_top.set_title(
        f"Acompanhamento Real de Anomalia — UC {uc_info['uc_censored']} | {cat_txt} {score_txt}\n{sub_lbl}"
    )
    ax_top.legend(loc="upper left", fontsize=8, ncol=2)

    # Bottom panel: Real Telemetry Coverage %
    covs = [p["coverage"] * 100.0 for p in pts]
    bar_colors = ["#ef4444" if c < 70.0 else "#10b981" for c in covs]
    ax_cov.bar(xs, covs, color=bar_colors, alpha=0.8, width=0.8)
    ax_cov.axhline(70.0, color="#64748b", ls=":", lw=1.2, label="Threshold Qualidade Telecom (70%)")
    ax_cov.set_ylim(0, 108)
    ax_cov.set_ylabel("Cobertura (%)")
    ax_cov.set_xlabel("Linha do Tempo Real")
    ax_cov.legend(loc="lower right", fontsize=8)

    tick_step = max(1, len(labels) // 10)
    ax_cov.set_xticks(xs[::tick_step])
    ax_cov.set_xticklabels(labels[::tick_step], rotation=22, ha="right", fontsize=8.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    return {
        "granularity": gran,
        "points_count": len(pts),
        "window_start": ts.get("window_start") or (labels[0] if labels else ""),
        "window_end": ts.get("window_end") or (labels[-1] if labels else ""),
        "baseline_median": ts.get("baseline_median"),
        "baseline_mad": ts.get("baseline_mad"),
        "slope_kwh_day": ts.get("slope_kwh_day"),
        "step_change": step,
        "detected_anomalies_count": len(ts.get("anomalies", [])),
        "detected_anomalies": ts.get("anomalies", [])[:10],
    }


def _sanitized_summary(info: dict[str, Any]) -> dict[str, Any]:
    """Return copy safe for docs/ without raw uc_raw."""
    return {k: v for k, v in info.items() if k != "uc_raw"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_interest_config()
    manual_pair = cfg.get("ucs_comparacao_par") if isinstance(cfg.get("ucs_comparacao_par"), list) else None
    manual_outlier = cfg.get("uc_outlier_principal")

    df = _load_classified_df()
    cols = _normalize_schema(df)

    pair = find_contrasting_uc_pair(df, cols, manual_ucs=manual_pair)
    outlier = find_outlier_with_drop_and_spike(df, cols, manual_uc=manual_outlier)

    public_meta: dict[str, Any] = {"lgpd_censored": True}
    private_meta: dict[str, Any] = {}

    if pair is not None:
        uc_a, uc_b = pair
        pair_path = OUT_DIR / "comparacao_duas_ucs_mesmo_consumo.png"
        pair_stats = plot_comparison_pair(uc_a, uc_b, pair_path)

        weekly_pair_path = OUT_DIR / "acompanhamento_semanal_comparacao_par.png"
        weekly_pair_stats = plot_weekly_comparison_pair(uc_a, uc_b, weekly_pair_path)

        public_meta["contrasting_pair"] = {
            "uc_stable": _sanitized_summary(uc_a),
            "uc_volatile": _sanitized_summary(uc_b),
            **pair_stats,
            "weekly_timeline": weekly_pair_stats,
        }
        private_meta["contrasting_pair"] = {
            "uc_stable_raw": uc_a["uc_raw"],
            "uc_stable_censored": uc_a["uc_censored"],
            "uc_volatile_raw": uc_b["uc_raw"],
            "uc_volatile_censored": uc_b["uc_censored"],
        }
        print(
            f"[Par Contrastante] {uc_a['uc_censored']} (real: {uc_a['uc_raw']}, med={uc_a['median_kwh_day']}, mad={uc_a['mad_kwh_day']}) "
            f"vs {uc_b['uc_censored']} (real: {uc_b['uc_raw']}, med={uc_b['median_kwh_day']}, mad={uc_b['mad_kwh_day']})"
        )
        print(f" -> Salvo em {pair_path}")
        print(f" -> Acompanhamento Semanal do Par salvo em {weekly_pair_path}")

    if outlier is not None:
        outlier_path = OUT_DIR / "acompanhamento_semanal_outlier.png"
        ts_stats = plot_real_outlier_timeline(outlier, outlier_path)
        public_meta["outlier_timeline"] = {
            "uc": _sanitized_summary(outlier),
            **ts_stats,
        }
        private_meta["outlier_timeline"] = {
            "uc_raw": outlier["uc_raw"],
            "uc_censored": outlier["uc_censored"],
            "cutoff": outlier.get("cutoff"),
        }
        print(
            f"[Outlier Real] {outlier['uc_censored']} (real: {outlier['uc_raw']}, cat={outlier['category']}, "
            f"score={outlier['anomaly_score']}, slope={outlier['slope']})"
        )
        print(f" -> Salvo em {outlier_path}")

    (OUT_DIR / "examples_metadata.json").write_text(
        json.dumps(public_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORTS_DIR / "private_examples_lookup.json").write_text(
        json.dumps(private_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Export Top 30 Anomaly Candidates for Study Case Review
    candidates = find_top_outlier_candidates(df, cols, top_n=30)
    pub_cands = [_sanitized_summary(c) for c in candidates]
    (REPORTS_DIR / "top_candidates_review.json").write_text(
        json.dumps(pub_cands, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORTS_DIR / "top_candidates_private.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Candidatos para Estudo de Caso] {len(candidates)} UCs exportadas para {REPORTS_DIR / 'top_candidates_review.json'}")
    print("Concluído com sucesso.")


if __name__ == "__main__":
    main()
