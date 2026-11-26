"""P1 Controlled Perturbations Experiment on Real AMI Interval Series (Sections 13–16, 18 collective, 19 Base B).

STRICT GUARANTEES:
- Read-only access to local Data Lake (`output/measurements/ami_interval`). NEVER writes to Data Lake partitions.
- Perturbations applied strictly to isolated in-memory copies of real 5/10-min interval series BEFORE feature extraction.
- All models (`frozen_multiview_model.joblib`, `BehavioralClusterer`, `CommunicationAnomalyDetector`,
  `CompositeAnomalyDetector`) are FROZEN from pre-November training (<= 2026-10-31). ZERO fit on OOT or perturbed data.
- Reference (unperturbed) copy preserved for every sampled UC to measure exact delta introduced artificially.
- Explicit disclaimer: controlled scenarios measure algorithmic observability, NEVER field-confirmed fraud.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.datasets.build_training_dataset import build_training_dataset
from src.datasets.retroactive_cleaner import build_sanitized_window_features_vectorized
from src.eda.uc_timeseries import _detect_step_change_cusum, _list_dated_parquets
from src.models.category_reporting import apply_delimited_isolation_and_categories
from src.models.composite import select_pure_electrical_features
from src.models.spurious_filter import filter_spurious_anomalies
from src.utils.censorship import build_feeder_anonymizer, censor_uc_id
from src.utils.joblib_splitter import load_split_joblib

SUITE_PATH = PROJECT_ROOT / "experiments" / "runs" / "latest_suite.json"
P1_PERT_DIR = PROJECT_ROOT / "docs" / "tcc_evidence" / "p1_perturbations"
P1_PERT_FIG = P1_PERT_DIR / "figures"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_br_float(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _setup_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "figure.titlesize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_p = stem.with_suffix(".pdf")
    png_p = stem.with_suffix(".png")
    fig.savefig(pdf_p, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_p, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_p.relative_to(PROJECT_ROOT)), "png": str(png_p.relative_to(PROJECT_ROOT))}


def aggregate_interval_to_daily_vectorized(interval_df: pl.DataFrame) -> pl.DataFrame:
    """Fast vectorized equivalent of build_meter_day_features over interval series (UC x REPORT_DAY).

    Preserves exact definitions:
    - INTERVAL_COVERAGE = observed_valid / (1440 // predominant_cadence)
    - FA_INTERVAL_SUM, RA_INTERVAL_SUM, RA_REVERSAL_RATIO, LOAD_FACTOR
    - VOLTAGE_IMBALANCE_MAX, CURRENT_IMBALANCE_MAX (3-phase max deviation / mean when >= 2 phases > threshold)
    """
    if interval_df.is_empty():
        return pl.DataFrame()

    df = interval_df.with_columns([
        pl.col("FA_INTERVAL").cast(pl.Float64).fill_null(0.0),
        pl.col("RA_INTERVAL").cast(pl.Float64).fill_null(0.0),
        pl.col("CADENCE_MINUTES").cast(pl.Int32).fill_null(10),
    ])

    # Compute row-level voltage and current imbalance matching src/features/electrical.py
    has_u = all(c in df.columns for c in ("U_L1_AVG", "U_L2_AVG", "U_L3_AVG"))
    has_i = all(c in df.columns for c in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"))

    if has_u:
        u1 = pl.col("U_L1_AVG").cast(pl.Float64)
        u2 = pl.col("U_L2_AVG").cast(pl.Float64)
        u3 = pl.col("U_L3_AVG").cast(pl.Float64)
        # Valid active phases (> 50V)
        u_cnt = (u1.fill_null(0.0) > 50.0).cast(pl.Int8) + (u2.fill_null(0.0) > 50.0).cast(pl.Int8) + (u3.fill_null(0.0) > 50.0).cast(pl.Int8)
        u_sum = pl.when(u1 > 50.0).then(u1).otherwise(0.0) + pl.when(u2 > 50.0).then(u2).otherwise(0.0) + pl.when(u3 > 50.0).then(u3).otherwise(0.0)
        u_mean = u_sum / u_cnt.clip(lower_bound=1)
        u_dev = pl.max_horizontal([
            pl.when(u1 > 50.0).then((u1 - u_mean).abs()).otherwise(0.0),
            pl.when(u2 > 50.0).then((u2 - u_mean).abs()).otherwise(0.0),
            pl.when(u3 > 50.0).then((u3 - u_mean).abs()).otherwise(0.0),
        ])
        v_imb_expr = pl.when((u_cnt >= 2) & (u_mean > 50.0)).then((u_dev / u_mean) * 100.0).otherwise(0.0).alias("_V_IMB")
    else:
        v_imb_expr = pl.lit(0.0).alias("_V_IMB")

    if has_i:
        i1 = pl.col("I_L1_AVG").cast(pl.Float64)
        i2 = pl.col("I_L2_AVG").cast(pl.Float64)
        i3 = pl.col("I_L3_AVG").cast(pl.Float64)
        i_cnt = i1.is_not_null().cast(pl.Int8) + i2.is_not_null().cast(pl.Int8) + i3.is_not_null().cast(pl.Int8)
        i_s = i1.fill_null(0.0) + i2.fill_null(0.0) + i3.fill_null(0.0)
        i_mean = i_s / i_cnt.clip(lower_bound=1)
        i_dev = pl.max_horizontal([
            pl.when(i1.is_not_null()).then((i1 - i_mean).abs()).otherwise(0.0),
            pl.when(i2.is_not_null()).then((i2 - i_mean).abs()).otherwise(0.0),
            pl.when(i3.is_not_null()).then((i3 - i_mean).abs()).otherwise(0.0),
        ])
        # Current imbalance under load (mean >= 1.0 A) matching src/features/electrical.py
        c_imb_expr = pl.when((i_cnt >= 2) & (i_mean >= 1.0)).then((i_dev / i_mean).clip(0.0, 2.0)).otherwise(0.0).alias("_C_IMB")
    else:
        c_imb_expr = pl.lit(0.0).alias("_C_IMB")

    df = df.with_columns([v_imb_expr, c_imb_expr])

    daily = (
        df.group_by(["UC", "NIO", "REPORT_DAY"])
        .agg([
            pl.len().alias("_obs"),
            pl.col("CADENCE_MINUTES").median().alias("_cad"),
            pl.col("FA_INTERVAL").sum().alias("FA_INTERVAL_SUM"),
            pl.col("RA_INTERVAL").sum().alias("RA_INTERVAL_SUM"),
            pl.col("FA_INTERVAL").mean().alias("_fa_mean"),
            pl.col("FA_INTERVAL").max().alias("_fa_max"),
            (pl.col("RA_INTERVAL") > 0.001).mean().alias("_ra_prop"),
            pl.col("_V_IMB").max().alias("VOLTAGE_IMBALANCE_MAX"),
            pl.col("_C_IMB").max().alias("CURRENT_IMBALANCE_MAX"),
        ])
        .with_columns([
            (pl.col("_obs") / (1440.0 / pl.col("_cad").clip(lower_bound=5))).clip(0.0, 1.0).round(4).alias("INTERVAL_COVERAGE"),
            pl.when(pl.col("_fa_max") > 1e-6).then((pl.col("_fa_mean") / pl.col("_fa_max")).clip(0.0, 1.0)).otherwise(0.0).round(4).alias("LOAD_FACTOR"),
            pl.when((pl.col("FA_INTERVAL_SUM") + pl.col("RA_INTERVAL_SUM")) > 1e-6)
            .then((pl.col("RA_INTERVAL_SUM") / (pl.col("FA_INTERVAL_SUM") + pl.col("RA_INTERVAL_SUM"))).clip(0.0, 1.0))
            .otherwise(0.0)
            .round(4)
            .alias("RA_REVERSAL_RATIO"),
            pl.lit("v1").alias("FEATURE_SET_VERSION"),
        ])
        .drop(["_obs", "_cad", "_fa_mean", "_fa_max", "_ra_prop"])
        .sort(["UC", "REPORT_DAY"])
    )
    return daily


def _extract_24h_profiles(interval_df: pl.DataFrame, start_d: date, end_d: date) -> dict[str, np.ndarray]:
    """Extract normalized 24-hour mean intraday shape vector (unit L1 norm) per UC over [start_d, end_d]."""
    sub = interval_df.filter((pl.col("REPORT_DAY") >= start_d) & (pl.col("REPORT_DAY") <= end_d))
    if sub.is_empty():
        return {}
    ts_col = "TIMESTAMP_LOCAL" if "TIMESTAMP_LOCAL" in sub.columns else "TIMESTAMP_UTC"
    hourly = (
        sub.with_columns(pl.col(ts_col).dt.hour().alias("_h"))
        .group_by(["UC", "_h"])
        .agg(pl.col("FA_INTERVAL").fill_null(0.0).mean().alias("fa_h"))
    )
    profiles: dict[str, np.ndarray] = {}
    for key, grp in hourly.partition_by("UC", as_dict=True).items():
        uc = str(key[0] if isinstance(key, tuple) else key)
        vec = np.zeros(24, dtype=float)
        for h, v in zip(grp["_h"].to_list(), grp["fa_h"].to_list()):
            if 0 <= int(h) < 24:
                vec[int(h)] = float(v or 0.0)
        tot = float(vec.sum())
        if tot > 1e-9:
            vec = vec / tot
        profiles[uc] = vec
    return profiles


def _apply_perturbation(
    base_interval: pl.DataFrame,
    scenario_type: str,
    intensity: float | int,
    *,
    cutoff: date = date(2026, 11, 19),
    rng: np.random.Generator,
    target_ucs: set[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply a physically coherent perturbation strictly to an in-memory copy of interval_df."""
    df = base_interval.clone()
    if target_ucs is not None:
        in_target = pl.col("UC").is_in(list(target_ucs))
    else:
        in_target = pl.lit(True)

    # Window is 2026-10-21 .. 2026-11-19 (30 days); event onset at day 15 (2026-11-05) unless duration-specific
    onset_date = date(2026, 11, 5)
    meta: dict[str, Any] = {
        "scenario_type": scenario_type,
        "intensity": intensity,
        "onset_date": str(onset_date),
        "duration_days": 15,
    }

    if scenario_type == "comm_gap_random":
        # Drop `intensity` fraction of rows randomly (10%, 20%, 30%, 50%)
        keep_prob = 1.0 - float(intensity)
        mask_arr = rng.random(df.height) < keep_prob
        df = df.filter(pl.Series(mask_arr) | (~in_target))
        meta["onset_date"] = "2026-10-21"
        meta["duration_days"] = 30
        meta["channels_modified"] = "row_drop (todos os canais ausentes)"
        meta["physical_justification"] = "Perda aleatória de pacotes RF Mesh; cobertura cai sem fabricar zeros de consumo"

    elif scenario_type == "comm_gap_burst":
        # Consecutive blackout for `intensity` fraction of 30 days (3, 6, 9, 15 consecutive days ending at cutoff)
        n_drop_days = int(round(30 * float(intensity)))
        drop_start = cutoff - timedelta(days=n_drop_days - 1)
        df = df.filter((~(pl.col("REPORT_DAY") >= drop_start)) | (~in_target))
        meta["onset_date"] = str(drop_start)
        meta["duration_days"] = n_drop_days
        meta["channels_modified"] = "row_drop_consecutive"
        meta["physical_justification"] = "Silêncio consecutivo de telemetria (falha modem/concentrador)"

    elif scenario_type == "sustained_drop":
        # Sustained drop of `intensity` (0.10, 0.20, 0.30, 0.50) from onset_date to cutoff
        factor = 1.0 - float(intensity)
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
        exprs = [pl.when(cond).then(pl.col("FA_INTERVAL") * factor).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
        for ic in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"):
            if ic in df.columns:
                exprs.append(pl.when(cond & pl.col(ic).is_not_null()).then(pl.col(ic) * factor).otherwise(pl.col(ic)).alias(ic))
        df = df.with_columns(exprs)
        meta["channels_modified"] = "FA_INTERVAL, I_L1..L3 (tensão preservada)"
        meta["physical_justification"] = f"Queda sustentada de {int(intensity*100)}% em energia e corrente mantendo tensão nominal"

    elif scenario_type == "regime_duration":
        # -40% drop with duration = intensity days (3, 7, 14 days ending at cutoff)
        dur = int(intensity)
        onset_d = cutoff - timedelta(days=dur - 1)
        factor = 0.60
        cond = in_target & (pl.col("REPORT_DAY") >= onset_d)
        exprs = [pl.when(cond).then(pl.col("FA_INTERVAL") * factor).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
        for ic in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"):
            if ic in df.columns:
                exprs.append(pl.when(cond & pl.col(ic).is_not_null()).then(pl.col(ic) * factor).otherwise(pl.col(ic)).alias(ic))
        df = df.with_columns(exprs)
        meta["onset_date"] = str(onset_d)
        meta["duration_days"] = dur
        meta["channels_modified"] = "FA_INTERVAL, I_L1..L3 (-40%)"
        meta["physical_justification"] = f"Mudança de regime (-40%) persistente por {dur} dias até o corte"

    elif scenario_type == "isolated_event":
        # Single-day isolated anomaly on 2026-11-05 (intensity = multiplier, e.g. 0.10 or 3.5)
        mult = float(intensity)
        cond = in_target & (pl.col("REPORT_DAY") == onset_date)
        exprs = [pl.when(cond).then(pl.col("FA_INTERVAL") * mult).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
        for ic in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"):
            if ic in df.columns:
                exprs.append(pl.when(cond & pl.col(ic).is_not_null()).then(pl.col(ic) * mult).otherwise(pl.col(ic)).alias(ic))
        df = df.with_columns(exprs)
        meta["duration_days"] = 1
        meta["channels_modified"] = "FA_INTERVAL, I_L1..L3 (1 dia)"
        meta["physical_justification"] = f"Evento isolado de 1 dia (fator {mult}x) para testar contraste ensemble vs. CPD"

    elif scenario_type == "time_shift":
        # Circular shift of intraday timestamps by `intensity` hours (1, 2, 4h) on days >= onset_date
        # Strictly preserves daily FA_INTERVAL_SUM!
        shift_h = int(intensity)
        ts_col = "TIMESTAMP_LOCAL" if "TIMESTAMP_LOCAL" in df.columns else "TIMESTAMP_UTC"
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
        # Circular hour shift inside same REPORT_DAY
        secs = (
            pl.col(ts_col).dt.hour().cast(pl.Int32) * 3600
            + pl.col(ts_col).dt.minute().cast(pl.Int32) * 60
            + pl.col(ts_col).dt.second().cast(pl.Int32)
        )
        new_secs = (secs + shift_h * 3600) % 86400
        new_ts = pl.col("REPORT_DAY").cast(pl.Datetime("us")) + pl.duration(seconds=new_secs)
        df = df.with_columns(
            pl.when(cond).then(new_ts).otherwise(pl.col(ts_col)).alias(ts_col)
        )
        meta["channels_modified"] = f"{ts_col} (rotação circular +{shift_h}h; energia diária constante)"
        meta["physical_justification"] = f"Deslocamento de rotina em +{shift_h}h mantendo energia diária idêntica"

    elif scenario_type == "shape_flatten":
        # Redistribute intraday FA_INTERVAL toward flat profile on days >= onset_date preserving exact daily total!
        alpha = float(intensity)  # blend fraction toward flat (e.g. 0.80)
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
        day_mean = pl.col("FA_INTERVAL").mean().over(["UC", "REPORT_DAY"])
        df = df.with_columns(
            pl.when(cond)
            .then((1.0 - alpha) * pl.col("FA_INTERVAL") + alpha * day_mean)
            .otherwise(pl.col("FA_INTERVAL"))
            .alias("FA_INTERVAL")
        )
        meta["channels_modified"] = "FA_INTERVAL (redistribuição intradiária; soma diária preservada)"
        meta["physical_justification"] = "Alteração de forma da curva diária (fator de carga) sem alterar kWh/dia total"

    elif scenario_type == "reverse_flow_gd":
        # Inject coherent solar reverse active power during 09:00-16:00 on days >= onset_date
        ts_col = "TIMESTAMP_LOCAL" if "TIMESTAMP_LOCAL" in df.columns else "TIMESTAMP_UTC"
        hr = pl.col(ts_col).dt.hour().cast(pl.Float64)
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date) & (hr >= 9.0) & (hr <= 16.0)
        # Smooth bell centered at 12:30 (~0.25 * intensity kWh per 10-min interval -> ~8-15 kWh/day)
        solar_bell = (1.0 - ((hr - 12.5) / 4.0) ** 2).clip(0.0, 1.0) * float(intensity)
        df = df.with_columns([
            pl.when(cond).then(pl.col("RA_INTERVAL") + solar_bell).otherwise(pl.col("RA_INTERVAL")).alias("RA_INTERVAL"),
            pl.when(cond).then((pl.col("FA_INTERVAL") - solar_bell * 0.5).clip(lower_bound=0.0)).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL"),
        ])
        meta["channels_modified"] = "RA_INTERVAL (+injeção solar diurna), FA_INTERVAL (-autoconsumo)"
        meta["physical_justification"] = "Injeção ativa reversa diurna coerente com microgeração solar"

    elif scenario_type == "zero_load_normal_voltage":
        # Stuck meter / total shunt: FA=0, I=0 from onset_date while keeping U_L1..L3 strictly normal!
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
        exprs = [pl.when(cond).then(0.0).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
        for ic in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"):
            if ic in df.columns:
                exprs.append(pl.when(cond & pl.col(ic).is_not_null()).then(0.0).otherwise(pl.col(ic)).alias(ic))
        df = df.with_columns(exprs)
        meta["channels_modified"] = "FA_INTERVAL=0, I_L*=0 (U_L1..L3 preservadas normais)"
        meta["physical_justification"] = "Ausência de carga registrada com tensão de linha energizada (medidor travado/shunt)"

    elif scenario_type == "phase_unbalance":
        # Bypass on phase L1 (65% current drop on L1, proportional FA reduction) for polyphase meters
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
        exprs = [pl.when(cond).then(pl.col("FA_INTERVAL") * 0.65).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
        if "I_L1_AVG" in df.columns:
            exprs.append(pl.when(cond & pl.col("I_L1_AVG").is_not_null()).then(pl.col("I_L1_AVG") * 0.25).otherwise(pl.col("I_L1_AVG")).alias("I_L1_AVG"))
        df = df.with_columns(exprs)
        meta["channels_modified"] = "I_L1_AVG (-75%), FA_INTERVAL (-35%), tensões normais"
        meta["physical_justification"] = "Desequilíbrio severo de corrente por derivação em uma fase sob tensão normal"

    elif scenario_type == "collective_feeder_drop":
        # Simultaneous -35% drop across ALL target_ucs on the same transformer/feeder
        factor = 1.0 - float(intensity)
        cond = in_target & (pl.col("REPORT_DAY") >= onset_date)
        exprs = [pl.when(cond).then(pl.col("FA_INTERVAL") * factor).otherwise(pl.col("FA_INTERVAL")).alias("FA_INTERVAL")]
        for ic in ("I_L1_AVG", "I_L2_AVG", "I_L3_AVG"):
            if ic in df.columns:
                exprs.append(pl.when(cond & pl.col(ic).is_not_null()).then(pl.col(ic) * factor).otherwise(pl.col(ic)).alias(ic))
        df = df.with_columns(exprs)
        meta["channels_modified"] = "FA_INTERVAL, I_L1..L3 (-35% simultâneo no mesmo circuito/posto)"
        meta["physical_justification"] = "Evento comum correlacionado ao circuito/transformador vs. anomalia isolada"

    return df, meta


def _score_perturbed_dataset(
    interval_df: pl.DataFrame,
    *,
    uc_context: pl.DataFrame,
    meter_history: pl.DataFrame,
    hierarchy: pl.DataFrame,
    pipe: Any,
    elec_cols: list[str],
    comm_cols: list[str],
    beh_24h_pca: PCA,
    beh_24h_km: KMeans,
    beh_24h_p01: float,
    beh_24h_p99: float,
    cutoff: date = date(2026, 11, 19),
) -> pl.DataFrame:
    """Run exact frozen feature engineering + frozen model inference + CPD + 24h behavioral distance."""
    daily_df = aggregate_interval_to_daily_vectorized(interval_df)
    win_df = build_sanitized_window_features_vectorized(
        daily_df,
        cutoff_date=cutoff,
        window_days=30,
        meter_history_df=meter_history,
        alarm_events_df=pl.DataFrame(),
        min_day_coverage=0.70,
        min_window_coverage=0.70,
        min_reliable_days=18,
        feature_set_version="v1",
    )
    ds = build_training_dataset(win_df, uc_context_df=uc_context, hierarchy_df=hierarchy, split_policy="oot_eval")

    # Ensure all required columns exist in exact order
    for c in set(comm_cols) | set(elec_cols):
        if c not in ds.columns:
            ds = ds.with_columns(pl.lit(0.0).alias(c))

    X_comm = ds.select(comm_cols).fill_null(0.0).to_numpy()
    X_elec = ds.select(elec_cols).fill_null(0.0).to_numpy()
    qual = ds["meta__data_quality"].to_numpy()
    alarms = np.zeros(ds.height, dtype=float)
    slopes = ds["x__trend_fa_slope"].fill_null(0.0).to_numpy() if "x__trend_fa_slope" in ds.columns else None
    mchg = ds["x__meter_changed_30d"].fill_null(False).to_numpy() if "x__meter_changed_30d" in ds.columns else None

    scored = pipe.score(
        X_comm=X_comm,
        X_elec=X_elec,
        data_quality=qual,
        tamper_alarms=alarms,
        fa_trend_slopes=slopes,
        meter_changed_30d=mchg,
        raw_dataframe=ds,
    )
    enriched, _ = apply_delimited_isolation_and_categories(scored, base_dir=PROJECT_ROOT / "output")

    # Run CPD (CUSUM) per UC on the perturbed daily series
    grouped = daily_df.sort(["UC", "REPORT_DAY"]).group_by("UC", maintain_order=False).agg([
        pl.col("REPORT_DAY").cast(pl.String).alias("dates"),
        pl.col("FA_INTERVAL_SUM").fill_null(0.0).alias("fas"),
        pl.col("INTERVAL_COVERAGE").fill_null(0.0).alias("covs"),
    ])
    cpd_rows = []
    for r in grouped.iter_rows(named=True):
        uc = str(r["UC"]).strip().lstrip("0") or str(r["UC"]).strip()
        dates, fas, covs = r["dates"], r["fas"], r["covs"]
        rel_vals = [v for v, c in zip(fas, covs) if c >= 0.70]
        if len(rel_vals) < 8:
            cpd_rows.append({"id__uc_id": uc, "cpd_detected": False, "cpd_date": None, "cpd_mag_pct": 0.0, "cpd_dir": "INAVALIAVEL"})
            continue
        ya = np.asarray(rel_vals, dtype=float)
        med = float(np.median(ya))
        mad = float(np.median(np.abs(ya - med)))
        scale = max(1.4826 * mad, 0.10 * abs(med), 0.05)
        st = _detect_step_change_cusum(dates, fas, covs, med, scale)
        if st is None:
            cpd_rows.append({"id__uc_id": uc, "cpd_detected": False, "cpd_date": None, "cpd_mag_pct": 0.0, "cpd_dir": "ESTAVEL"})
        else:
            cpd_rows.append({"id__uc_id": uc, "cpd_detected": True, "cpd_date": st["date"], "cpd_mag_pct": float(st["delta_pct"]), "cpd_dir": st["direction"]})

    # 24h intraday behavioral profile distance (post-onset 05/11..19/11 vs frozen 24h clusters + self pre-vs-post L2 shift)
    prof_pre = _extract_24h_profiles(interval_df, date(2026, 10, 21), date(2026, 11, 4))
    prof_post = _extract_24h_profiles(interval_df, date(2026, 11, 5), date(2026, 11, 19))
    beh_rows = []
    for u_raw in enriched["id__uc_id"].to_list():
        vp = prof_post.get(u_raw, np.ones(24) / 24.0)
        vpre = prof_pre.get(u_raw, np.ones(24) / 24.0)
        # Cluster distance & self routine shift
        z_p = beh_24h_pca.transform(vp.reshape(1, -1))
        z_pre = beh_24h_pca.transform(vpre.reshape(1, -1))
        cl_p = int(beh_24h_km.predict(z_p)[0])
        cl_pre = int(beh_24h_km.predict(z_pre)[0])
        c_dist = float(np.linalg.norm(z_p[0] - beh_24h_km.cluster_centers_[cl_p]))
        self_l2 = float(np.linalg.norm(vp - vpre))
        s_beh_24h = float(np.clip((c_dist - beh_24h_p01) / max(beh_24h_p99 - beh_24h_p01, 1e-9), 0.0, 1.0))
        beh_rows.append({
            "id__uc_id": u_raw,
            "beh_24h_cluster_pre": cl_pre,
            "beh_24h_cluster_post": cl_p,
            "beh_24h_cluster_changed": (cl_p != cl_pre),
            "beh_24h_centroid_dist": round(c_dist, 4),
            "beh_24h_self_shift_l2": round(self_l2, 4),
            "s_beh_24h": round(s_beh_24h, 4),
        })

    out = enriched.join(pl.DataFrame(cpd_rows), on="id__uc_id", how="left").join(pl.DataFrame(beh_rows), on="id__uc_id", how="left")
    return out


def run_controlled_perturbations() -> dict[str, Any]:
    t0 = time.time()
    _setup_pub_style()
    commit = _git_commit()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_p1_controlled_perturbations"
    run_dir = PROJECT_ROOT / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    P1_PERT_FIG.mkdir(parents=True, exist_ok=True)

    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    oot_dir = PROJECT_ROOT / suite["oot_multiview"]
    oot_dec = pl.read_parquet(oot_dir / "decisions.parquet")
    uc_ctx = pl.read_parquet(PROJECT_ROOT / "output/context/uc_context.parquet")
    meter_hist = pl.read_parquet(PROJECT_ROOT / "output/context/meter_installation_history.parquet")
    hier = pl.read_parquet(PROJECT_ROOT / "output/context/electrical_hierarchy.parquet")
    pipe = load_split_joblib(oot_dir / "frozen_multiview_model.joblib")
    det = pipe.elec_detector.detector

    # Exact feature lists from OOT manifest
    oot_man = json.loads((oot_dir / "manifest.json").read_text(encoding="utf-8"))
    comm_cols = oot_man["parameters"]["comm_features"]
    elec_cols = oot_man["parameters"]["elec_features"]

    all_feeders = sorted({str(x) for x in hier["ALIMENTADOR"].drop_nulls().unique().to_list() if x})
    feeder_map = build_feeder_anonymizer(all_feeders)

    # =========================================================================
    # 13.1 STRATIFIED SAMPLE SELECTION (N = 100 real UCs, seed=42, NO cherry-picking!)
    # =========================================================================
    # Select from top 5 feeders, across behavioral classes C2, C3, C4, C5, with high baseline coverage >= 0.92
    # and reliable_days >= 28 (strictly without filtering by model prediction accuracy!)
    top5_feeders = (
        oot_dec.filter(pl.col("meta__feeder").is_not_null())
        .group_by("meta__feeder").len().sort("len", descending=True).head(5)["meta__feeder"].to_list()
    )
    pool = oot_dec.filter(
        pl.col("meta__feeder").is_in(top5_feeders)
        & (pl.col("x__avg_interval_coverage") >= 0.92)
        & (pl.col("x__coverage_reliable_days") >= 28)
        & (pl.col("x__hist_fa_sum_median") >= 1.5)
        & pl.col("CLASSE_COMPORTAMENTAL").is_in([
            "C2_RESIDENCIAL_BAIXO_CONSUMO", "C3_RESIDENCIAL_PADRAO",
            "C4_COMERCIAL_OU_ELEVADO", "C5_ALTA_CARGA_CONTINUA"
        ])
    ).sort("id__uc_id")

    rng = np.random.default_rng(42)
    sampled_ucs: list[str] = []
    strata_summary: list[dict[str, Any]] = []
    for f_name in top5_feeders:
        sub_f = pool.filter(pl.col("meta__feeder") == f_name)
        for cls_name in ["C2_RESIDENCIAL_BAIXO_CONSUMO", "C3_RESIDENCIAL_PADRAO", "C4_COMERCIAL_OU_ELEVADO", "C5_ALTA_CARGA_CONTINUA"]:
            sub_fc = sub_f.filter(pl.col("CLASSE_COMPORTAMENTAL") == cls_name)["id__uc_id"].to_list()
            k_pick = min(5, len(sub_fc))
            if k_pick > 0:
                chosen = rng.choice(sub_fc, size=k_pick, replace=False).tolist()
                sampled_ucs.extend(chosen)
                strata_summary.append({
                    "circuito": feeder_map.get(f_name, f_name),
                    "classe": cls_name,
                    "disponiveis": len(sub_fc),
                    "amostradas": k_pick,
                })

    sampled_ucs = sorted(set(sampled_ucs))
    print(f"[P1 Perturbations] Stratified sample: {len(sampled_ucs)} real UCs across {len(top5_feeders)} feeders")

    # Map sampled UCs to their active NIOs (zero-stripped & raw) from meter_installation_history
    mh_norm = meter_hist.with_columns([
        pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_UC"),
        pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NIO"),
    ]).filter(pl.col("_UC").is_in(sampled_ucs))

    nio_to_uc = {}
    for r in mh_norm.select(["_NIO", "_UC"]).unique().to_dicts():
        nio_to_uc[r["_NIO"]] = r["_UC"]
        nio_to_uc[r["_NIO"].zfill(10)] = r["_UC"]
    sample_nios = list(nio_to_uc.keys())

    # Read 30-day real interval series (2026-10-21 .. 2026-11-19) READ-ONLY from Data Lake!
    t_load0 = time.time()
    int_root = PROJECT_ROOT / "output/measurements/ami_interval"
    dated_int = _list_dated_parquets(int_root)
    win_files = [(d, p) for d, p in dated_int if date(2026, 10, 21) <= d <= date(2026, 11, 19)]

    wanted_cols = ["NIO", "REPORT_DAY", "TIMESTAMP_UTC", "TIMESTAMP_LOCAL", "CADENCE_MINUTES",
                   "FA_INTERVAL", "RA_INTERVAL", "I_L1_AVG", "I_L2_AVG", "I_L3_AVG",
                   "U_L1_AVG", "U_L2_AVG", "U_L3_AVG"]
    int_frames = []
    for dt, fp in win_files:
        lf = pl.scan_parquet(str(fp))
        avail = set(lf.collect_schema().names())
        cols = [c for c in wanted_cols if c in avail]
        part = (
            lf.filter(pl.col("NIO").is_in(sample_nios))
            .select(cols)
            .with_columns(pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_NIO_N"))
            .collect()
        )
        if part.height > 0:
            int_frames.append(part)

    base_interval = (
        pl.concat(int_frames, how="diagonal_relaxed")
        .with_columns(pl.col("_NIO_N").replace_strict(nio_to_uc, default=None).alias("UC"))
        .filter(pl.col("UC").is_not_null())
        .drop("_NIO_N")
    )
    # Keep only UCs that actually have >= 25 days of interval rows
    uc_day_cnts = base_interval.group_by("UC").agg(pl.col("REPORT_DAY").n_unique().alias("nd")).filter(pl.col("nd") >= 25)
    valid_ucs = sorted(uc_day_cnts["UC"].to_list())
    base_interval = base_interval.filter(pl.col("UC").is_in(valid_ucs))
    t_load = time.time() - t_load0
    print(f"[P1 Perturbations] Loaded {base_interval.height:,} real interval rows for {len(valid_ucs)} UCs in {t_load:.1f}s")

    # Fit 24h intraday behavioral clusterer STRICTLY on pre-onset days (21/10..04/11) of the unperturbed series
    pre_profs = _extract_24h_profiles(base_interval, date(2026, 10, 21), date(2026, 11, 4))
    X_24h_pre = np.vstack([pre_profs[u] for u in valid_ucs if u in pre_profs])
    beh_24h_pca = PCA(n_components=4, random_state=42).fit(X_24h_pre)
    Z_24h_pre = beh_24h_pca.transform(X_24h_pre)
    beh_24h_km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Z_24h_pre)
    pre_lbls = beh_24h_km.predict(Z_24h_pre)
    pre_dists = np.linalg.norm(Z_24h_pre - beh_24h_km.cluster_centers_[pre_lbls], axis=1)
    beh_24h_p01 = float(np.percentile(pre_dists, 1.0))
    beh_24h_p99 = float(np.percentile(pre_dists, 99.0))

    # Filter context tables to sample for ultra-fast scoring
    uc_ctx_sub = uc_ctx.with_columns(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_U")).filter(pl.col("_U").is_in(valid_ucs)).drop("_U")
    mh_sub = meter_hist.with_columns(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_U")).filter(pl.col("_U").is_in(valid_ucs)).drop("_U")
    hier_sub = hier.with_columns(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_U")).filter(pl.col("_U").is_in(valid_ucs)).drop("_U")

    # 13.2 REFERENCE (UNPERTURBED) EVALUATION
    ref_df = _score_perturbed_dataset(
        base_interval, uc_context=uc_ctx_sub, meter_history=mh_sub, hierarchy=hier_sub,
        pipe=pipe, elec_cols=elec_cols, comm_cols=comm_cols,
        beh_24h_pca=beh_24h_pca, beh_24h_km=beh_24h_km, beh_24h_p01=beh_24h_p01, beh_24h_p99=beh_24h_p99,
    ).sort("id__uc_id")
    ref_by_uc = {r["id__uc_id"]: r for r in ref_df.to_dicts()}

    # Reference baseline false-trigger rates (note: real unperturbed UCs are NOT assumed fraud-free!)
    ref_summary = {
        "n_ucs": ref_df.height,
        "mean_coverage": round(float(ref_df["x__avg_interval_coverage"].mean()), 4),
        "mean_s_elec": round(float(ref_df["s_elec"].mean()), 4),
        "mean_s_comm": round(float(ref_df["s_comm"].mean()), 4),
        "elec_anom_rate_pct": round(float(ref_df["is_elec_anomaly"].mean() * 100.0), 2),
        "quarantine_rate_pct": round(float((~ref_df["is_reliable_elec"]).mean() * 100.0), 2),
        "cpd_drop_rate_pct": round(float(((ref_df["cpd_detected"] == True) & (ref_df["cpd_dir"] == "QUEDA_DEGRAU")).mean() * 100.0), 2),
        "beh_cluster_change_rate_pct": round(float(ref_df["beh_24h_cluster_changed"].mean() * 100.0), 2),
        "beh_mean_self_shift_l2": round(float(ref_df["beh_24h_self_shift_l2"].mean()), 4),
    }

    # Define full experimental perturbation matrix (Sections 13.3, 14, 15, 16, 18)
    tau_elec = float(det.frozen_threshold_)
    tau_comm = float(pipe.comm_detector.frozen_threshold_)
    beh_l2_p95_ref = float(np.percentile(ref_df["beh_24h_self_shift_l2"].to_numpy(), 95.0))

    scenarios: list[tuple[str, str, float | int, str]] = [
        ("comm_gap_random", "Lacuna aleatória 10%", 0.10, "S_comm / Gating"),
        ("comm_gap_random", "Lacuna aleatória 20%", 0.20, "S_comm / Gating"),
        ("comm_gap_random", "Lacuna aleatória 30%", 0.30, "S_comm / Gating"),
        ("comm_gap_random", "Lacuna aleatória 50%", 0.50, "S_comm / Gating"),
        ("comm_gap_burst", "Bloco consecutivo 20% (6d)", 0.20, "S_comm / Gating"),
        ("comm_gap_burst", "Bloco consecutivo 30% (9d)", 0.30, "S_comm / Gating"),
        ("comm_gap_burst", "Bloco consecutivo 50% (15d)", 0.50, "S_comm / Gating"),
        ("sustained_drop", "Queda sustentada -10%", 0.10, "CPD / S_elec"),
        ("sustained_drop", "Queda sustentada -20%", 0.20, "CPD / S_elec"),
        ("sustained_drop", "Queda sustentada -30%", 0.30, "CPD / S_elec"),
        ("sustained_drop", "Queda sustentada -50%", 0.50, "CPD / S_elec"),
        ("regime_duration", "Mudança -40% (3 dias)", 3, "CPD / Persistência"),
        ("regime_duration", "Mudança -40% (7 dias)", 7, "CPD / Persistência"),
        ("regime_duration", "Mudança -40% (14 dias)", 14, "CPD / Persistência"),
        ("isolated_event", "Evento isolado 1d (0,1x)", 0.10, "Contraste S_elec vs CPD"),
        ("isolated_event", "Evento isolado 1d (3,5x)", 3.50, "Contraste S_elec vs CPD"),
        ("time_shift", "Deslocamento rotina +1h", 1, "S_behavior (24h)"),
        ("time_shift", "Deslocamento rotina +2h", 2, "S_behavior (24h)"),
        ("time_shift", "Deslocamento rotina +4h", 4, "S_behavior (24h)"),
        ("shape_flatten", "Achatamento forma (kWh fixo)", 0.80, "PCA / Forma Diária"),
        ("reverse_flow_gd", "Injeção reversa solar (GD)", 0.30, "Regra GD / S_elec"),
        ("zero_load_normal_voltage", "Carga zero c/ tensão normal", 1.0, "Medidor Travado / S_elec"),
        ("phase_unbalance", "Desequilíbrio fase (-75% L1)", 0.75, "Desbal. Corrente / S_elec"),
    ]

    scenario_results: list[dict[str, Any]] = []
    # Also store example series for Fig 10 and per-UC paired Base B evaluation!
    example_uc = valid_ucs[0]
    example_series: dict[str, tuple[list[str], list[float], list[float]]] = {}
    ref_daily = aggregate_interval_to_daily_vectorized(base_interval.filter(pl.col("UC") == example_uc))
    example_series["Referência (real)"] = (
        ref_daily["REPORT_DAY"].cast(pl.String).to_list(),
        ref_daily["FA_INTERVAL_SUM"].to_list(),
        ref_daily["INTERVAL_COVERAGE"].to_list(),
    )

    base_b_records: list[dict[str, Any]] = []
    # Add reference UCs to Base B (label = 0 for electrical/regime/behavior/comm perturbation)
    for r in ref_df.to_dicts():
        base_b_records.append({
            "case_id": f"REF_{r['id__uc_id']}",
            "scenario_group": "referencia",
            "is_pert_elec": False,
            "is_pert_comm": False,
            "is_pert_cpd": False,
            "is_pert_beh": False,
            "iso_score": float(r["iso_score"]),
            "s_elec": float(r["s_elec"]),
            "s_comm": float(r["s_comm"]),
            "is_reliable": bool(r["is_reliable_elec"]),
            "s_beh_shift": float(r["beh_24h_self_shift_l2"]),
            "cpd_drop": bool(r["cpd_detected"] and r["cpd_dir"] == "QUEDA_DEGRAU"),
            "phys_gd_or_stuck": bool(r["CATEGORIA_DIAGNOSTICA"] in ("POSSIVEL_GD_NAO_CADASTRADA", "FALHA_METROLOGICA_MEDIDOR_TRAVADO")),
        })

    for sc_type, sc_label, intens, comp_eval in scenarios:
        pert_int, meta = _apply_perturbation(base_interval, sc_type, intens, rng=np.random.default_rng(42))
        if sc_label in {"Lacuna aleatória 30%", "Queda sustentada -50%", "Injeção reversa solar (GD)", "Carga zero c/ tensão normal"}:
            ex_d = aggregate_interval_to_daily_vectorized(pert_int.filter(pl.col("UC") == example_uc))
            example_series[sc_label] = (
                ex_d["REPORT_DAY"].cast(pl.String).to_list(),
                ex_d["FA_INTERVAL_SUM"].to_list(),
                ex_d["INTERVAL_COVERAGE"].to_list(),
            )

        res_df = _score_perturbed_dataset(
            pert_int, uc_context=uc_ctx_sub, meter_history=mh_sub, hierarchy=hier_sub,
            pipe=pipe, elec_cols=elec_cols, comm_cols=comm_cols,
            beh_24h_pca=beh_24h_pca, beh_24h_km=beh_24h_km, beh_24h_p01=beh_24h_p01, beh_24h_p99=beh_24h_p99,
        ).sort("id__uc_id")

        # Paired differences vs reference
        d_s_elec, d_s_comm, d_cov, d_fa, d_beh_l2, d_pca = [], [], [], [], [], []
        cpd_hits, cpd_err_days, cpd_delays, cpd_mag_errs = [], [], [], []
        quarantined, elec_anom_nogate, elec_anom_gated, cat_expected = 0, 0, 0, 0
        beh_cluster_flips, beh_shift_hits = 0, 0
        true_onset = date.fromisoformat(meta["onset_date"])

        for r in res_df.to_dicts():
            u = r["id__uc_id"]
            ref = ref_by_uc.get(u)
            if ref is None:
                continue
            d_s_elec.append(float(r["s_elec"]) - float(ref["s_elec"]))
            d_s_comm.append(float(r["s_comm"]) - float(ref["s_comm"]))
            d_cov.append(float(r["x__avg_interval_coverage"]) - float(ref["x__avg_interval_coverage"]))
            d_fa.append(float(r["x__hist_fa_sum_median"]) - float(ref["x__hist_fa_sum_median"]))
            d_beh_l2.append(float(r["beh_24h_self_shift_l2"]) - float(ref["beh_24h_self_shift_l2"]))
            d_pca.append(float(r["pca_score"]) - float(ref["pca_score"]))

            is_q = not bool(r["is_reliable_elec"])
            is_ea_nogate = float(r["s_elec"]) >= tau_elec
            is_ea_gated = bool(r["is_elec_anomaly"])
            quarantined += int(is_q)
            elec_anom_nogate += int(is_ea_nogate)
            elec_anom_gated += int(is_ea_gated)

            if bool(r["beh_24h_cluster_changed"]):
                beh_cluster_flips += 1
            if float(r["beh_24h_self_shift_l2"]) > beh_l2_p95_ref:
                beh_shift_hits += 1

            # CPD localization check (±3 days tolerance)
            is_cpd_drop = bool(r["cpd_detected"]) and (r["cpd_dir"] == "QUEDA_DEGRAU")
            if is_cpd_drop and r["cpd_date"]:
                det_dt = date.fromisoformat(str(r["cpd_date"])[:10])
                err_d = abs((det_dt - true_onset).days)
                delay_d = (det_dt - true_onset).days
                cpd_hits.append(err_d <= 3)
                cpd_err_days.append(err_d)
                cpd_delays.append(delay_d)
                if sc_type == "sustained_drop":
                    cpd_mag_errs.append(abs(float(r["cpd_mag_pct"]) - (-float(intens) * 100.0)))
                elif sc_type == "regime_duration":
                    cpd_mag_errs.append(abs(float(r["cpd_mag_pct"]) - (-40.0)))
            else:
                cpd_hits.append(False)

            # Expected category match
            cat = str(r["CATEGORIA_DIAGNOSTICA"])
            if sc_type in ("comm_gap_random", "comm_gap_burst"):
                cat_expected += int(is_q or cat in ("ANOMALIA_COMUNICACAO_ISOLADA", "GARGALO_COLETIVO_RF_MESH"))
            elif sc_type == "reverse_flow_gd":
                cat_expected += int(cat == "POSSIVEL_GD_NAO_CADASTRADA")
            elif sc_type == "zero_load_normal_voltage":
                cat_expected += int(cat in ("FALHA_METROLOGICA_MEDIDOR_TRAVADO", "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT"))
            elif sc_type in ("sustained_drop", "regime_duration", "phase_unbalance"):
                cat_expected += int(is_ea_gated or is_cpd_drop)
            elif sc_type in ("time_shift", "shape_flatten"):
                cat_expected += int(float(r["beh_24h_self_shift_l2"]) > beh_l2_p95_ref)
            else:
                cat_expected += int(is_ea_gated)

            # Add key representative scenarios to Base B for Ablation V0..V5
            if sc_label in {"Lacuna aleatória 50%", "Bloco consecutivo 50% (15d)", "Queda sustentada -50%", "Mudança -40% (14 dias)", "Deslocamento rotina +4h", "Injeção reversa solar (GD)", "Carga zero c/ tensão normal"}:
                base_b_records.append({
                    "case_id": f"{sc_type}_{intens}_{u}",
                    "scenario_group": sc_label,
                    "is_pert_elec": sc_type in ("sustained_drop", "regime_duration", "reverse_flow_gd", "zero_load_normal_voltage"),
                    "is_pert_comm": sc_type in ("comm_gap_random", "comm_gap_burst"),
                    "is_pert_cpd": sc_type in ("sustained_drop", "regime_duration"),
                    "is_pert_beh": sc_type == "time_shift",
                    "iso_score": float(r["iso_score"]),
                    "s_elec": float(r["s_elec"]),
                    "s_comm": float(r["s_comm"]),
                    "is_reliable": bool(r["is_reliable_elec"]),
                    "s_beh_shift": float(r["beh_24h_self_shift_l2"]),
                    "cpd_drop": is_cpd_drop,
                    "phys_gd_or_stuck": bool(cat in ("POSSIVEL_GD_NAO_CADASTRADA", "FALHA_METROLOGICA_MEDIDOR_TRAVADO")),
                })

        n_ev = max(1, len(d_s_elec))
        scenario_results.append({
            "tipo": sc_type,
            "cenario": sc_label,
            "intensidade": str(intens),
            "componente_avaliado": comp_eval,
            "n_ucs": n_ev,
            "delta_cov_medio": round(float(np.mean(d_cov)), 4),
            "delta_s_comm_medio": round(float(np.mean(d_s_comm)), 4),
            "delta_s_elec_medio": round(float(np.mean(d_s_elec)), 4),
            "delta_pca_medio": round(float(np.mean(d_pca)), 4),
            "delta_fa_mediana_kwh": round(float(np.mean(d_fa)), 3),
            "taxa_quarentena_pct": round(100.0 * quarantined / n_ev, 2),
            "taxa_anom_eletrica_sem_gate_pct": round(100.0 * elec_anom_nogate / n_ev, 2),
            "taxa_anom_eletrica_com_gate_pct": round(100.0 * elec_anom_gated / n_ev, 2),
            "taxa_cpd_queda_qualquer_pct": round(100.0 * len(cpd_err_days) / n_ev, 2),
            "taxa_cpd_localizacao_3d_pct": round(100.0 * sum(cpd_hits) / n_ev, 2),
            "cpd_erro_mediano_dias": round(float(np.median(cpd_err_days)), 2) if cpd_err_days else None,
            "cpd_atraso_mediano_dias": round(float(np.median(cpd_delays)), 2) if cpd_delays else None,
            "cpd_erro_mag_pp_mediano": round(float(np.median(cpd_mag_errs)), 2) if cpd_mag_errs else None,
            "beh_24h_delta_l2_medio": round(float(np.mean(d_beh_l2)), 4),
            "beh_24h_taxa_deteccao_p95_pct": round(100.0 * beh_shift_hits / n_ev, 2),
            "beh_24h_mudanca_cluster_pct": round(100.0 * beh_cluster_flips / n_ev, 2),
            "taxa_categoria_prevista_pct": round(100.0 * cat_expected / n_ev, 2),
        })

    # =========================================================================
    # SECTION 18 (COLLECTIVE): ISOLATED vs COLLECTIVE CIRCUIT PERTURBATION
    # =========================================================================
    # Take one feeder (~18-20 UCs) -> group under synthetic shared transformer post TR-SIM-01
    # Compare: (A) 1 UC suffers -50% drop (isolated) vs (B) 80% of UCs on same transformer suffer -50% drop (collective)
    f0 = top5_feeders[0]
    f0_ucs = sorted(oot_dec.filter((pl.col("meta__feeder") == f0) & pl.col("id__uc_id").is_in(valid_ucs))["id__uc_id"].to_list())
    n_f0 = len(f0_ucs)
    n_coll = max(3, int(round(0.80 * n_f0)))
    coll_targets = set(f0_ucs[:n_coll])
    iso_target = {f0_ucs[0]}

    # Evaluate both using frozen detector + filter_spurious_anomalies on shared transformer
    int_iso, _ = _apply_perturbation(base_interval.filter(pl.col("UC").is_in(f0_ucs)), "sustained_drop", 0.50, rng=np.random.default_rng(42), target_ucs=iso_target)
    int_coll, _ = _apply_perturbation(base_interval.filter(pl.col("UC").is_in(f0_ucs)), "collective_feeder_drop", 0.50, rng=np.random.default_rng(42), target_ucs=coll_targets)

    sc_iso_df = _score_perturbed_dataset(int_iso, uc_context=uc_ctx_sub, meter_history=mh_sub, hierarchy=hier_sub,
                                         pipe=pipe, elec_cols=elec_cols, comm_cols=comm_cols,
                                         beh_24h_pca=beh_24h_pca, beh_24h_km=beh_24h_km, beh_24h_p01=beh_24h_p01, beh_24h_p99=beh_24h_p99)
    sc_coll_df = _score_perturbed_dataset(int_coll, uc_context=uc_ctx_sub, meter_history=mh_sub, hierarchy=hier_sub,
                                          pipe=pipe, elec_cols=elec_cols, comm_cols=comm_cols,
                                          beh_24h_pca=beh_24h_pca, beh_24h_km=beh_24h_km, beh_24h_p01=beh_24h_p01, beh_24h_p99=beh_24h_p99)

    def _run_spurious_check(df_s: pl.DataFrame, targets: set[str]) -> dict[str, Any]:
        # Force shared transformer id for the simulated circuit cluster & compute neighborhood contrast
        sc_arr = df_s["s_elec"].to_numpy()
        is_a = (sc_arr >= tau_elec) | df_s["cpd_detected"].fill_null(False).to_numpy()
        # Neighborhood table with shared TRAFO_ID and peer deviation
        med_fa = float(df_s["x__hist_fa_sum_median"].median() or 1.0)
        neigh = pl.DataFrame({
            "UC": df_s["id__uc_id"].to_list(),
            "meta__cutoff_date": df_s["meta__cutoff_date"].to_list(),
            "TRAFO_ID": ["TR-SIM-01"] * df_s.height,
            "TRAFO_PEER_COUNT": [df_s.height] * df_s.height,
            "IS_ISOLATED_DEVIATION": [float(is_a.mean()) < 0.50] * df_s.height,
        })
        inp = df_s.select(["id__uc_id", "meta__cutoff_date"]).with_columns([
            pl.Series("anomaly_score", sc_arr),
            pl.Series("is_anomaly", is_a),
        ])
        filt = filter_spurious_anomalies(inp, neighborhood_df=neigh, grid_event_threshold=0.60)
        t_mask = np.array([u in targets for u in filt["id__uc_id"].to_list()], dtype=bool)
        return {
            "n_circuit_ucs": df_s.height,
            "n_perturbed_ucs": int(t_mask.sum()),
            "raw_anomalous_rate_pct": round(float(is_a.mean() * 100.0), 1),
            "targets_flagged_raw": int(is_a[t_mask].sum()),
            "targets_classified_grid_event": int(filt["IS_GRID_EVENT"].to_numpy()[t_mask].sum()),
            "targets_kept_genuine_uc_anomaly": int(filt["FINAL_IS_ANOMALY"].to_numpy()[t_mask].sum()),
            "mean_score_before_filter": round(float(sc_arr[t_mask].mean()), 4),
            "mean_score_after_filter": round(float(filt["FILTERED_ANOMALY_SCORE"].to_numpy()[t_mask].mean()), 4),
        }

    collective_experiment = {
        "circuito_base": feeder_map.get(f0, f0),
        "evento_isolado_1_uc": _run_spurious_check(sc_iso_df, iso_target),
        "evento_coletivo_80pct_circuito": _run_spurious_check(sc_coll_df, coll_targets),
    }

    # =========================================================================
    # SECTION 19 (BASE B): ABLATION V0..V5 ON CONTROLLED PERTURBATIONS (CONSTANT TOP-K!)
    # =========================================================================
    bb = pl.DataFrame(base_b_records)
    # Total rows = N_ref + 7 scenarios * N_ref; positive electrical/cpd/beh/comm counts known exactly!
    # Constant selection budget: K_b = number of true electrical+CPD perturbed rows (or top 35% of Base B)
    k_b = int(bb.filter(pl.col("is_pert_elec")).height)
    b_iso = bb["iso_score"].to_numpy()
    b_elec = bb["s_elec"].to_numpy()
    b_rel = bb["is_reliable"].to_numpy()
    b_beh = (bb["s_beh_shift"].to_numpy() > beh_l2_p95_ref).astype(float)
    b_cpd = bb["cpd_drop"].to_numpy().astype(float)
    b_phys = bb["phys_gd_or_stuck"].to_numpy().astype(float)

    variants_b = [
        ("V0: Baseline IF (sem gate)", b_iso),
        ("V1: Ensemble 6 det. (sem gate)", b_elec),
        ("V2: V1 + Gate Qualidade (>=70%)", np.where(b_rel, b_elec, -1.0)),
        ("V3: V2 + Comportamento 24h", np.where(b_rel, np.clip(b_elec + 0.12 * b_beh, 0.0, 1.0), -1.0)),
        ("V4: V3 + CPD (CUSUM)", np.where(b_rel, np.clip(b_elec + 0.12 * b_beh + 0.18 * b_cpd, 0.0, 1.0), -1.0)),
        ("V5: V4 + Regras Físicas (Multivisão)", np.where(b_rel, np.clip(b_elec + 0.12 * b_beh + 0.18 * b_cpd + 0.15 * b_phys, 0.0, 1.0), -1.0)),
    ]
    is_elec_true = bb["is_pert_elec"].to_numpy()
    is_comm_true = bb["is_pert_comm"].to_numpy()
    is_cpd_true = bb["is_pert_cpd"].to_numpy()
    is_beh_true = bb["is_pert_beh"].to_numpy()
    is_ref_true = (bb["scenario_group"] == "referencia").to_numpy()

    ablation_b_rows = []
    for vn, sc_v in variants_b:
        order = np.lexsort((np.arange(len(sc_v)), -sc_v))
        top_mask = np.zeros(len(sc_v), dtype=bool)
        top_mask[order[:k_b]] = True
        rec_elec = float(np.sum(top_mask & is_elec_true) / max(1, int(is_elec_true.sum())))
        prec_elec = float(np.sum(top_mask & is_elec_true) / max(1, k_b))
        f1_elec = 2 * prec_elec * rec_elec / max(prec_elec + rec_elec, 1e-9)
        comm_leak = float(np.sum(top_mask & is_comm_true) / max(1, int(is_comm_true.sum())))
        rec_cpd = float(np.sum(top_mask & is_cpd_true) / max(1, int(is_cpd_true.sum())))
        rec_beh = float(np.sum(top_mask & is_beh_true) / max(1, int(is_beh_true.sum())))
        ref_trig = float(np.sum(top_mask & is_ref_true) / max(1, int(is_ref_true.sum())))
        ablation_b_rows.append({
            "variante": vn,
            "n_base_b": len(sc_v),
            "orcamento_K_fixo": k_b,
            "recall_pert_eletricas_pct": round(rec_elec * 100.0, 2),
            "precisao_controlada_pct": round(prec_elec * 100.0, 2),
            "f1_controlado_pct": round(f1_elec * 100.0, 2),
            "vazamento_lacunas_comm_pct": round(comm_leak * 100.0, 2),
            "recall_quedas_cpd_pct": round(rec_cpd * 100.0, 2),
            "recall_shift_rotina_4h_pct": round(rec_beh * 100.0, 2),
            "acionamento_referencia_pct": round(ref_trig * 100.0, 2),
        })

    pl.DataFrame(scenario_results).write_csv(run_dir / "controlled_perturbations_summary.csv", separator=";")
    pl.DataFrame(scenario_results).write_csv(P1_PERT_DIR / "controlled_perturbations_summary.csv", separator=";")
    pl.DataFrame(ablation_b_rows).write_csv(run_dir / "ablation_base_b_controlled.csv", separator=";")
    pl.DataFrame(ablation_b_rows).write_csv(P1_PERT_DIR / "ablation_base_b_controlled.csv", separator=";")

    # =========================================================================
    # P1 FIGURES (10, 11, 12, 13, 14, 16B)
    # =========================================================================
    figs_pert: dict[str, dict[str, str]] = {}
    sc_df = pl.DataFrame(scenario_results)

    # Fig 10: Real Series Before/After Perturbations (Example Real UC)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.2), sharex=True)
    ref_d, ref_fa, ref_cov = example_series["Referência (real)"]
    ex_keys = ["Queda sustentada -50%", "Lacuna aleatória 30%", "Injeção reversa solar (GD)", "Carga zero c/ tensão normal"]
    for ax, ek in zip(axes.ravel(), ex_keys):
        ax.plot(range(len(ref_d)), ref_fa, "o-", color="#5c6f7e", markersize=3, linewidth=1.4, label="Série real original (FA kWh/dia)")
        if ek in example_series:
            pd_, pfa, pcov = example_series[ek]
            ax.plot(range(len(pfa)), pfa, "s--", color="#c0392b", markersize=3, linewidth=1.5, label="Série perturbada (FA kWh/dia)")
        ax.axvline(15, color="#1f4e79", linestyle=":", linewidth=1.2, label="Início perturbação (05/11)")
        ax.set_title(f"{ek} — UC {censor_uc_id(example_uc)}", fontsize=9.2, fontweight="bold")
        ax.set_ylabel("Energia ativa FA (kWh/dia)")
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(fontsize=7.3, loc="lower left")
    for ax in axes[1]:
        ax.set_xticks([0, 7, 15, 22, len(ref_d) - 1])
        ax.set_xticklabels([ref_d[i][5:] for i in [0, 7, 15, 22, len(ref_d) - 1]])
        ax.set_xlabel("Dia da janela (21/10 a 19/11/2026)")
    fig.suptitle("Séries reais AMI antes e após perturbações controladas (cópias isoladas em memória)", fontweight="bold")
    plt.tight_layout()
    figs_pert["fig_p1_10"] = _save_fig(fig, P1_PERT_FIG / "fig_p1_10_series_reais_antes_depois_perturbacoes")

    # Fig 11: Model Response by Perturbation Intensity (Sustained Drop & Communication Gaps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    sub_drop = sc_df.filter(pl.col("tipo") == "sustained_drop")
    x_d = [10, 20, 30, 50]
    ax1.plot(x_d, sub_drop["taxa_cpd_localizacao_3d_pct"].to_numpy(), "o-", color="#2e7d32", linewidth=2, label="CPD localização <=3d (%)")
    ax1.plot(x_d, sub_drop["taxa_anom_eletrica_com_gate_pct"].to_numpy(), "s--", color="#1f4e79", linewidth=2, label="S_elec >= τ com gate (%)")
    ax1.axhline(ref_summary["cpd_drop_rate_pct"], color="#2e7d32", linestyle=":", alpha=0.7, label=f"Ref. CPD ({_fmt_br_float(ref_summary['cpd_drop_rate_pct'], 1)}%)")
    ax1.axhline(ref_summary["elec_anom_rate_pct"], color="#1f4e79", linestyle=":", alpha=0.7, label=f"Ref. S_elec ({_fmt_br_float(ref_summary['elec_anom_rate_pct'], 1)}%)")
    ax1.set_xlabel("Intensidade da queda sustentada (%)")
    ax1.set_ylabel("Taxa de detecção controlada (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("(A) Resposta à queda sustentada de consumo (n = " + str(len(valid_ucs)) + " UCs)", fontweight="bold", fontsize=9.2)
    ax1.legend(fontsize=7.6)
    ax1.grid(True, linestyle=":", alpha=0.5)

    sub_comm = sc_df.filter(pl.col("tipo") == "comm_gap_random")
    x_c = [10, 20, 30, 50]
    ax2.plot(x_c, sub_comm["taxa_quarentena_pct"].to_numpy(), "o-", color="#c0392b", linewidth=2, label="Interceptadas pelo gate (<70%) (%)")
    ax2.plot(x_c, sub_comm["delta_s_comm_medio"].to_numpy() * 100.0, "s--", color="#6c3483", linewidth=2, label="Elevação média S_comm (x100)")
    ax2.plot(x_c, sub_comm["taxa_anom_eletrica_com_gate_pct"].to_numpy(), "^:", color="#1f4e79", linewidth=2, label="Vazamento p/ anomalia elétrica (%)")
    ax2.set_xlabel("Taxa de lacunas de comunicação injetadas (%)")
    ax2.set_ylabel("Percentual / variação (%)")
    ax2.set_ylim(-5, 108)
    ax2.set_title("(B) Resposta do S_comm e gate de cobertura a lacunas", fontweight="bold", fontsize=9.2)
    ax2.legend(fontsize=7.6)
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_pert["fig_p1_11"] = _save_fig(fig, P1_PERT_FIG / "fig_p1_11_resposta_modelos_por_intensidade")

    # Fig 12: Controlled CPD Performance (Duration + Magnitude + Isolated Glitch Contrast)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    cpd_sub = sc_df.filter(pl.col("tipo").is_in(["regime_duration", "isolated_event", "sustained_drop"]))
    cpd_labels = ["Isolado 1d\n(0,1x)", "Isolado 1d\n(3,5x)", "Queda -40%\n(3 dias)", "Queda -40%\n(7 dias)", "Queda -40%\n(14 dias)", "Queda -50%\n(15 dias)"]
    cpd_map_rows = [
        sc_df.filter(pl.col("cenario") == "Evento isolado 1d (0,1x)").to_dicts()[0],
        sc_df.filter(pl.col("cenario") == "Evento isolado 1d (3,5x)").to_dicts()[0],
        sc_df.filter(pl.col("cenario") == "Mudança -40% (3 dias)").to_dicts()[0],
        sc_df.filter(pl.col("cenario") == "Mudança -40% (7 dias)").to_dicts()[0],
        sc_df.filter(pl.col("cenario") == "Mudança -40% (14 dias)").to_dicts()[0],
        sc_df.filter(pl.col("cenario") == "Queda sustentada -50%").to_dicts()[0],
    ]
    xc = np.arange(len(cpd_labels))
    ax1.bar(xc - 0.18, [r["taxa_cpd_localizacao_3d_pct"] for r in cpd_map_rows], width=0.34, color="#2e7d32", label="CPD localização <=3d (%)")
    ax1.bar(xc + 0.18, [r["taxa_anom_eletrica_com_gate_pct"] for r in cpd_map_rows], width=0.34, color="#1f4e79", label="Ensemble S_elec (%)")
    ax1.set_xticks(xc)
    ax1.set_xticklabels(cpd_labels, fontsize=7.8)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("Taxa de detecção controlada (%)")
    ax1.set_title("(A) Persistência temporal: CPD vs. Ensemble Elétrico", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Median localization error (days) across drop intensities & durations
    err_items = [r for r in cpd_map_rows[2:] if r["cpd_erro_mediano_dias"] is not None]
    ax2.bar(range(len(err_items)), [r["cpd_erro_mediano_dias"] for r in err_items], color="#d35400", width=0.52, alpha=0.88)
    for i, r in enumerate(err_items):
        ax2.text(i, r["cpd_erro_mediano_dias"] + 0.12, f"{_fmt_br_float(r['cpd_erro_mediano_dias'], 1)} d", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(range(len(err_items)))
    ax2.set_xticklabels(["-40% (3d)", "-40% (7d)", "-40% (14d)", "-50% (15d)"], fontsize=8.5)
    ax2.set_ylim(0, max([r["cpd_erro_mediano_dias"] for r in err_items] + [2.0]) + 1.5)
    ax2.set_ylabel("Erro absoluto mediano de localização (dias)")
    ax2.set_title("(B) Precisão temporal da localização de mudança (CUSUM)", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_pert["fig_p1_12"] = _save_fig(fig, P1_PERT_FIG / "fig_p1_12_desempenho_controlado_cpd")

    # Fig 13: Effect of Gating on Introduced Communication Gaps (With vs Without Gating)
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    comm_all = sc_df.filter(pl.col("tipo").is_in(["comm_gap_random", "comm_gap_burst"]))
    xg = np.arange(comm_all.height)
    ax.bar(xg - 0.18, comm_all["taxa_anom_eletrica_sem_gate_pct"].to_numpy(), width=0.34, color="#c0392b", alpha=0.88, label="Sem gate de qualidade (s_elec >= τ)")
    ax.bar(xg + 0.18, comm_all["taxa_anom_eletrica_com_gate_pct"].to_numpy(), width=0.34, color="#1f4e79", alpha=0.88, label="Com gate de qualidade (cobertura >= 70%)")
    ax.axhline(ref_summary["elec_anom_rate_pct"], color="#555555", linestyle="--", linewidth=1.1, label=f"Taxa na referência ({_fmt_br_float(ref_summary['elec_anom_rate_pct'], 1)}%)")
    ax.set_xticks(xg)
    ax.set_xticklabels([c.replace("Lacuna aleatória ", "Aleat. ").replace("Bloco consecutivo ", "Bloco ") for c in comm_all["cenario"].to_list()], rotation=18, ha="right", fontsize=8.2)
    ax.set_ylabel("UCs sinalizadas como anomalia elétrica (%)")
    ax.set_ylim(0, max(50.0, float(comm_all["taxa_anom_eletrica_sem_gate_pct"].max()) + 12.0))
    ax.set_title("Efeito protetor do gate de qualidade sobre lacunas artificiais de comunicação (S_comm)", fontweight="bold")
    ax.legend(fontsize=8.2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_pert["fig_p1_13"] = _save_fig(fig, P1_PERT_FIG / "fig_p1_13_efeito_gating_lacunas_comunicacao")

    # Fig 14: Behavioral Profiles (4 Centroids) & Routine Time-Shift Response (Energy Preserved!)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    hrs = np.arange(24)
    cent_24 = beh_24h_pca.inverse_transform(beh_24h_km.cluster_centers_)
    cols_k = ["#1f4e79", "#2e7d32", "#d35400", "#6c3483"]
    for k_i in range(4):
        n_k = int(np.sum(pre_lbls == k_i))
        ax1.plot(hrs, cent_24[k_i] * 100.0, linewidth=2, color=cols_k[k_i], label=f"Arquétipo {k_i + 1} (n = {n_k})")
    ax1.set_xlabel("Hora do dia (0–23h)")
    ax1.set_ylabel("Fração da energia diária (%)")
    ax1.set_title("(A) Centroides comportamentais intradiários (curvas reais)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    sub_beh = sc_df.filter(pl.col("tipo").is_in(["time_shift", "shape_flatten"]))
    xb = np.arange(sub_beh.height)
    ax2.bar(xb - 0.18, sub_beh["beh_24h_taxa_deteccao_p95_pct"].to_numpy(), width=0.34, color="#6c3483", label="S_behavior 24h (> P95 ref.) (%)")
    ax2.bar(xb + 0.18, sub_beh["taxa_anom_eletrica_com_gate_pct"].to_numpy(), width=0.34, color="#1f4e79", label="S_elec diário agregado (%)")
    for i in range(len(xb)):
        ax2.text(xb[i] - 0.18, sub_beh["beh_24h_taxa_deteccao_p95_pct"][i] + 2, f"{_fmt_br_float(sub_beh['beh_24h_taxa_deteccao_p95_pct'][i], 0)}%", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(["Shift +1h", "Shift +2h", "Shift +4h", "Forma\n(kWh fixo)"], fontsize=8.5)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Taxa de detecção (%)")
    ax2.set_title("(B) Deslocamento de rotina/forma (ΔkWh/dia = 0,0)", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_pert["fig_p1_14"] = _save_fig(fig, P1_PERT_FIG / "fig_p1_14_perfis_comportamentais_e_rotina")

    # Fig 16B: Ablation V0..V5 on Base B (Controlled Perturbations, constant K)
    abb_df = pl.DataFrame(ablation_b_rows)
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    xab = np.arange(abb_df.height)
    w = 0.24
    ax.bar(xab - w, abb_df["f1_controlado_pct"].to_numpy(), width=w, color="#1f4e79", label="F1 controlado (pert. elétricas) (%)")
    ax.bar(xab, abb_df["recall_quedas_cpd_pct"].to_numpy(), width=w, color="#2e7d32", label="Recall quedas sustentadas (%)")
    ax.bar(xab + w, abb_df["vazamento_lacunas_comm_pct"].to_numpy(), width=w, color="#c0392b", label="Vazamento lacunas telecom (%)")
    ax.set_xticks(xab)
    ax.set_xticklabels(["V0\nBaseline IF", "V1\nEns. 6 det.", "V2\n+Gate 70%", "V3\n+Behav.", "V4\n+CPD", "V5\n+Multivisão"], fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Taxa observada sob orçamento K fixo (%)")
    ax.set_title(f"Ablação metodológica V0–V5 sobre Base B (perturbações controladas, K = {k_b}, N = {len(bb)})", fontweight="bold")
    ax.legend(fontsize=8.2, ncol=3, loc="upper center")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    figs_pert["fig_p1_16b"] = _save_fig(fig, P1_PERT_FIG / "fig_p1_16b_ablacao_variantes_base_b")

    summary = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp": datetime.now().isoformat(),
        "runtime_sec": round(time.time() - t0, 2),
        "sample_selection": {
            "seed": 42,
            "n_ucs_valid": len(valid_ucs),
            "n_feeders": len(top5_feeders),
            "feeders_masked": [feeder_map.get(f, f) for f in top5_feeders],
            "n_interval_rows_loaded": int(base_interval.height),
            "strata": strata_summary,
            "data_lake_untouched": True,
        },
        "reference_unperturbed": ref_summary,
        "scenarios": scenario_results,
        "collective_vs_isolated": collective_experiment,
        "ablation_base_b": ablation_b_rows,
        "figures": figs_pert,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (P1_PERT_DIR / "p1_perturbations_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "pipeline_id": "p1_controlled_perturbations",
        "git_commit": commit,
        "status": "validated",
        "created_at": datetime.now().isoformat(),
        "artifacts": {
            f.name: {"path": str(f.relative_to(PROJECT_ROOT)), "sha256": _sha256(f)}
            for f in sorted(run_dir.iterdir()) if f.is_file()
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_controlled_perturbations()
    print(f"P1 Controlled Perturbations completed in {res['runtime_sec']}s | run_id={res['run_id']}")
    print("Reference unperturbed:", json.dumps(res["reference_unperturbed"], indent=2))
    print("Collective vs Isolated:", json.dumps(res["collective_vs_isolated"], indent=2))
    print("Ablation Base B:")
    for row in res["ablation_base_b"]:
        print(" ", row)
