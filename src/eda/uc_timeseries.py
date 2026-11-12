"""
src/eda/uc_timeseries.py
-------------------------------------------------------------------------------
Extração sob demanda de séries temporais REAIS do Data Lake (ami_interval e
uc_day_features) e decomposição estatística dos sinais de anomalia por UC:
  - Gating de cobertura (INTERVAL_COVERAGE / intervalos válidos na hora)
  - Baseline robusta (Mediana + MAD + escala robusta 1.4826*MAD)
  - Detecção de Picos isolados (Modified Z-Score |z_self| > 3.5)
  - Detecção de Queda / Degrau sustentado (Slope OLS + CUSUM change-point)
  - Busca automática de pares contrastantes e UCs com anomalias reais
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.utils.censorship import censor_uc_id

_DATE_DIR_RE = re.compile(
    r"report_year=(\d{4})[/\\]report_month=(\d{2})(?:[/\\]report_day=(\d{2}))?"
)


def parse_partition_date(path: Path) -> date | None:
    """Extract YYYY-MM-DD (or YYYY-MM-01 for monthly partitions) from Hive-style path."""
    m = _DATE_DIR_RE.search(str(path))
    if not m:
        return None
    try:
        day = int(m.group(3)) if m.group(3) else 1
        return date(int(m.group(1)), int(m.group(2)), day)
    except ValueError:
        return None


def _list_dated_parquets(root_dir: Path) -> list[tuple[date, Path]]:
    """Return sorted list of (report_date, parquet_path) under a partition root."""
    if not root_dir.exists():
        return []
    items: list[tuple[date, Path]] = []
    for p in root_dir.rglob("*.parquet"):
        d = parse_partition_date(p)
        if d is not None:
            items.append((d, p))
    items.sort(key=lambda x: x[0])
    return items


def _normalize_id_variants(val: str) -> list[str]:
    """Return raw, trimmed and stripped-zero variants for robust UC/NIO matching."""
    s = str(val).strip()
    if not s:
        return []
    variants = {s, s.lstrip("0") or "0"}
    if s.isdigit():
        variants.add(s.zfill(9))
        variants.add(s.zfill(10))
    return sorted(variants)


def resolve_uc_to_nios(
    uc_id: str,
    base_dir: Path | str = "output",
) -> list[str]:
    """Find all NIO meter IDs associated with a UC via context & day features."""
    base = Path(base_dir)
    uc_variants = _normalize_id_variants(uc_id)
    if not uc_variants:
        return []

    found_nios: set[str] = set()

    # 1. Context installation history (Parquet or CSV)
    inst_pq = base / "context" / "meter_installation_history.parquet"
    inst_csv = base / "context" / "meter_installation_history.csv"
    try:
        if inst_pq.exists():
            lf = pl.scan_parquet(str(inst_pq))
            cols = set(lf.collect_schema().names())
            uc_col = "UC" if "UC" in cols else ("id__uc_id" if "id__uc_id" in cols else None)
            nio_col = "NIO" if "NIO" in cols else ("id__nio" if "id__nio" in cols else None)
            if uc_col and nio_col:
                df_n = (
                    lf.with_columns(pl.col(uc_col).cast(pl.String).str.strip_chars().alias("_uc"))
                    .filter(pl.col("_uc").is_in(uc_variants))
                    .select(pl.col(nio_col).cast(pl.String).str.strip_chars())
                    .unique()
                    .collect()
                )
                for row in df_n.iter_rows():
                    if row[0]:
                        found_nios.update(_normalize_id_variants(row[0]))
        elif inst_csv.exists():
            df_n = (
                pl.read_csv(str(inst_csv), separator=";", infer_schema_length=2000)
                .filter(pl.col("UC").cast(pl.String).str.strip_chars().is_in(uc_variants))
            )
            if "NIO" in df_n.columns:
                for v in df_n["NIO"].cast(pl.String).drop_nulls().to_list():
                    found_nios.update(_normalize_id_variants(v))
    except Exception:
        pass

    # 2. Check uc_day_features partitions if still empty (or supplement)
    if not found_nios:
        day_files = _list_dated_parquets(base / "features" / "uc_day_features")
        for _, p in reversed(day_files[-35:]):
            try:
                lf = pl.scan_parquet(str(p))
                cols = set(lf.collect_schema().names())
                if "UC" in cols and "NIO" in cols:
                    res = (
                        lf.with_columns(pl.col("UC").cast(pl.String).str.strip_chars().alias("_uc"))
                        .filter(pl.col("_uc").is_in(uc_variants))
                        .select(pl.col("NIO").cast(pl.String).str.strip_chars())
                        .unique()
                        .collect()
                    )
                    for row in res.iter_rows():
                        if row[0]:
                            found_nios.update(_normalize_id_variants(row[0]))
                    if found_nios:
                        break
            except Exception:
                continue

    return sorted(found_nios)


def _select_window_files(
    dated_files: list[tuple[date, Path]],
    cutoff_date: date | None = None,
    days: int = 30,
) -> list[Path]:
    """Filter dated parquet files to [end - days + 1, end]."""
    if not dated_files:
        return []
    eligible = [item for item in dated_files if cutoff_date is None or item[0] <= cutoff_date]
    if not eligible:
        eligible = dated_files
    end_dt = eligible[-1][0]
    raw_start = end_dt - timedelta(days=max(1, days) - 1)
    start_dt = raw_start.replace(day=1) if all(d.day == 1 for d, _ in eligible[-5:]) else raw_start
    selected = [p for d, p in eligible if start_dt <= d <= end_dt]
    return selected if selected else [p for _, p in eligible[-days:]]


def _detect_step_change_cusum(
    dates: list[str],
    values: list[float],
    coverages: list[float],
    baseline_med: float,
    robust_scale: float,
) -> dict[str, Any] | None:
    """Detect the strongest sustained step-down (or step-up) change-point via CUSUM & mean shift."""
    reliable_idx = [
        i for i, (v, c) in enumerate(zip(values, coverages))
        if v is not None and not math.isnan(v) and c >= 0.70
    ]
    if len(reliable_idx) < 8:
        return None

    y = np.array([values[i] for i in reliable_idx], dtype=np.float64)
    mean_y = float(np.mean(y))
    centered = y - mean_y
    cusum = np.cumsum(centered)
    # Locate split point with maximum absolute excursion (excluding boundary edges)
    margin = max(2, len(y) // 6)
    if len(y) <= 2 * margin:
        return None
    sub_cusum = np.abs(cusum[margin:-margin])
    k = int(np.argmax(sub_cusum)) + margin

    pre_med = float(np.median(y[: k + 1]))
    post_med = float(np.median(y[k + 1 :]))
    delta_abs = post_med - pre_med
    delta_pct = delta_abs / max(abs(pre_med), 0.05)

    # Confirm statistical magnitude relative to robust scale
    if abs(delta_abs) < max(1.25 * robust_scale, 0.25 * max(abs(pre_med), 0.1)):
        return None

    orig_idx = reliable_idx[k + 1]
    return {
        "index": int(orig_idx),
        "date": dates[orig_idx],
        "pre_median": round(pre_med, 4),
        "post_median": round(post_med, 4),
        "delta_abs": round(delta_abs, 4),
        "delta_pct": round(delta_pct * 100.0, 2),
        "cusum_peak": round(float(np.abs(cusum[k])), 4),
        "direction": "QUEDA_DEGRAU" if delta_abs < 0 else "AUMENTO_DEGRAU",
    }


def _compute_ols_slope(values: list[float], coverages: list[float]) -> float | None:
    """Compute OLS linear trend slope (kWh/day) over reliable days."""
    pts = [
        (i, v)
        for i, (v, c) in enumerate(zip(values, coverages))
        if v is not None and not math.isnan(v) and c >= 0.70
    ]
    if len(pts) < 4:
        return None
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    x_mean = xs.mean()
    denom = float(np.sum((xs - x_mean) ** 2))
    if denom <= 1e-12:
        return 0.0
    slope = float(np.sum((xs - x_mean) * (ys - ys.mean())) / denom)
    return round(slope, 4)


def load_uc_daily_timeseries(
    uc_id: str,
    base_dir: Path | str = "output",
    cutoff_date: str | date | None = None,
    days: int = 30,
) -> dict[str, Any]:
    """Query real daily telemetry for a UC from uc_day_features and annotate anomalies."""
    base = Path(base_dir)
    parsed_cutoff: date | None = None
    if isinstance(cutoff_date, date):
        parsed_cutoff = cutoff_date
    elif isinstance(cutoff_date, str) and cutoff_date.strip():
        try:
            parsed_cutoff = date.fromisoformat(cutoff_date.strip()[:10])
        except ValueError:
            parsed_cutoff = None

    day_files = _list_dated_parquets(base / "features" / "uc_day_features")
    selected_files = _select_window_files(day_files, cutoff_date=parsed_cutoff, days=days)
    if not selected_files:
        return {
            "status": "no_partitions",
            "uc": str(uc_id),
            "uc_censored": censor_uc_id(uc_id),
            "granularity": "daily",
            "points": [],
            "anomalies": [],
            "message": "Nenhuma partição uc_day_features encontrada no Data Lake local.",
        }

    uc_variants = _normalize_id_variants(uc_id)
    frames: list[pl.DataFrame] = []
    wanted_cols = [
        "UC", "NIO", "REPORT_DAY", "INTERVAL_COVERAGE",
        "FA_INTERVAL_SUM", "RA_INTERVAL_SUM", "LOAD_FACTOR",
        "CURRENT_IMBALANCE_MAX", "VOLTAGE_IMBALANCE_MAX",
        "FA_INTERVAL_MEDIAN", "FA_INTERVAL_IQR", "FA_RAMP_MAX",
        "U_L1_MEDIAN",
    ]
    for fpath in selected_files:
        try:
            lf = pl.scan_parquet(str(fpath))
            avail = set(lf.collect_schema().names())
            if "UC" not in avail:
                continue
            cols_to_read = [c for c in wanted_cols if c in avail]
            df_part = (
                lf.select(cols_to_read)
                .with_columns(pl.col("UC").cast(pl.String).str.strip_chars().alias("_uc_clean"))
                .filter(pl.col("_uc_clean").is_in(uc_variants))
                .drop("_uc_clean")
                .collect()
            )
            if df_part.height > 0:
                frames.append(df_part)
        except Exception:
            continue

    if not frames:
        return {
            "status": "not_found",
            "uc": str(uc_id),
            "uc_censored": censor_uc_id(uc_id),
            "granularity": "daily",
            "points": [],
            "anomalies": [],
            "message": f"Sem registros diários para a UC {censor_uc_id(uc_id)} na janela selecionada.",
        }

    df = pl.concat(frames, how="diagonal_relaxed")
    if "REPORT_DAY" in df.columns:
        df = df.sort("REPORT_DAY").unique(subset=["REPORT_DAY"], keep="last").sort("REPORT_DAY")

    # Convert columns to Python lists safely
    dates = [str(d)[:10] for d in df["REPORT_DAY"].to_list()]
    fa_vals = [float(v) if v is not None else 0.0 for v in df["FA_INTERVAL_SUM"].to_list()]
    cov_vals = [
        float(v) if v is not None else 0.0
        for v in (df["INTERVAL_COVERAGE"].to_list() if "INTERVAL_COVERAGE" in df.columns else [1.0] * len(dates))
    ]
    ra_vals = [
        float(v) if v is not None else 0.0
        for v in (df["RA_INTERVAL_SUM"].to_list() if "RA_INTERVAL_SUM" in df.columns else [0.0] * len(dates))
    ]
    ci_vals = [
        float(v) if v is not None else 0.0
        for v in (df["CURRENT_IMBALANCE_MAX"].to_list() if "CURRENT_IMBALANCE_MAX" in df.columns else [0.0] * len(dates))
    ]
    vi_vals = [
        float(v) if v is not None else 0.0
        for v in (df["VOLTAGE_IMBALANCE_MAX"].to_list() if "VOLTAGE_IMBALANCE_MAX" in df.columns else [0.0] * len(dates))
    ]
    lf_vals = [
        float(v) if v is not None else 0.0
        for v in (df["LOAD_FACTOR"].to_list() if "LOAD_FACTOR" in df.columns else [0.0] * len(dates))
    ]

    reliable_y = [v for v, c in zip(fa_vals, cov_vals) if c >= 0.70]
    if len(reliable_y) < 3:
        reliable_y = list(fa_vals)

    y_arr = np.array(reliable_y, dtype=np.float64) if reliable_y else np.array([0.0])
    median_val = float(np.median(y_arr))
    mad_val = float(np.median(np.abs(y_arr - median_val)))
    scale = max(1.4826 * mad_val, 0.10 * abs(median_val), 0.05)

    upper_band = round(median_val + 3.5 * scale, 4)
    lower_band = round(max(0.0, median_val - 3.5 * scale), 4)
    step_info = _detect_step_change_cusum(dates, fa_vals, cov_vals, median_val, scale)
    slope_val = _compute_ols_slope(fa_vals, cov_vals)

    points: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    step_idx = step_info["index"] if step_info and step_info["direction"] == "QUEDA_DEGRAU" else None

    for idx, dt_str in enumerate(dates):
        val = fa_vals[idx]
        cov = cov_vals[idx]
        z_self = (val - median_val) / scale if scale > 0 else 0.0
        is_low_cov = cov < 0.70

        flag = "NORMAL"
        if is_low_cov:
            flag = "GAP_TELECOM"
        elif z_self > 3.5:
            flag = "PICO_OUTLIER"
            anomalies.append({
                "index": idx,
                "label": dt_str,
                "type": "PICO_OUTLIER",
                "value": round(val, 4),
                "z_score": round(z_self, 2),
                "reason": f"Pico de consumo (+{z_self:.1f} sigma MAD acima da mediana)",
            })
        elif z_self < -3.5:
            flag = "QUEDA_OUTLIER"
            anomalies.append({
                "index": idx,
                "label": dt_str,
                "type": "QUEDA_OUTLIER",
                "value": round(val, 4),
                "z_score": round(z_self, 2),
                "reason": f"Queda abrupta ({z_self:.1f} sigma MAD abaixo da mediana)",
            })
        elif step_idx is not None and idx >= step_idx and val <= 0.65 * max(median_val, 0.05):
            flag = "DEGRAU_QUEDA"

        if not is_low_cov and ra_vals[idx] > 0.5 and val > 0 and (ra_vals[idx] / max(val, 1e-6)) > 0.2:
            if flag == "NORMAL":
                flag = "FLUXO_REVERSO"
            anomalies.append({
                "index": idx,
                "label": dt_str,
                "type": "FLUXO_REVERSO",
                "value": round(val, 4),
                "z_score": round(z_self, 2),
                "reason": f"Injeção reversa relevante (RA={ra_vals[idx]:.2f} kWh)",
            })

        points.append({
            "label": dt_str,
            "value": round(val, 4),
            "ra": round(ra_vals[idx], 4),
            "coverage": round(cov, 4),
            "z_self": round(z_self, 2),
            "current_imbalance": round(ci_vals[idx], 4),
            "voltage_imbalance": round(vi_vals[idx], 4),
            "load_factor": round(lf_vals[idx], 4),
            "flag": flag,
        })

    if step_info is not None:
        s_idx = step_info["index"]
        anomalies.append({
            "index": s_idx,
            "label": step_info["date"],
            "type": step_info["direction"],
            "value": round(fa_vals[s_idx], 4),
            "z_score": round((fa_vals[s_idx] - median_val) / scale, 2),
            "reason": (
                f"Degrau CUSUM ({step_info['delta_pct']:+.1f}%: "
                f"mediana {step_info['pre_median']:.2f} -> {step_info['post_median']:.2f} kWh/dia)"
            ),
        })

    # Compute consecutive anomalous days streaks
    max_streak = 0
    cur_streak = 0
    for p in points:
        is_anom_day = p["flag"] in ("PICO_OUTLIER", "QUEDA_OUTLIER", "DEGRAU_QUEDA", "FLUXO_REVERSO")
        if is_anom_day:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            cur_streak = 0
    active_streak_at_end = cur_streak

    nio_list = (
        df["NIO"].cast(pl.String).drop_nulls().unique().to_list()
        if "NIO" in df.columns
        else []
    )

    return {
        "status": "ok",
        "uc": str(uc_id),
        "uc_censored": censor_uc_id(uc_id),
        "nios": nio_list,
        "granularity": "daily",
        "window_days": len(points),
        "reliable_days": int(sum(1 for c in cov_vals if c >= 0.70)),
        "baseline_median": round(median_val, 4),
        "baseline_mad": round(mad_val, 4),
        "robust_scale": round(scale, 4),
        "upper_band": upper_band,
        "lower_band": lower_band,
        "slope_kwh_day": slope_val,
        "step_change": step_info,
        "dias_anomalia_seguidos": max_streak,
        "active_streak_days": active_streak_at_end,
        "points": points,
        "anomalies": anomalies,
    }


def load_uc_hourly_timeseries(
    uc_id: str,
    base_dir: Path | str = "output",
    cutoff_date: str | date | None = None,
    week_days: int = 7,
    baseline_days: int = 21,
) -> dict[str, Any]:
    """
    Build real hourly (e.g. 168h = 7 days) series from raw 5-min ami_interval,
    anchoring on the week with the strongest anomaly (or latest week up to cutoff)
    and computing per-hour-of-day robust MAD baselines across baseline_days.
    """
    base = Path(base_dir)
    daily_res = load_uc_daily_timeseries(
        uc_id, base_dir=base, cutoff_date=cutoff_date, days=max(baseline_days, week_days, 30)
    )

    parsed_cutoff: date | None = None
    if isinstance(cutoff_date, date):
        parsed_cutoff = cutoff_date
    elif isinstance(cutoff_date, str) and cutoff_date.strip():
        try:
            parsed_cutoff = date.fromisoformat(cutoff_date.strip()[:10])
        except ValueError:
            parsed_cutoff = None

    # Choose best 7-day anchor date: centered around strongest daily anomaly/step if present
    anchor_end: date | None = parsed_cutoff
    if daily_res.get("status") == "ok" and daily_res["points"]:
        pts = daily_res["points"]
        all_dates = [date.fromisoformat(p["label"]) for p in pts]
        last_dt = all_dates[-1]
        best_anom_dt: date | None = None
        anoms = daily_res.get("anomalies", [])
        # Prefer showing both step change and spike in the same 7-day window if possible
        step_dt = None
        spike_dts: list[date] = []
        for a in anoms:
            dt = date.fromisoformat(a["label"])
            if a["type"] in ("QUEDA_DEGRAU", "DEGRAU_QUEDA"):
                step_dt = dt
            elif "PICO" in a["type"]:
                spike_dts.append(dt)
        if step_dt and spike_dts:
            # Find midpoint window covering step & closest spike
            nearest_spike = min(spike_dts, key=lambda d: abs((d - step_dt).days))
            if abs((nearest_spike - step_dt).days) <= week_days - 1:
                best_anom_dt = min(last_dt, max(step_dt, nearest_spike) + timedelta(days=1))
            else:
                best_anom_dt = min(last_dt, nearest_spike + timedelta(days=2))
        elif spike_dts:
            best_anom_dt = min(last_dt, spike_dts[-1] + timedelta(days=2))
        elif step_dt:
            best_anom_dt = min(last_dt, step_dt + timedelta(days=3))
        anchor_end = best_anom_dt or last_dt

    interval_files = _list_dated_parquets(base / "measurements" / "ami_interval")
    if not interval_files:
        daily_res["fallback_reason"] = "ami_interval_unavailable"
        return daily_res

    nios = set(daily_res.get("nios", []))
    nios.update(resolve_uc_to_nios(uc_id, base_dir=base))
    nio_variants: list[str] = []
    for n in nios:
        nio_variants.extend(_normalize_id_variants(n))
    nio_variants = sorted(set(nio_variants))

    if not nio_variants:
        daily_res["fallback_reason"] = "nio_not_resolved"
        return daily_res

    # Load baseline_days up to anchor_end (or latest available in ami_interval)
    window_files = _select_window_files(
        interval_files, cutoff_date=anchor_end, days=max(baseline_days, week_days)
    )
    if not window_files:
        daily_res["fallback_reason"] = "no_interval_files_in_window"
        return daily_res

    frames: list[pl.DataFrame] = []
    for fpath in window_files:
        try:
            lf = pl.scan_parquet(str(fpath))
            avail = set(lf.collect_schema().names())
            if "NIO" not in avail or "FA_INTERVAL" not in avail:
                continue
            ts_col = "TIMESTAMP_LOCAL" if "TIMESTAMP_LOCAL" in avail else "TIMESTAMP_UTC"
            if ts_col not in avail:
                continue
            sel = ["NIO", ts_col, "FA_INTERVAL"]
            if "RA_INTERVAL" in avail:
                sel.append("RA_INTERVAL")
            df_p = (
                lf.select(sel)
                .with_columns(pl.col("NIO").cast(pl.String).str.strip_chars().alias("_nio"))
                .filter(pl.col("_nio").is_in(nio_variants))
                .rename({ts_col: "TS"})
                .drop("_nio")
                .collect()
            )
            if df_p.height > 0:
                frames.append(df_p)
        except Exception:
            continue

    if not frames:
        daily_res["fallback_reason"] = "no_interval_rows_for_uc"
        return daily_res

    raw_df = pl.concat(frames, how="diagonal_relaxed")
    if "RA_INTERVAL" not in raw_df.columns:
        raw_df = raw_df.with_columns(pl.lit(0.0).alias("RA_INTERVAL"))

    # Parse timestamp if string, strip any timezone safely, deduplicate, and truncate to 1h
    if raw_df.schema["TS"] == pl.String:
        raw_df = raw_df.with_columns(
            pl.col("TS").str.to_datetime(strict=False).alias("TS")
        ).drop_nulls(subset=["TS"])

    if raw_df.height == 0:
        daily_res["fallback_reason"] = "empty_timestamps"
        return daily_res

    raw_df = raw_df.with_columns(
        pl.col("TS").dt.replace_time_zone(None).alias("TS")
    ).unique(subset=["TS"], keep="last")

    hourly_df = (
        raw_df.with_columns(
            pl.col("TS").dt.truncate("1h").alias("HOUR_TS"),
            pl.col("FA_INTERVAL").cast(pl.Float64).fill_null(0.0),
            pl.col("RA_INTERVAL").cast(pl.Float64).fill_null(0.0),
        )
        .group_by("HOUR_TS")
        .agg([
            pl.col("FA_INTERVAL").sum().alias("FA_KWH"),
            pl.col("RA_INTERVAL").sum().alias("RA_KWH"),
            pl.len().alias("INTERVAL_COUNT"),
        ])
        .sort("HOUR_TS")
        .with_columns(pl.col("HOUR_TS").dt.hour().alias("HOD"))
    )

    # Reindex to continuous 1-hour grid for the final week_days (168h)
    max_ts = hourly_df["HOUR_TS"].max()
    if anchor_end is not None:
        target_end_ts = min(
            max_ts,
            datetime(anchor_end.year, anchor_end.month, anchor_end.day, 23, 0, 0),
        )
    else:
        target_end_ts = max_ts
    target_start_ts = target_end_ts - timedelta(hours=week_days * 24 - 1)

    # Hour-of-day baseline from ALL loaded reliable hours
    reliable_hourly = hourly_df.filter(pl.col("INTERVAL_COUNT") >= 8)
    if reliable_hourly.height < 24:
        reliable_hourly = hourly_df

    all_rel_vals = reliable_hourly["FA_KWH"].to_numpy()
    global_med = float(np.median(all_rel_vals)) if len(all_rel_vals) else 0.0
    global_mad = float(np.median(np.abs(all_rel_vals - global_med))) if len(all_rel_vals) else 0.0
    global_scale = max(1.4826 * global_mad, 0.10 * abs(global_med), 0.01)

    hod_stats: dict[int, tuple[float, float]] = {}
    for h in range(24):
        sub_h = reliable_hourly.filter(pl.col("HOD") == h)["FA_KWH"].to_numpy()
        if len(sub_h) >= 3:
            m_h = float(np.median(sub_h))
            mad_h = float(np.median(np.abs(sub_h - m_h)))
            sc_h = max(1.4826 * mad_h, 0.12 * abs(m_h), global_scale * 0.5, 0.01)
            hod_stats[h] = (m_h, sc_h)
        else:
            hod_stats[h] = (global_med, global_scale)

    # Build lookup of target week hours
    week_lookup: dict[datetime, tuple[float, float, int]] = {}
    for row in hourly_df.filter(
        (pl.col("HOUR_TS") >= target_start_ts) & (pl.col("HOUR_TS") <= target_end_ts)
    ).iter_rows(named=True):
        week_lookup[row["HOUR_TS"]] = (
            float(row["FA_KWH"]),
            float(row["RA_KWH"]),
            int(row["INTERVAL_COUNT"]),
        )

    total_hours = week_days * 24
    points: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    hour_vals: list[float] = []
    hour_covs: list[float] = []
    hour_labels: list[str] = []

    for i in range(total_hours):
        cur_ts = target_start_ts + timedelta(hours=i)
        lbl = cur_ts.strftime("%d/%m %Hh")
        hod = cur_ts.hour
        m_h, sc_h = hod_stats[hod]

        if cur_ts in week_lookup:
            fa_h, ra_h, cnt_h = week_lookup[cur_ts]
            cov_h = min(1.0, cnt_h / 12.0)
        else:
            fa_h, ra_h, cnt_h, cov_h = 0.0, 0.0, 0, 0.0

        hour_vals.append(fa_h)
        hour_covs.append(1.0 if cnt_h >= 8 else 0.0)
        hour_labels.append(cur_ts.strftime("%Y-%m-%d %H:%M"))

        z_h = (fa_h - m_h) / sc_h if sc_h > 0 else 0.0
        upper_h = m_h + 3.5 * sc_h
        lower_h = max(0.0, m_h - 3.0 * sc_h)

        flag = "NORMAL"
        if cnt_h < 8:
            flag = "GAP_TELECOM"
        elif z_h > 3.5:
            flag = "PICO_OUTLIER"
            anomalies.append({
                "index": i,
                "label": lbl,
                "type": "PICO_OUTLIER",
                "value": round(fa_h, 4),
                "z_score": round(z_h, 2),
                "reason": f"Pico horário ({fa_h:.2f} kWh vs mediana {m_h:.2f} kWh p/ {hod:02d}h; z={z_h:+.1f})",
            })
        elif z_h < -3.0 and m_h > 0.05:
            flag = "QUEDA_OUTLIER"

        points.append({
            "label": lbl,
            "timestamp": cur_ts.strftime("%Y-%m-%d %H:%M"),
            "value": round(fa_h, 4),
            "ra": round(ra_h, 4),
            "baseline_hod": round(m_h, 4),
            "upper_hod": round(upper_h, 4),
            "lower_hod": round(lower_h, 4),
            "coverage": round(cov_h, 3),
            "interval_count": cnt_h,
            "z_self": round(z_h, 2),
            "flag": flag,
        })

    step_hourly = _detect_step_change_cusum(
        hour_labels, hour_vals, hour_covs, global_med, global_scale
    )
    if step_hourly is not None:
        s_idx = step_hourly["index"]
        anomalies.append({
            "index": s_idx,
            "label": points[s_idx]["label"],
            "type": step_hourly["direction"],
            "value": round(hour_vals[s_idx], 4),
            "z_score": round(points[s_idx]["z_self"], 2),
            "reason": (
                f"Mudança de patamar ({step_hourly['delta_pct']:+.1f}%: "
                f"{step_hourly['pre_median']:.2f} -> {step_hourly['post_median']:.2f} kWh/h)"
            ),
        })

    # Compute 24h profile summary (median, P10, P90 across loaded days)
    profile_24h: list[dict[str, Any]] = []
    for h in range(24):
        sub_h = reliable_hourly.filter(pl.col("HOD") == h)["FA_KWH"].to_numpy()
        if len(sub_h) > 0:
            profile_24h.append({
                "hour": h,
                "median": round(float(np.median(sub_h)), 4),
                "p10": round(float(np.percentile(sub_h, 10)), 4),
                "p90": round(float(np.percentile(sub_h, 90)), 4),
                "n_days": int(len(sub_h)),
            })
        else:
            profile_24h.append({"hour": h, "median": 0.0, "p10": 0.0, "p90": 0.0, "n_days": 0})

    return {
        "status": "ok",
        "uc": str(uc_id),
        "uc_censored": censor_uc_id(uc_id),
        "nios": sorted(nios),
        "granularity": "hourly",
        "window_days": week_days,
        "window_start": target_start_ts.strftime("%Y-%m-%d %H:%M"),
        "window_end": target_end_ts.strftime("%Y-%m-%d %H:%M"),
        "baseline_median": round(global_med, 4),
        "baseline_mad": round(global_mad, 4),
        "robust_scale": round(global_scale, 4),
        "upper_band": round(global_med + 3.5 * global_scale, 4),
        "lower_band": round(max(0.0, global_med - 3.5 * global_scale), 4),
        "slope_kwh_day": daily_res.get("slope_kwh_day"),
        "step_change": step_hourly or daily_res.get("step_change"),
        "dias_anomalia_seguidos": daily_res.get("dias_anomalia_seguidos", 0),
        "active_streak_days": daily_res.get("active_streak_days", 0),
        "profile_24h": profile_24h,
        "points": points,
        "anomalies": anomalies,
        "daily_fallback": daily_res if daily_res.get("status") == "ok" else None,
    }


def load_uc_24h_profile(
    uc_id: str,
    base_dir: Path | str = "output",
    days: int = 30,
) -> dict[str, Any]:
    """Return real 24-hour median/P10/P90 load curve for a UC from ami_interval."""
    res = load_uc_hourly_timeseries(uc_id, base_dir=base_dir, week_days=7, baseline_days=days)
    if res.get("granularity") == "hourly" and res.get("profile_24h"):
        return {
            "status": "ok",
            "uc": str(uc_id),
            "uc_censored": censor_uc_id(uc_id),
            "profile_24h": res["profile_24h"],
            "baseline_median": res["baseline_median"],
            "baseline_mad": res["baseline_mad"],
        }
    return {
        "status": "unavailable",
        "uc": str(uc_id),
        "uc_censored": censor_uc_id(uc_id),
        "profile_24h": [],
    }


def resolve_col(df: pl.DataFrame, candidates: list[str]) -> str | None:
    """Resolve first matching column name across raw and prefixed feature schemas."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def plot_uc_interactive(
    uc_id: str,
    base_dir: Path | str = "output",
    cutoff_date: str | date | None = None,
    granularity: str = "hourly",
    days: int = 7,
    figsize: tuple[float, float] = (13.5, 6.5),
    dpi: int = 120,
    show_censored: bool = True,
) -> Any:
    """
    Render dual interactive on-demand diagnostic plots for a single UC directly from real Data Lake:
      - Upper Panel: Real consumption curve, MAD operational baseline, upper/lower ±3.5σ bands,
        CUSUM step-down shading, peak/drop anomaly markers, and telecom gaps.
      - Lower Panel: Real telecom completeness / coverage diagnosis (% of 5-min intervals received),
        color-coded (green/yellow/red) against the 70% reliability threshold.

    Returns:
        matplotlib.figure.Figure: Ready for display in Marimo (mo.as_html(fig)), Jupyter, or export.
    """
    import matplotlib.pyplot as plt

    if granularity == "hourly":
        ts = load_uc_hourly_timeseries(
            uc_id, base_dir=base_dir, cutoff_date=cutoff_date, week_days=days
        )
        if ts.get("status") != "ok" or not ts.get("points"):
            ts = load_uc_daily_timeseries(
                uc_id, base_dir=base_dir, cutoff_date=cutoff_date, days=max(days, 30)
            )
    else:
        ts = load_uc_daily_timeseries(
            uc_id, base_dir=base_dir, cutoff_date=cutoff_date, days=days
        )

    fig, (ax_top, ax_cov) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True, gridspec_kw={"height_ratios": [3.2, 1.1]}
    )

    uc_disp = censor_uc_id(uc_id) if show_censored else str(uc_id)
    if ts.get("status") != "ok" or not ts.get("points"):
        ax_top.text(
            0.5, 0.5,
            f"Sem telemetria bruta disponível no Data Lake para UC {uc_disp}.\n({ts.get('message', 'Nenhum ponto')})",
            ha="center", va="center", color="#ef4444", fontsize=12, fontweight="bold"
        )
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_cov.set_visible(False)
        plt.tight_layout()
        return fig

    pts = ts["points"]
    xs = np.arange(len(pts))
    vals = np.array([p["value"] for p in pts], dtype=np.float64)
    labels = [p["label"] for p in pts]
    flags = [p["flag"] for p in pts]
    gran = ts.get("granularity", "daily")
    unit = "kWh/h" if gran == "hourly" else "kWh/dia"

    # Top panel: Band and Baseline
    if gran == "hourly" and "baseline_hod" in pts[0]:
        base_line = np.array([p["baseline_hod"] for p in pts], dtype=np.float64)
        upper_line = np.array([p["upper_hod"] for p in pts], dtype=np.float64)
        lower_line = np.array([p["lower_hod"] for p in pts], dtype=np.float64)
        ax_top.fill_between(xs, lower_line, upper_line, color="#0284c7", alpha=0.14, label="Banda Operacional MAD (±3.5σ por hora)")
        ax_top.plot(xs, base_line, color="#0369a1", ls="--", lw=1.3, label="Baseline Mediana Histórica (Hora do Dia)")
    else:
        med = float(ts.get("baseline_median", 0.0))
        up = float(ts.get("upper_band", med))
        lo = float(ts.get("lower_band", 0.0))
        ax_top.axhspan(lo, up, color="#0284c7", alpha=0.12, label="Banda Operacional MAD (±3.5σ)")
        ax_top.axhline(med, color="#0369a1", ls="--", lw=1.4, label=f"Baseline Mediana ({med:.2f} {unit})")

    ax_top.plot(xs, vals, color="#0f172a", lw=1.8, label="Consumo Real Medido (FA)")

    # Gaps (< 70% coverage)
    gap_labeled = False
    for i, f in enumerate(flags):
        if f == "GAP_TELECOM":
            ax_top.axvspan(i - 0.5, i + 0.5, color="#94a3b8", alpha=0.25, label="Gap Telecom (Cobertura < 70%)" if not gap_labeled else None)
            gap_labeled = True

    # Step-change CUSUM
    step = ts.get("step_change")
    if step and step.get("direction") == "QUEDA_DEGRAU":
        s_idx = int(step["index"])
        ax_top.axvspan(s_idx, len(xs) - 1, color="#f59e0b", alpha=0.13, label=f"Regime Pós-Degrau ({step['delta_pct']:+.1f}%)")
        ax_top.axvline(s_idx, color="#d97706", ls="-.", lw=1.8, label=f"Alarme CUSUM: Queda ({labels[s_idx]})")

    # Spikes & Drops
    spike_xs = [i for i, f in enumerate(flags) if f == "PICO_OUTLIER"]
    if spike_xs:
        ax_top.scatter(spike_xs, vals[spike_xs], color="#dc2626", s=65, zorder=5, marker="^", label="Pico Anômalo (Z-MAD > +3.5)")
        peak_idx = max(spike_xs, key=lambda idx: vals[idx])
        ax_top.annotate(
            f"Pico: {vals[peak_idx]:.2f} {unit}",
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
        ax_top.scatter(drop_xs, vals[drop_xs], color="#d97706", s=50, zorder=5, marker="v", label="Queda Pontual (Z-MAD < -3.5)")

    streak_val = ts.get("dias_anomalia_seguidos", 0)
    streak_txt = f" | Dias Anomalia Seguidos: {streak_val}d" if streak_val > 0 else ""
    ax_top.set_ylabel(f"Energia Ativa ({unit})")
    ax_top.set_title(
        f"Diagnóstico On-Demand de Telemetria Real — UC {uc_disp}{streak_txt}\n"
        f"Série Real ({len(pts)} pontos · Granularidade: {gran})",
        fontsize=11.5,
        fontweight="bold"
    )
    ax_top.legend(loc="upper left", fontsize=8, ncol=2)

    # Bottom panel: Coverage
    covs = [p["coverage"] * 100.0 for p in pts]
    bar_colors = ["#ef4444" if c < 70.0 else ("#f59e0b" if c < 90.0 else "#10b981") for c in covs]
    ax_cov.bar(xs, covs, color=bar_colors, alpha=0.85, width=0.8)
    ax_cov.axhline(70.0, color="#dc2626", ls=":", lw=1.3, label="Limiar Qualidade Telecom (70%)")
    ax_cov.set_ylim(0, 108)
    ax_cov.set_ylabel("Cobertura (%)")
    ax_cov.set_xlabel("Linha do Tempo (Horas / Dias)")
    ax_cov.legend(loc="lower right", fontsize=8)

    tick_step = max(1, len(labels) // 10)
    ax_cov.set_xticks(xs[::tick_step])
    ax_cov.set_xticklabels(labels[::tick_step], rotation=22, ha="right", fontsize=8.5)

    plt.tight_layout()
    return fig

