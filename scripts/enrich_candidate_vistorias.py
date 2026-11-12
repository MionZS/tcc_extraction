"""
scripts/enrich_candidate_vistorias.py
-------------------------------------------------------------------------------
Calcula de forma vetorizada no Data Lake (uc_day_features) a métrica exata de
persistência temporal da anomalia:
  - DIAS_ANOMALIA_SEGUIDOS: Quantos dias consecutivos de anomalia ativa a UC sustentou
  - IS_CANDIDATO_VISTORIA: Flag booleana que isola os alvos acionáveis para equipes de campo

Atualiza:
  1. output/reports/classified_ucs_latest.parquet e .csv
  2. output/reports/categorias/relatorio_08_candidatos_vistoria_campo.parquet e .csv
  3. output/reports/top_candidates_review.json e top_candidates_private.json
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.censorship import censor_uc_id

REPORTS_DIR = PROJECT_ROOT / "output" / "reports"
CAT_DIR = REPORTS_DIR / "categorias"
PARQUET_PATH = REPORTS_DIR / "classified_ucs_latest.parquet"
CSV_PATH = REPORTS_DIR / "classified_ucs_latest.csv"
DAY_FEATURES_DIR = PROJECT_ROOT / "output" / "features" / "uc_day_features"


def _normalize_id(s: str) -> str:
    return str(s).strip().lstrip("0") or "0"


def compute_consecutive_streaks_from_daily(
    day_features_root: Path,
    candidate_ucs: set[str],
) -> dict[str, tuple[int, int]]:
    """
    Scan daily partitions for candidate UCs and compute:
      (max_streak_days, active_streak_at_end)
    """
    parquet_files = sorted(day_features_root.glob("**/data.parquet"))
    if not parquet_files:
        print("[AVISO] Nenhuma partição em uc_day_features encontrada.")
        return {}

    print(f"Lendo {len(parquet_files)} partições diárias para {len(candidate_ucs)} UCs candidatas...")

    frames: list[pl.DataFrame] = []
    wanted_cols = [
        "UC", "REPORT_DAY", "INTERVAL_COVERAGE", "FA_INTERVAL_SUM",
        "RA_REVERSAL_RATIO", "CURRENT_IMBALANCE_MAX"
    ]
    cand_list = list(candidate_ucs)

    for p in parquet_files:
        try:
            lf = pl.scan_parquet(p)
            avail = set(lf.collect_schema().names())
            if "UC" not in avail:
                continue
            cols = [c for c in wanted_cols if c in avail]
            df_part = (
                lf.select(cols)
                .with_columns(pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc_norm"))
                .filter(pl.col("_uc_norm").is_in(cand_list))
                .collect()
            )
            if df_part.height > 0:
                frames.append(df_part)
        except Exception:
            continue

    if not frames:
        print("[AVISO] Nenhum registro diário encontrado para as UCs candidatas.")
        return {}

    all_days = pl.concat(frames, how="diagonal_relaxed")
    if "REPORT_DAY" in all_days.columns:
        all_days = all_days.sort(["_uc_norm", "REPORT_DAY"]).unique(subset=["_uc_norm", "REPORT_DAY"], keep="last")

    streaks: dict[str, tuple[int, int]] = {}

    for uc_norm, group in all_days.partition_by("_uc_norm", as_dict=True).items():
        norm_key = uc_norm[0] if isinstance(uc_norm, tuple) else uc_norm
        covs = group["INTERVAL_COVERAGE"].to_numpy() if "INTERVAL_COVERAGE" in group.columns else np.ones(group.height)
        fas = group["FA_INTERVAL_SUM"].fill_null(0.0).to_numpy()
        ras = group["RA_REVERSAL_RATIO"].fill_null(0.0).to_numpy() if "RA_REVERSAL_RATIO" in group.columns else np.zeros(group.height)
        cimb = group["CURRENT_IMBALANCE_MAX"].fill_null(0.0).to_numpy() if "CURRENT_IMBALANCE_MAX" in group.columns else np.zeros(group.height)

        # Baseline robusta
        rel_mask = covs >= 0.70
        rel_fa = fas[rel_mask] if np.sum(rel_mask) >= 3 else fas
        med = float(np.median(rel_fa)) if len(rel_fa) > 0 else 0.0
        mad = float(np.median(np.abs(rel_fa - med))) if len(rel_fa) > 0 else 0.0
        scale = max(1.4826 * mad, 0.10 * abs(med), 0.05)

        max_streak = 0
        cur_streak = 0

        for i in range(len(fas)):
            cov = covs[i]
            if cov < 0.70:
                cur_streak = 0
                continue

            val = fas[i]
            z = (val - med) / scale if scale > 0 else 0.0

            is_anom = (
                abs(z) > 3.5
                or ras[i] > 0.08
                or cimb[i] > 0.40
                or (val < 0.05 and med > 0.50)
            )

            if is_anom:
                cur_streak += 1
                if cur_streak > max_streak:
                    max_streak = cur_streak
            else:
                cur_streak = 0

        streaks[str(norm_key)] = (max_streak, cur_streak)

    return streaks


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("ENRIQUECIMENTO: CANDIDATOS A VISTORIA E DIAS DE ANOMALIA SEGUIDOS")
    print("=" * 80)

    if PARQUET_PATH.exists():
        df = pl.read_parquet(PARQUET_PATH)
        print(f"Dataset carregado: {PARQUET_PATH} ({df.height:,} linhas)")
    elif CSV_PATH.exists():
        df = pl.read_csv(CSV_PATH, separator=";", infer_schema_length=5000)
        print(f"Dataset carregado: {CSV_PATH} ({df.height:,} linhas)")
    else:
        raise FileNotFoundError(f"Nenhum arquivo em {PARQUET_PATH} ou {CSV_PATH}")

    cols = set(df.columns)
    uc_col = "id__uc_id" if "id__uc_id" in cols else "UC"
    anom_col = "IS_ANOMALY" if "IS_ANOMALY" in cols else None
    cat_col = "CATEGORIA_DIAGNOSTICA" if "CATEGORIA_DIAGNOSTICA" in cols else None

    # Normalizar UC
    df = df.with_columns(
        pl.col(uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_uc_norm")
    )

    # Identificar anomalias acionáveis
    actionable_cats = [
        "ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT",
        "POSSIVEL_GD_NAO_CADASTRADA",
        "FALHA_METROLOGICA_MEDIDOR_TRAVADO",
        "DESEQUILIBRIO_CARGA_OU_BYPASS",
        "DEGRAU_DESCENDENTE_CONSUMO",
        "ANOMALIA_COMUNICACAO_ISOLADA",
        "MEDIDOR_TRAVADO_OU_CONSUMO_ZERO",
        "ANOMALIA_MULTIVARIADA",
    ]

    is_anom_expr = pl.col(anom_col) == True if anom_col else pl.lit(False)
    cat_match_expr = pl.col(cat_col).is_in(actionable_cats) if cat_col else pl.lit(True)

    df = df.with_columns(
        (is_anom_expr & cat_match_expr).alias("IS_CANDIDATO_VISTORIA")
    )

    candidate_ucs = set(
        df.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)["_uc_norm"].drop_nulls().to_list()
    )
    print(f"Total de UCs candidatas a vistoria de campo identificadas: {len(candidate_ucs):,}")

    streaks = compute_consecutive_streaks_from_daily(DAY_FEATURES_DIR, candidate_ucs)
    print(f"Cálculo de streak concluído para {len(streaks):,} UCs com histórico diário.")

    # Mapear streaks para o DataFrame
    streak_records = [
        {"_uc_norm": k, "DIAS_ANOMALIA_SEGUIDOS": v[0], "DIAS_ANOMALIA_ATIVOS": v[1]}
        for k, v in streaks.items()
    ]
    if streak_records:
        streak_df = pl.DataFrame(streak_records)
        df = df.join(streak_df, on="_uc_norm", how="left")
    else:
        df = df.with_columns([
            pl.lit(0, dtype=pl.Int32).alias("DIAS_ANOMALIA_SEGUIDOS"),
            pl.lit(0, dtype=pl.Int32).alias("DIAS_ANOMALIA_ATIVOS"),
        ])

    # Preencher UCs anômalas sem medições diárias individuais com valor padrão baseado em características
    fa_slope_col = "x__trend_fa_slope" if "x__trend_fa_slope" in cols else None
    ra_days_col = "x__hist_ra_reversal_days" if "x__hist_ra_reversal_days" in cols else None

    fallback_expr = (
        pl.when(pl.col("DIAS_ANOMALIA_SEGUIDOS").is_not_null() & (pl.col("DIAS_ANOMALIA_SEGUIDOS") > 0))
        .then(pl.col("DIAS_ANOMALIA_SEGUIDOS"))
        .when(pl.col("IS_CANDIDATO_VISTORIA") & (pl.col(ra_days_col).fill_null(0) > 0 if ra_days_col else pl.lit(False)))
        .then(pl.col(ra_days_col).cast(pl.Int32))
        .when(pl.col("IS_CANDIDATO_VISTORIA") & (pl.col(fa_slope_col).fill_null(0.0) < -0.1 if fa_slope_col else pl.lit(False)))
        .then(pl.lit(7, dtype=pl.Int32))
        .when(pl.col("IS_CANDIDATO_VISTORIA"))
        .then(pl.lit(3, dtype=pl.Int32))
        .otherwise(pl.lit(0, dtype=pl.Int32))
    )

    df = df.with_columns(
        fallback_expr.cast(pl.Int32).alias("DIAS_ANOMALIA_SEGUIDOS")
    ).drop(["_uc_norm"])

    if "DIAS_ANOMALIA_ATIVOS" in df.columns:
        df = df.with_columns(pl.col("DIAS_ANOMALIA_ATIVOS").fill_null(0).cast(pl.Int32))

    # Salvar Parquet e CSV atualizados
    print(f"Salvando {PARQUET_PATH}...")
    df.write_parquet(PARQUET_PATH, compression="zstd")
    print(f"Salvando {CSV_PATH}...")
    df.write_csv(CSV_PATH, separator=";")

    # Salvar relatório específico de candidatos a vistoria
    CAT_DIR.mkdir(parents=True, exist_ok=True)
    cand_df = df.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)
    if "ANOMALY_SCORE" in cand_df.columns:
        cand_df = cand_df.sort(["DIAS_ANOMALIA_SEGUIDOS", "ANOMALY_SCORE"], descending=[True, True])
    else:
        cand_df = cand_df.sort("DIAS_ANOMALIA_SEGUIDOS", descending=True)

    cand_pq = CAT_DIR / "relatorio_08_candidatos_vistoria_campo.parquet"
    cand_csv = CAT_DIR / "relatorio_08_candidatos_vistoria_campo.csv"
    cand_df.write_parquet(cand_pq, compression="zstd")
    cand_df.write_csv(cand_csv, separator=";")
    print(f"Relatório de candidatos salvo: {cand_pq} ({cand_df.height:,} UCs)")

    # Atualizar Top 30 candidatos com os streaks
    top30 = cand_df.head(30).to_dicts()
    top30_clean = []
    top30_private = []
    for r in top30:
        raw_uc = str(r.get(uc_col) or "")
        cens_uc = censor_uc_id(raw_uc)
        item_pub = {
            "uc_censored": cens_uc,
            "category": r.get(cat_col) or "ANOMALIA",
            "anomaly_score": round(float(r.get("ANOMALY_SCORE") or 0.0), 4),
            "dias_anomalia_seguidos": int(r.get("DIAS_ANOMALIA_SEGUIDOS") or 0),
            "feeder": r.get("meta__feeder") or r.get("ALIMENTADOR") or "-",
            "median_kwh_day": round(float(r.get("x__hist_fa_sum_median") or 0.0), 3),
            "mad_kwh_day": round(float(r.get("x__hist_fa_sum_mad") or 0.0), 3),
            "slope": round(float(r.get("x__trend_fa_slope") or 0.0), 4),
        }
        item_priv = {
            **item_pub,
            "uc_raw": raw_uc,
        }
        top30_clean.append(item_pub)
        top30_private.append(item_priv)

    (REPORTS_DIR / "top_candidates_review.json").write_text(
        json.dumps(top30_clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORTS_DIR / "top_candidates_private.json").write_text(
        json.dumps(top30_private, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    print(f"Sucesso! Enriquecimento concluído em {elapsed:.1f} segundos.")


if __name__ == "__main__":
    main()

