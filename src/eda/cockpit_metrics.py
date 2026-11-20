"""Reconciled KPI and Granularity Engine for the AMI Analytical Cockpit.

Formalizes the analytical entities and guarantees exact mathematical consistency
between KPI cards, category breakdown panels, table filters, and CSV exports:

Entities:
- UC: Unidade Consumidora (distinct consumer unit, `id__uc_id`).
- NIO: Meter serial/logical identifier (`NIO`).
- Janela (Window): One row in `classified_ucs_latest` (`UC x cutoff_date`).
- Ocorrência: A window row where a flag/category is active.
- Candidato: A unique UC (at its latest cutoff) with `IS_CANDIDATO_VISTORIA == True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import polars as pl


@dataclass(frozen=True)
class KPIDefinition:
    """Formal specification of a Cockpit KPI for auditability."""

    key: str
    label: str
    granularity: str
    denominator: str
    formula: str
    filters: str


KPI_SPECIFICATIONS: tuple[KPIDefinition, ...] = (
    KPIDefinition(
        key="total_unique_ucs",
        label="UCs Únicas Analisadas",
        granularity="UC (último cutoff por id__uc_id)",
        denominator="Todas as UCs presentes no recorte consolidado",
        formula="COUNT(DISTINCT id__uc_id)",
        filters="Sem filtro (deduplicado por id__uc_id mantendo último meta__cutoff_date)",
    ),
    KPIDefinition(
        key="total_windows",
        label="Janelas Avaliadas (UC × Cutoff)",
        granularity="Janela (id__uc_id × meta__cutoff_date)",
        denominator="Total de linhas da tabela classificada",
        formula="COUNT(*)",
        filters="Sem filtro",
    ),
    KPIDefinition(
        key="ucs_with_anomaly",
        label="UCs c/ Anomalia Detectada",
        granularity="UC (último cutoff por id__uc_id)",
        denominator="total_unique_ucs",
        formula="COUNT(DISTINCT id__uc_id WHERE IS_ANOMALY == True no último cutoff)",
        filters="IS_ANOMALY == True",
    ),
    KPIDefinition(
        key="candidatos_vistoria",
        label="Candidatos a Vistoria (Campo)",
        granularity="UC (último cutoff por id__uc_id)",
        denominator="total_unique_ucs",
        formula="COUNT(DISTINCT id__uc_id WHERE IS_CANDIDATO_VISTORIA == True)",
        filters="IS_CANDIDATO_VISTORIA == True",
    ),
    KPIDefinition(
        key="ucs_possivel_gd",
        label="UCs c/ Possível GD Não Cadastrada",
        granularity="UC (último cutoff por id__uc_id)",
        denominator="total_unique_ucs",
        formula="COUNT(DISTINCT id__uc_id WHERE CATEGORIA_DIAGNOSTICA == 'POSSIVEL_GD_NAO_CADASTRADA')",
        filters="CATEGORIA_DIAGNOSTICA == 'POSSIVEL_GD_NAO_CADASTRADA'",
    ),
    KPIDefinition(
        key="ucs_quarentena_telecom",
        label="UCs em Quarentena Telecom",
        granularity="UC (último cutoff por id__uc_id)",
        denominator="total_unique_ucs",
        formula="COUNT(DISTINCT id__uc_id WHERE CATEGORIA_DIAGNOSTICA IN telecom_categories)",
        filters="CATEGORIA_DIAGNOSTICA IN ('TELECOM_QUARANTINE', 'ANOMALIA_COMUNICACAO_ISOLADA', 'GARGALO_COLETIVO_RF_MESH')",
    ),
)

TELECOM_CATEGORIES: tuple[str, ...] = (
    "TELECOM_QUARANTINE",
    "ANOMALIA_COMUNICACAO_ISOLADA",
    "GARGALO_COLETIVO_RF_MESH",
)


def deduplicate_latest_uc(df: pl.DataFrame) -> pl.DataFrame:
    """Reduce window-level table (UC x cutoff_date) to 1 latest row per unique UC."""
    if df.is_empty():
        return df
    uc_col = "id__uc_id" if "id__uc_id" in df.columns else ("UC" if "UC" in df.columns else None)
    if uc_col is None:
        return df
    cutoff_col = "meta__cutoff_date" if "meta__cutoff_date" in df.columns else (
        "CUTOFF_DATE" if "CUTOFF_DATE" in df.columns else None
    )
    if cutoff_col:
        return df.sort(cutoff_col).unique(subset=[uc_col], keep="last")
    return df.unique(subset=[uc_col], keep="last")


def filter_cockpit_dataframe(
    df: pl.DataFrame,
    *,
    granularity: str = "unique_ucs",
    category: str = "",
    feeder: str = "",
    search_q: str = "",
    filter_col: str = "",
    filter_op: str = "eq",
    filter_val: str = "",
) -> tuple[pl.DataFrame, bool]:
    """Apply reconciled filtering matching both dashboard counts and table rows.

    Returns (filtered_df, sort_candidates_flag).
    """
    if df.is_empty():
        return df, False

    work = deduplicate_latest_uc(df) if granularity == "unique_ucs" else df
    sort_candidates = False

    if category:
        if category == "CANDIDATOS_A_VISTORIA":
            sort_candidates = True
            if "IS_CANDIDATO_VISTORIA" in work.columns:
                work = work.filter(pl.col("IS_CANDIDATO_VISTORIA") == True)  # noqa: E712
            elif "IS_ANOMALY" in work.columns:
                work = work.filter(pl.col("IS_ANOMALY") == True)  # noqa: E712
                if "CATEGORIA_DIAGNOSTICA" in work.columns:
                    work = work.filter(
                        ~pl.col("CATEGORIA_DIAGNOSTICA").is_in([
                            "NORMAL", "REGULAR", "OPERACAO_REGULAR",
                            "QUEDA_TELECOM_ISOLADA", "QUARENTENA_TELECOM",
                            "TELECOM_QUARANTINE", "MEDIDOR_NOVO_QUARENTENA",
                        ])
                    )
        elif category == "TELECOM_ALL":
            if "CATEGORIA_DIAGNOSTICA" in work.columns:
                work = work.filter(pl.col("CATEGORIA_DIAGNOSTICA").is_in(list(TELECOM_CATEGORIES)))
        elif "CATEGORIA_DIAGNOSTICA" in work.columns:
            work = work.filter(
                (pl.col("CATEGORIA_DIAGNOSTICA") == category)
                | (pl.col("CATEGORIA_DIAGNOSTICA").cast(pl.String).str.contains(category))
            )

    if feeder:
        if "meta__feeder" in work.columns:
            work = work.filter(pl.col("meta__feeder") == feeder)
        elif "ALIMENTADOR" in work.columns:
            work = work.filter(pl.col("ALIMENTADOR") == feeder)

    if search_q:
        q_low = search_q.strip().lower()
        uc_col = "id__uc_id" if "id__uc_id" in work.columns else "UC"
        feeder_col = "meta__feeder" if "meta__feeder" in work.columns else ("ALIMENTADOR" if "ALIMENTADOR" in work.columns else None)
        cond = pl.col(uc_col).cast(pl.String).str.to_lowercase().str.contains(q_low, literal=True)
        if feeder_col:
            cond = cond | pl.col(feeder_col).cast(pl.String).fill_null("").str.to_lowercase().str.contains(q_low, literal=True)
        work = work.filter(cond)

    if filter_col and filter_col in work.columns and filter_val != "":
        col_type = str(work.schema[filter_col]).lower()
        if "int" in col_type or "float" in col_type:
            f_val = float(filter_val)
            if filter_op == "eq":
                work = work.filter(pl.col(filter_col) == f_val)
            elif filter_op == "gt":
                work = work.filter(pl.col(filter_col) > f_val)
            elif filter_op == "gte":
                work = work.filter(pl.col(filter_col) >= f_val)
            elif filter_op == "lt":
                work = work.filter(pl.col(filter_col) < f_val)
            elif filter_op == "lte":
                work = work.filter(pl.col(filter_col) <= f_val)
            elif filter_op == "ne":
                work = work.filter(pl.col(filter_col) != f_val)
        elif "bool" in col_type:
            b_val = filter_val.lower() in ("true", "1", "t", "sim", "yes")
            work = work.filter(pl.col(filter_col) == b_val)
        else:
            if filter_op == "eq":
                work = work.filter(pl.col(filter_col).cast(pl.String) == filter_val)
            elif filter_op == "contains":
                work = work.filter(
                    pl.col(filter_col).cast(pl.String).str.to_lowercase().str.contains(filter_val.lower(), literal=True)
                )
            elif filter_op == "ne":
                work = work.filter(pl.col(filter_col).cast(pl.String) != filter_val)

    if sort_candidates:
        if "DIAS_ANOMALIA_SEGUIDOS" in work.columns and "ANOMALY_SCORE" in work.columns:
            work = work.sort(["DIAS_ANOMALIA_SEGUIDOS", "ANOMALY_SCORE"], descending=[True, True])
        elif "ANOMALY_SCORE" in work.columns:
            work = work.sort("ANOMALY_SCORE", descending=True)
    elif "ANOMALY_SCORE" in work.columns:
        work = work.sort("ANOMALY_SCORE", descending=True)

    return work, sort_candidates


def compute_reconciled_kpis(df: pl.DataFrame) -> dict[str, Any]:
    """Compute strictly reconciled KPIs at both UC and Window granularities."""
    if df.is_empty():
        return {
            "total_windows": 0,
            "total_unique_ucs": 0,
            "windows_with_anomaly": 0,
            "ucs_with_anomaly": 0,
            "anomaly_rate_ucs_pct": 0.0,
            "anomaly_rate_windows_pct": 0.0,
            "candidatos_vistoria_ucs": 0,
            "candidatos_vistoria_windows": 0,
            "possivel_gd_ucs": 0,
            "possivel_gd_windows": 0,
            "quarentena_telecom_ucs": 0,
            "quarentena_telecom_windows": 0,
            "reference_cutoff_min": None,
            "reference_cutoff_max": None,
        }

    cols = set(df.columns)
    anom_col = "IS_ANOMALY" if "IS_ANOMALY" in cols else None
    cat_col = "CATEGORIA_DIAGNOSTICA" if "CATEGORIA_DIAGNOSTICA" in cols else None
    cand_col = "IS_CANDIDATO_VISTORIA" if "IS_CANDIDATO_VISTORIA" in cols else None
    cutoff_col = "meta__cutoff_date" if "meta__cutoff_date" in cols else None

    unique_df = deduplicate_latest_uc(df)

    total_windows = int(df.height)
    total_unique_ucs = int(unique_df.height)

    win_anom = int(df.filter(pl.col(anom_col) == True).height) if anom_col else 0  # noqa: E712
    uc_anom = int(unique_df.filter(pl.col(anom_col) == True).height) if anom_col else 0  # noqa: E712

    cand_win = int(df.filter(pl.col(cand_col) == True).height) if cand_col else 0  # noqa: E712
    cand_uc = int(unique_df.filter(pl.col(cand_col) == True).height) if cand_col else 0  # noqa: E712

    gd_win = int(df.filter(pl.col(cat_col) == "POSSIVEL_GD_NAO_CADASTRADA").height) if cat_col else 0
    gd_uc = int(unique_df.filter(pl.col(cat_col) == "POSSIVEL_GD_NAO_CADASTRADA").height) if cat_col else 0

    tel_win = int(df.filter(pl.col(cat_col).is_in(list(TELECOM_CATEGORIES))).height) if cat_col else 0
    tel_uc = int(unique_df.filter(pl.col(cat_col).is_in(list(TELECOM_CATEGORIES))).height) if cat_col else 0

    cutoff_min = str(df[cutoff_col].min()) if cutoff_col else None
    cutoff_max = str(df[cutoff_col].max()) if cutoff_col else None

    return {
        "total_windows": total_windows,
        "total_unique_ucs": total_unique_ucs,
        "windows_with_anomaly": win_anom,
        "ucs_with_anomaly": uc_anom,
        "anomaly_rate_ucs_pct": round((uc_anom / max(total_unique_ucs, 1)) * 100.0, 2),
        "anomaly_rate_windows_pct": round((win_anom / max(total_windows, 1)) * 100.0, 2),
        "candidatos_vistoria_ucs": cand_uc,
        "candidatos_vistoria_windows": cand_win,
        "possivel_gd_ucs": gd_uc,
        "possivel_gd_windows": gd_win,
        "quarentena_telecom_ucs": tel_uc,
        "quarentena_telecom_windows": tel_win,
        "reference_cutoff_min": cutoff_min,
        "reference_cutoff_max": cutoff_max,
    }
