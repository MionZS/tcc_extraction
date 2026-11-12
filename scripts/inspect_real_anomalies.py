"""
scripts/inspect_real_anomalies.py
-------------------------------------------------------------------------------
Lê o dataset de classificação real e encontra:
1. O par real exato de UCs com mesmo consumo e perfis opostos.
2. A UC anômala real mais severa em cada categoria diagnóstica.
3. Imprime os IDs e valores para documentação no TCC e atualiza o YAML.
-------------------------------------------------------------------------------
"""

from pathlib import Path
import polars as pl
import yaml
from src.utils.censorship import censor_uc_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "output" / "reports" / "classified_ucs_latest.parquet"
CSV_PATH = PROJECT_ROOT / "output" / "reports" / "classified_ucs_latest.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "ucs_interesse.yaml"


def main():
    if PARQUET_PATH.exists():
        df = pl.read_parquet(PARQUET_PATH)
    else:
        df = pl.read_csv(CSV_PATH, separator=";")

    print(f"Dataset carregado com {df.height:,} linhas.")

    # 1. Encontrar o par real
    median_col = "x__hist_fa_sum_median" if "x__hist_fa_sum_median" in df.columns else "fa_sum_median"
    mad_col = "x__hist_fa_sum_mad" if "x__hist_fa_sum_mad" in df.columns else "fa_sum_mad"
    cov_col = "x__avg_interval_coverage" if "x__avg_interval_coverage" in df.columns else "score_comunicacao"
    id_col = "id__uc_id" if "id__uc_id" in df.columns else "UC"

    df_valid = df.filter(
        (pl.col(cov_col) >= 0.80) &
        (pl.col(median_col) >= 15.0) &
        (pl.col(median_col) <= 25.0)
    ).sort(median_col)

    rows = df_valid.to_dicts()
    best_pair = None
    best_ratio = 1.0

    for i in range(len(rows) - 1):
        r1 = rows[i]
        r2 = rows[i + 1]
        c1 = r1.get(median_col, 0.0)
        c2 = r2.get(median_col, 0.0)
        if c1 and c2 and abs(c1 - c2) / c1 < 0.005:  # menos de 0.5% de diferença de consumo!
            m1 = max(float(r1.get(mad_col) or 0.1), 1e-4)
            m2 = max(float(r2.get(mad_col) or 0.1), 1e-4)
            ratio = max(m1, m2) / min(m1, m2)
            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (r1, r2) if m1 < m2 else (r2, m1)

    print("\n" + "="*70)
    print("PAR REAL COM MESMO CONSUMO E DISPERSÃO EXTREMAMENTE DIVERGENTE:")
    if best_pair:
        u_st, u_vol = best_pair
        print(f"UC Estável : {u_st[id_col]} | Consumo={u_st[median_col]:.3f} kWh/d | MAD={u_st[mad_col]:.3f} | Score={u_st.get('ANOMALY_SCORE', 0):.3f}")
        print(f"UC Volátil : {u_vol[id_col]} | Consumo={u_vol[median_col]:.3f} kWh/d | MAD={u_vol[mad_col]:.3f} | Score={u_vol.get('ANOMALY_SCORE', 0):.3f}")
        print(f"Razão de Divergência de Variabilidade: {best_ratio:.2f}x")

    # 2. UCs reais anômalas por categoria
    print("\n" + "="*70)
    print("TOP UCS REAIS POR CATEGORIA DIAGNÓSTICA:")
    cat_col = "CATEGORIA_DIAGNOSTICA" if "CATEGORIA_DIAGNOSTICA" in df.columns else "categoria_diagnostica"
    cats = df.get_column(cat_col).drop_nulls().unique().to_list()
    
    top_ucs = {}
    for c in cats:
        sub = df.filter(pl.col(cat_col) == c).sort("ANOMALY_SCORE", descending=True)
        if sub.height > 0:
            top_row = sub.to_dicts()[0]
            top_ucs[c] = top_row
            print(f"[{c}] -> UC: {top_row[id_col]} | Score={top_row.get('ANOMALY_SCORE', 0):.4f} | Mediana={top_row.get(median_col, 0):.2f} kWh/d | MAD={top_row.get(mad_col, 0):.2f}")

    # 3. Salvar no YAML de configuração
    yaml_data = {
        "par_comparativo_consumo_identico": {
            "uc_estavel": {
                "id_real": str(best_pair[0][id_col]) if best_pair else "N/A",
                "id_censurado": censor_uc_id(best_pair[0][id_col]) if best_pair else "N/A",
                "consumo_mediano_kwh": float(best_pair[0][median_col]) if best_pair else 0.0,
                "mad_kwh": float(best_pair[0][mad_col]) if best_pair else 0.0,
                "score_anomalia": float(best_pair[0].get("ANOMALY_SCORE", 0)) if best_pair else 0.0,
            },
            "uc_volatil": {
                "id_real": str(best_pair[1][id_col]) if best_pair else "N/A",
                "id_censurado": censor_uc_id(best_pair[1][id_col]) if best_pair else "N/A",
                "consumo_mediano_kwh": float(best_pair[1][median_col]) if best_pair else 0.0,
                "mad_kwh": float(best_pair[1][mad_col]) if best_pair else 0.0,
                "score_anomalia": float(best_pair[1].get("ANOMALY_SCORE", 0)) if best_pair else 0.0,
            }
        },
        "top_ucs_anomalas_por_categoria": {
            c: {
                "uc_id_real": str(row[id_col]),
                "uc_id_censurado": censor_uc_id(row[id_col]),
                "score_anomalia": float(row.get("ANOMALY_SCORE", 0)),
                "consumo_mediano": float(row.get(median_col, 0)),
                "mad": float(row.get(mad_col, 0)),
                "alimentador": str(row.get("meta__feeder", row.get("ALIMENTADOR", ""))),
            }
            for c, row in top_ucs.items()
        }
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
    print(f"\nConfiguração salva em: {CONFIG_PATH}")


if __name__ == "__main__":
    main()

