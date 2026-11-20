import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl

target = "90850700"
print(f"=== RECONCILIAÇÃO CRÍTICA DA UC {target} ===")

# 1. classified_ucs_latest.parquet
p_clf = Path("output/reports/classified_ucs_latest.parquet")
if p_clf.exists():
    df_clf = pl.read_parquet(p_clf)
    sub = df_clf.filter((pl.col("id__uc_id") == target) | (pl.col("id__uc_id") == target.lstrip("0")))
    print(f"\n1. Em classified_ucs_latest.parquet ({len(sub)} linhas):")
    if len(sub) > 0:
        for r in sub.to_dicts():
            for k, v in r.items():
                if any(x in k.lower() for x in ["score", "anom", "diag", "categ", "vistoria", "streak", "dias", "slope", "median", "mad", "cutoff", "split"]):
                    print(f"   {k}: {v}")

# 2. training_dataset.parquet
p_trn = Path("output/model_input/v1/training_dataset.parquet")
if p_trn.exists():
    df_trn = pl.read_parquet(p_trn)
    sub_trn = df_trn.filter((pl.col("id__uc_id") == target) | (pl.col("id__uc_id") == target.lstrip("0")))
    print(f"\n2. Em training_dataset.parquet ({len(sub_trn)} linhas):")
    if len(sub_trn) > 0:
        for r in sub_trn.to_dicts():
            print(f"   Cutoff={r.get('meta__cutoff_date')}, split={r.get('meta__split')}, quality={r.get('meta__data_quality')}, fa_median={r.get('x__hist_fa_sum_median')}, slope={r.get('x__trend_fa_slope')}, reliable_days={r.get('x__coverage_reliable_days')}")

# 3. Daily features history
p_day = Path("output/features/uc_day_features")
day_files = sorted(p_day.glob("**/data.parquet"))
rows = []
for f in day_files:
    try:
        df_d = pl.read_parquet(f)
        sub_d = df_d.filter((pl.col("UC") == target) | (pl.col("UC") == target.lstrip("0")) | (pl.col("UC") == target.zfill(10)))
        if len(sub_d) > 0:
            rows.append(sub_d)
    except Exception:
        pass
if rows:
    all_d = pl.concat(rows).sort("REPORT_DAY")
    print(f"\n3. Em uc_day_features: {len(all_d)} dias encontrados!")
    print(f"   Primeiro dia: {all_d['REPORT_DAY'].min()}, Último dia: {all_d['REPORT_DAY'].max()}")
    print("   Amostra de consumo e cobertura dos últimos 15 dias:")
    for r in all_d.tail(15).to_dicts():
        print(f"     {r.get('REPORT_DAY')}: FA={r.get('FA_INTERVAL_SUM'):.2f} kWh, cov={r.get('INTERVAL_COVERAGE'):.1%}, reg={r.get('REGISTER_COVERAGE')}")
