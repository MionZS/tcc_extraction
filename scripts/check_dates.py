"""One-off utility to find the earliest installation dates in the dataset."""
import polars as pl
from pathlib import Path

BASE = Path("output")

print("=== VERIFICANDO DATAS DE INSTALAÇÃO ===")

# 1. meter_installation_history.csv
hist_path = BASE / "context" / "meter_installation_history.csv"
if hist_path.exists():
    df_hist = pl.read_csv(hist_path, separator=";", infer_schema_length=5000, ignore_errors=True)
    if "DATA_INSTALACAO" in df_hist.columns:
        dates = df_hist.filter(pl.col("DATA_INSTALACAO").is_not_null() & (pl.col("DATA_INSTALACAO") != "")).get_column("DATA_INSTALACAO").sort()
        print(f"Total registros em meter_installation_history: {df_hist.height}")
        print(f"Primeiras 5 datas de DATA_INSTALACAO: {dates.head(5).to_list()}")
        print(f"Ultimas 5 datas de DATA_INSTALACAO: {dates.tail(5).to_list()}")

# 2. uc_context.csv
uc_path = BASE / "context" / "uc_context.csv"
if uc_path.exists():
    df_uc = pl.read_csv(uc_path, separator=";", infer_schema_length=5000, ignore_errors=True)
    if "INICIO_UC" in df_uc.columns:
        dates_uc = df_uc.filter(pl.col("INICIO_UC").is_not_null() & (pl.col("INICIO_UC") != "")).get_column("INICIO_UC").sort()
        print(f"Total registros em uc_context: {df_uc.height}")
        print(f"Total UCs com INICIO_UC preenchido: {dates_uc.len()}")
        if dates_uc.len() > 0:
            print(f"Primeiras 5 datas de INICIO_UC: {dates_uc.head(5).to_list()}")

# 3. training_dataset meter_age_days
train_path = BASE / "model_input" / "v1" / "training_dataset.parquet"
if train_path.exists():
    df_train = pl.read_parquet(train_path, columns=["meta__cutoff_date", "x__meter_age_days", "x__meter_changed_30d"])
    print(f"Max meter_age_days: {df_train['x__meter_age_days'].max()}")
    print(f"Min meter_age_days: {df_train['x__meter_age_days'].min()}")
    cutoffs = df_train["meta__cutoff_date"].unique().to_list()
    print(f"Cutoffs no dataset: {cutoffs}")

