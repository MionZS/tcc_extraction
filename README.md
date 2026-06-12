# ARAUCARIA daily pipeline

This project now runs a chained daily extraction based on the same Oracle/SQLAlchemy pattern used in `E:\mion\PythonVSCode\Daimon`.

## What it does

1. Runs `queries/cis_araucaria_ml_extract_lightweight_alt.sql` on the **CIS** database.
2. Reads the resulting `NIO` population for ARAUCARIA.
3. Sends those NIOs in batches to `queries/mdm_coluna.sql` on the **ORCA/AMI** database using SQLAlchemy plus `SYS.ODCIVARCHAR2LIST`.
4. Writes three daily outputs:
   - `araucaria_cis_YYYYMMDD.{csv,parquet}`
   - `araucaria_mdm_YYYYMMDD.{csv,parquet}`
   - `araucaria_daily_report_YYYYMMDD.{csv,parquet}`

The joined report is a left join from CIS to MDM on `NIO`, plus:
- `REPORT_DAY`
- `HAS_MDM_DATA`

## Important note about `DAYS_BACK`

`mdm_coluna.sql` currently uses:

- `DAYS_BACK = 1` -> yesterday
- `DAYS_BACK = 0` -> today

So the pipeline defaults to `--days-back 1`.

## Configuration

Create `config.json` from `config_example.json`.

This pipeline only uses these entries:
- `oracle.ORCA`
- `oracle.CIS`

The other database entries in the config are ignored by this project.

## Install

Use your preferred Python environment manager, then install dependencies from `pyproject.toml`.

Main dependencies:
- `sqlalchemy`
- `oracledb`
- `polars`

## Run

```bash
python main.py --days-back 1 --output-dir output
```

Optional flags:

```bash
python main.py \
  --days-back 1 \
  --mdm-batch-size 500 \
  --fetch-size 1000 \
  --keep-temp
```

## Current behavior

- Uses the two SQL files already present in `queries/`
- Keeps `mdm_coluna.sql` semantics unchanged
- Batches NIOs for the MDM step
- Writes both CSV and Parquet outputs
- Produces a final joined daily report

## Files added

- `db.py`
- `pipeline.py`
- updated `main.py`
- `config_example.json`
- `.gitignore`
