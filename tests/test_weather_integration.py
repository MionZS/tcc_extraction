"""Integration test: weather module × Open-Meteo API × pipeline data.

Chama a API real da Open-Meteo com coordenadas de UCs extraídas
do pipeline, gera um DataFrame no formato do pipeline, enriquece
uma grade horária sintética e salva tudo para inspeção manual.

Usage
-----
    uv run python tests/test_weather_integration.py

Opcional:
    --days 30       Quantos dias de weather buscar (default: 30)
    --nios  5       Quantos NIOs amostrar do CIS (default: 5)
    --output-dir    Onde salvar os resultados (default: output/tests/weather_integration)
    --cis-csv       Caminho do CSV CIS para ler UCs (default: output/raw/CIS/araucaria_cis_YYYYMMDD.csv mais recente)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

# Garantir que o diretório raiz do projeto está no sys.path
# para que `from src.weather import ...` funcione
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.weather import (
    _format_weather_url,
    build_weather_dataframe,
    enrich_timegrid,
    get_coordinates_from_cis,
    fetch_historical_weather,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "tests" / "weather_integration"
DEFAULT_CIS_DIR = ROOT / "output" / "raw" / "CIS"
LATEST_CIS_GLOB = "araucaria_cis_*.csv"


def _find_latest_cis_csv() -> Path:
    """Encontra o CSV CIS mais recente (pela data no nome)."""
    files = sorted(DEFAULT_CIS_DIR.glob(LATEST_CIS_GLOB))
    if not files:
        # Tentar diretório alternativo
        alt = ROOT / "output" / "raw" / "CIS"
        files = sorted(alt.glob(LATEST_CIS_GLOB))
    if not files:
        print(f"Nenhum arquivo CIS encontrado em {DEFAULT_CIS_DIR}")
        # Fallback: dados sintéticos baseados no CSV que já vimos
        return None
    # O último arquivo (ordem alfabética = data mais recente)
    return files[-1]


def _sample_nios_from_cis(cis_path: Path, n: int = 5) -> tuple[pl.DataFrame, list[str]]:
    """Lê o CSV CIS e retorna n NIOs (primeiros n válidos com LAT/LONG)."""
    df = pl.read_csv(
        cis_path,
        separator=";",
        encoding="utf-8-sig",
        infer_schema_length=None,
    )
    # Garantir que lat/LONG são strings que podemos converter
    lat_col = next((c for c in df.columns if c.strip().upper() == "LAT"), None)
    lon_col = next((c for c in df.columns if c.strip().upper() == "LONG"), None)
    nio_col = next((c for c in df.columns if c.strip().upper() == "NIO"), None)
    uc_col = next((c for c in df.columns if c.strip().upper() == "UC"), None)

    if not all([lat_col, lon_col, nio_col]):
        print(f"Colunas necessárias não encontradas. Disponíveis: {df.columns}")
        sys.exit(1)

    # Filtrar linhas com LAT/LONG não nulos
    valid = df.filter(
        pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()
    )

    if valid.height < n:
        print(f"Aviso: só {valid.height} linhas com coordenadas válidas, usando todas")
        n = valid.height

    sampled = valid.head(n)
    nios = sampled[nio_col].cast(pl.String).to_list()

    print(f"Amostra de {n} NIOs do CIS:")
    if uc_col:
        for row in sampled.iter_rows(named=True):
            print(f"  UC={row[uc_col]}  NIO={row[nio_col]}  LAT={row[lat_col]}  LON={row[lon_col]}")
    else:
        for row in sampled.iter_rows(named=True):
            print(f"  NIO={row[nio_col]}  LAT={row[lat_col]}  LON={row[lon_col]}")

    return sampled, nios


def _build_synthetic_timegrid(
    nios: list[str],
    target_date: date,
    n_hours: int = 3,
) -> pl.DataFrame:
    """Cria um timegrid sintético (poucas horas) para testar o enrich."""
    rows: list[dict[str, Any]] = []
    for nio in nios:
        for hour in range(n_hours):
            for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
                rows.append({
                    "NIO": nio,
                    "Data Time": dt.strftime("%d/%m/%Y %H:%M:%S"),
                    "Active energy (+) TOTAL": 100.0 + hash(nio) % 50 + minute * 0.1,
                })
    return pl.DataFrame(rows)


def run_integration_test(
    cis_path: Path,
    output_dir: Path,
    n_nios: int = 5,
    days: int = 30,
) -> None:
    """Executa o teste de integração completo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()

    print("=" * 60)
    print("WEATHER INTEGRATION TEST")
    print("=" * 60)

    # ── Step 1: Amostrar NIOs do CIS ──────────────────────────────
    print(f"\n[1/5] Lendo CIS: {cis_path}")
    cis_df, nios = _sample_nios_from_cis(cis_path, n=n_nios)

    # ── Step 2: Extrair coordenadas medianas ──────────────────────
    print(f"\n[2/5] Extraindo coordenadas do CIS...")
    coords = get_coordinates_from_cis(cis_df)
    if coords is None:
        print("ERRO: não foi possível extrair coordenadas do CIS")
        sys.exit(1)
    lat, lon = coords
    print(f"  Coordenadas medianas: lat={lat:.6f}, lon={lon:.6f}")

    # ── Step 3: URL de teste ──────────────────────────────────────
    print(f"\n[3/5] Verificando URL da API...")
    test_date = today - timedelta(days=1)
    url = _format_weather_url(test_date, lat, lon)
    print(f"  URL gerada: {url[:100]}...")
    print(f"  (Acesse no navegador para ver o JSON bruto)")

    # ── Step 4: Fetch weather para o período ──────────────────────
    print(f"\n[4/5] Buscando weather da Open-Meteo para {days} dias...")
    start_date = today - timedelta(days=days)
    end_date = today - timedelta(days=1)

    all_weather_dfs: list[pl.DataFrame] = []
    success_count = 0
    fail_count = 0

    for i in range((end_date - start_date).days + 1):
        d = start_date + timedelta(days=i)
        try:
            df_day = fetch_historical_weather(d, lat, lon, timeout=30)
            if df_day.is_empty():
                print(f"  ⚠ {d.isoformat()}: sem dados")
                fail_count += 1
            else:
                # Adicionar coluna de data para referência
                df_day = df_day.with_columns(pl.lit(d.isoformat()).alias("data_ref"))
                all_weather_dfs.append(df_day)
                success_count += 1
                if success_count == 1:
                    print(f"  ✓ {d.isoformat()}: {df_day.height}h, colunas={df_day.columns}")
        except Exception as exc:
            print(f"  ✗ {d.isoformat()}: {exc}")
            fail_count += 1

    print(f"\n  Resultado: {success_count} dias OK, {fail_count} dias falha")

    if not all_weather_dfs:
        print("ERRO: nenhum dado de weather foi obtido")
        sys.exit(1)

    # Concatenar todos os dias
    weather_all = pl.concat(all_weather_dfs, how="vertical_relaxed")
    print(f"  Weather consolidado: {weather_all.height:,} linhas, {weather_all.width} colunas")

    # Salvar weather bruto
    weather_csv = output_dir / "weather_bruto.csv"
    weather_parquet = output_dir / "weather_bruto.parquet"
    weather_all.write_csv(weather_csv, separator=";")
    weather_all.write_parquet(weather_parquet)
    print(f"  Weather bruto salvo: {weather_csv}")
    print(f"  Weather parquet salvo: {weather_parquet}")

    # ── Step 4b: Weather no formato MDM (JSON columns por NIO) ────
    # Usar build_weather_dataframe para o primeiro dia como amostra
    print(f"\n  --- Weather MDM-style (primeiro dia como amostra) ---")
    first_day = start_date
    try:
        w_df = build_weather_dataframe(nios, first_day, lat, lon)
        if not w_df.is_empty():
            w_csv = output_dir / f"weather_mdm_{first_day.isoformat()}.csv"
            w_parquet = output_dir / f"weather_mdm_{first_day.isoformat()}.parquet"
            w_df.write_csv(w_csv, separator=";")
            w_df.write_parquet(w_parquet)
            print(f"  MDM-style salvo: {w_csv}")
            print(f"  Linhas: {w_df.height}, Colunas: {w_df.columns}")
            # Mostrar um exemplo do JSON
            json_col = [c for c in w_df.columns if c != "NIO" and c != "DIA"][0]
            print(f"  Exemplo coluna '{json_col}': {w_df[json_col][0][:80]}...")
        else:
            print(f"  ⚠ DataFrame MDM-style vazio para {first_day}")
    except Exception as exc:
        import traceback
        print(f"  ✗ build_weather_dataframe falhou: {exc}")
        traceback.print_exc()

    # ── Step 5: Enrich timegrid ───────────────────────────────────
    print(f"\n[5/5] Testando enrich_timegrid...")
    tg = _build_synthetic_timegrid(nios, today - timedelta(days=1), n_hours=3)
    print(f"  Timegrid sintético: {tg.height:,} linhas, {tg.width} colunas")

    try:
        enriched = enrich_timegrid(tg, weather_all)
        enriched_csv = output_dir / "timegrid_enriquecido.csv"
        enriched_parquet = output_dir / "timegrid_enriquecido.parquet"
        enriched.write_csv(enriched_csv, separator=";")
        enriched.write_parquet(enriched_parquet)
        print(f"  Enriquecido salvo: {enriched_csv}")
        print(f"  Linhas: {enriched.height}, Colunas: {enriched.width}")
        weather_cols = [c for c in enriched.columns if c.startswith("weather_")]
        print(f"  Colunas de weather adicionadas: {weather_cols}")
    except Exception as exc:
        print(f"  ✗ enrich_timegrid falhou: {exc}")

    # ── Resumo final ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESUMO")
    print(f"{'=' * 60}")
    print(f"  NIOs amostrados: {len(nios)}")
    print(f"  Coordenadas: {lat:.6f}, {lon:.6f}")
    print(f"  Período weather: {start_date} a {end_date} ({days}dias)")
    print(f"  Dias com dados: {success_count}")
    print(f"  Dias sem dados: {fail_count}")
    print(f"  Weather bruto: {weather_csv}")
    print(f"  Timegrid enriquecido: {enriched_csv}")
    print(f"\n  Saída completa em: {output_dir}")
    print(f"{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teste de integração do módulo weather com API real e dados do pipeline"
    )
    parser.add_argument("--days", type=int, default=30, help="Quantos dias de weather buscar")
    parser.add_argument("--nios", type=int, default=5, help="Quantos NIOs amostrar do CIS")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cis-csv", type=Path, default=None, help="Caminho do CSV CIS (default: mais recente)")

    args = parser.parse_args()

    cis_path = args.cis_csv or _find_latest_cis_csv()
    if cis_path is None or not cis_path.exists():
        print(f"Arquivo CIS não encontrado. Use --cis-csv para especificar.")
        return 1

    run_integration_test(
        cis_path=cis_path,
        output_dir=args.output_dir,
        n_nios=args.nios,
        days=args.days,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
