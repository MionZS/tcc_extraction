"""Weather data integration with Open-Meteo API.

Two usage modes
---------------
1. **Pipeline mode** — ``build_weather_dataframe()`` returns one row per
   NIO with JSON-string columns (matching the MDM output layout).
2. **Timegrid enrichment** — ``fetch_historical_weather()`` returns a
   DataFrame with one row per hour; ``enrich_timegrid()`` merges it into
   the 5-min timegrid.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import date
from typing import Any, cast

import polars as pl

logger = logging.getLogger(__name__)

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Variables fetched from Open-Meteo and turned into JSON columns
DEFAULT_WEATHER_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover",
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_json_column(hourly_values: list[float | None]) -> str:
    """Build a JSON string with 24 hourly keys (``HH:00`` → value).

    Example: ``{"00:00": 18.5, "01:00": 18.2, ..., "23:00": 15.1}``
    """
    obj: dict[str, float | None] = {}
    for hour in range(24):
        key = f"{hour:02d}:00"
        value = hourly_values[hour] if hour < len(hourly_values) else None
        obj[key] = value
    return json.dumps(obj, ensure_ascii=False)


def _format_weather_url(
    target_date: date,
    latitude: float,
    longitude: float,
    variables: list[str] | None = None,
) -> str:
    """Build the Open-Meteo Archive API URL (pure function, no I/O)."""
    if variables is None:
        variables = DEFAULT_WEATHER_VARS

    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": ",".join(variables),
        "timezone": "America/Sao_Paulo",
    }
    return f"{OPEN_METEO_BASE}?{urllib.parse.urlencode(params)}"  # type: ignore[arg-type]


def _build_empty_weather_frame(variables: list[str]) -> pl.DataFrame:
    """Return an empty DataFrame with the expected weather schema."""
    schema: dict[str, type[pl.DataType]] = {"hour": pl.Int64}
    for var in variables:
        schema[var] = pl.Float64
    return pl.DataFrame(schema=schema)


def _find_column(df: pl.DataFrame, name: str) -> str | None:
    """Find a column by name ignoring case."""
    for col in df.columns:
        if str(col).strip().upper() == name.upper():
            return col
    return None


# ── Pipeline mode: JSON-per-NIO ────────────────────────────────────────────


def fetch_weather_json(
    target_date: date,
    latitude: float,
    longitude: float,
    variables: list[str] | None = None,
) -> dict[str, str] | None:
    """Fetch hourly weather and return JSON strings per variable.

    Returns a dict mapping variable names to JSON strings (24 hourly
    keys each), or ``None`` on failure.
    """
    if variables is None:
        variables = DEFAULT_WEATHER_VARS
    url = _format_weather_url(target_date, latitude, longitude, variables)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Open-Meteo request failed: %s", exc)
        return None

    hourly = data.get("hourly")
    if not hourly:
        logger.warning("No hourly data in Open-Meteo response")
        return None

    result: dict[str, str] = {}
    for var in variables:
        values: list[float | None] = hourly.get(var, [])
        if not values:
            continue
        result[var] = _build_json_column(values)

    if not result:
        logger.warning("No weather variables retrieved")
        return None

    return result


def build_weather_dataframe(
    nios: list[str],
    report_day: date,
    latitude: float,
    longitude: float,
) -> pl.DataFrame:
    """Build a Polars DataFrame with one row per NIO and JSON column per var.

    All column names are UPPER_CASE for consistency with the rest of the
    pipeline (Oracle convention).

    Columns
    -------
    - ``NIO`` — join key
    - ``DIA`` — ISO date string
    - one JSON column per weather variable (e.g. ``TEMPERATURE_2M``)

    Returns an empty DataFrame with the correct schema if the API fails.
    """
    weather = fetch_weather_json(report_day, latitude, longitude)
    if weather is None:
        return pl.DataFrame(schema={
            "NIO": pl.String,
            "DIA": pl.String,
            **{var.upper(): pl.String for var in DEFAULT_WEATHER_VARS},
        })

    base_row: dict[str, Any] = {"DIA": report_day.isoformat()}
    base_row.update({k.upper(): v for k, v in weather.items()})

    rows: list[dict[str, Any]] = []
    for nio in nios:
        row = dict(base_row)
        row["NIO"] = nio
        rows.append(row)

    df = pl.DataFrame(rows, infer_schema_length=None)

    # Garantir que todas as colunas estejam UPPER_CASE
    rename = {c: c.upper() for c in df.columns if c != c.upper()}
    if rename:
        df = df.rename(rename)
    return df


# ── Timegrid enrichment mode: hourly DataFrame ──────────────────────────────


def fetch_historical_weather(
    target_date: date,
    latitude: float,
    longitude: float,
    variables: list[str] | None = None,
    *,
    timeout: int = 30,
) -> pl.DataFrame:
    """Fetch hourly weather and return a DataFrame with one row per hour.

    Columns: ``hour`` (int, 0-23) plus one column per variable.

    Returns an empty DataFrame with the correct schema on failure.
    """
    if variables is None:
        variables = DEFAULT_WEATHER_VARS

    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": ",".join(variables),
        "timezone": "America/Sao_Paulo",
    }
    url = f"{OPEN_METEO_BASE}?{urllib.parse.urlencode(params)}"  # type: ignore[arg-type]

    logger.info(
        "Fetching weather from Open-Meteo for %s at %.4f, %.4f",
        target_date.isoformat(),
        latitude,
        longitude,
    )

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Open-Meteo request failed: %s", exc)
        return _build_empty_weather_frame(variables)

    hourly = data.get("hourly")
    if hourly is None:
        logger.warning("Open-Meteo response missing 'hourly' key")
        return _build_empty_weather_frame(variables)

    times: list[str] = hourly.get("time", [])
    if not times:
        logger.warning("Open-Meteo returned no hourly data")
        return _build_empty_weather_frame(variables)

    rows: list[dict[str, Any]] = []
    for i, time_str in enumerate(times):
        try:
            hour = int(time_str.split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        row: dict[str, Any] = {"hour": hour}
        for var in variables:
            raw = hourly.get(var)
            if raw is not None and i < len(raw):
                row[var] = raw[i]
            else:
                row[var] = None
        rows.append(row)

    if not rows:
        return _build_empty_weather_frame(variables)

    return pl.DataFrame(rows)


def enrich_timegrid(
    timegrid_df: pl.DataFrame,
    weather_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge hourly weather data into the 5-min timegrid DataFrame.

    Each 5-min slot is mapped to its corresponding hour (0-23).
    Weather columns are prefixed with ``weather_``.
    """
    if weather_df.is_empty() or "hour" not in weather_df.columns:
        return timegrid_df

    if "DATA TIME" not in timegrid_df.columns:
        logger.warning("timegrid DataFrame has no 'DATA TIME' column; skipping enrichment")
        return timegrid_df

    # Parse "DATA TIME" (format: "DD/MM/YYYY HH24:MI:SS") and extract hour
    parsed = timegrid_df.with_columns(
        pl.col("DATA TIME")
        .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False)
        .dt.hour()
        .alias("__hour__")
    )

    weather_rename = {
        col: f"weather_{col}"
        for col in weather_df.columns
        if col != "hour"
    }
    weather_renamed = weather_df.rename(weather_rename)

    enriched = parsed.join(
        weather_renamed,
        left_on="__hour__",
        right_on="hour",
        how="left",
    ).drop("__hour__")

    return enriched


# ── Coordinates ─────────────────────────────────────────────────────────────


def get_coordinates_from_cis(cis_df: pl.DataFrame) -> tuple[float, float] | None:
    """Extract representative (latitude, longitude) from the CIS DataFrame.

    Uses the median of all LAT / LONG values.  Returns ``None`` if the
    required columns are missing or empty.
    """
    lat_col = _find_column(cis_df, "LAT")
    lon_col = _find_column(cis_df, "LONG")

    if lat_col is None or lon_col is None:
        logger.warning("CIS DataFrame missing LAT / LONG columns")
        return None

    lat_series = cis_df[lat_col].drop_nulls()
    lon_series = cis_df[lon_col].drop_nulls()

    if lat_series.is_empty() or lon_series.is_empty():
        return None

    lat_median_val = lat_series.cast(pl.Float64).median()
    lon_median_val = lon_series.cast(pl.Float64).median()
    lat_median = float(cast(float, lat_median_val))
    lon_median = float(cast(float, lon_median_val))

    return lat_median, lon_median
