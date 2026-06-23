"""Weather data integration with Open-Meteo API.

Fetches historical hourly weather data and formats it as JSON columns
matching the MDM output layout (one row per NIO, JSON per variable).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from datetime import date
from typing import Any

import polars as pl

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Variables that will be fetched and turned into JSON columns
DEFAULT_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover",
]


def _build_json_column(hourly_values: list[float | None]) -> str:
    """Build a JSON string with 24 hourly keys, matching MDM's ``HH24:MI`` format.

    Example output: ``{"00:00": 18.5, "01:00": 18.2, ..., "23:00": 15.1}``
    """
    obj: dict[str, float | None] = {}
    for hour in range(24):
        key = f"{hour:02d}:00"
        value = hourly_values[hour] if hour < len(hourly_values) else None
        obj[key] = value
    return json.dumps(obj, ensure_ascii=False)


def fetch_weather_json(
    target_date: date,
    latitude: float,
    longitude: float,
    variables: list[str] | None = None,
) -> dict[str, str] | None:
    """Fetch hourly weather from Open-Meteo Historical API and return JSON strings.

    Returns a dict mapping variable names to JSON strings, or ``None`` on failure.
    Each JSON string has 24 hourly keys (``HH:00`` → value).
    """
    if variables is None:
        variables = DEFAULT_WEATHER_VARS

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": ",".join(variables),
        "timezone": "America/Sao_Paulo",
    }

    url = f"{OPEN_METEO_BASE}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[weather] Open-Meteo request failed: {exc}")
        return None

    hourly = data.get("hourly")
    if not hourly:
        print("[weather] No hourly data in Open-Meteo response")
        return None

    times: list[str] = hourly.get("time", [])
    result: dict[str, str] = {}
    for var in variables:
        values: list[float | None] = hourly.get(var, [])
        if not values:
            continue
        result[var] = _build_json_column(values)

    if not result:
        print("[weather] No weather variables retrieved")
        return None

    return result


def build_weather_dataframe(
    nios: list[str],
    report_day: date,
    latitude: float,
    longitude: float,
) -> pl.DataFrame:
    """Build a Polars DataFrame with weather data in MDM-like JSON format.

    Returns one row per NIO with columns:
      - ``NIO``
      - ``dia``  (ISO date string)
      - one JSON column per weather variable

    If the API call fails, returns an empty DataFrame.
    """
    weather = fetch_weather_json(report_day, latitude, longitude)
    if weather is None:
        return pl.DataFrame(schema={
            "NIO": pl.String,
            "dia": pl.String,
            **{var: pl.String for var in DEFAULT_WEATHER_VARS},
        })

    # A single weather-data row
    base_row: dict[str, Any] = {"dia": report_day.isoformat()}
    base_row.update(weather)

    # Replicate for every NIO
    rows: list[dict[str, Any]] = []
    for nio in nios:
        row = dict(base_row)
        row["NIO"] = nio
        rows.append(row)

    return pl.DataFrame(rows, infer_schema_length=None)


import json
import logging
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Base URL for the Open-Meteo Historical Weather API
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Default set of hourly weather variables
DEFAULT_WEATHER_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
]


def fetch_historical_weather(
    target_date: date,
    latitude: float,
    longitude: float,
    variables: list[str] | None = None,
    *,
    timeout: int = 30,
) -> pl.DataFrame:
    """Fetch hourly weather data from Open-Meteo for a single date.

    Returns a Polars DataFrame with columns ``hour`` (int, 0-23) plus one
    column per requested variable (e.g. ``temperature_2m``, ``precipitation``).

    Parameters
    ----------
    target_date:
        The date to fetch weather for.
    latitude, longitude:
        WGS84 coordinates of the location.
    variables:
        List of hourly weather variable names.  Defaults to
        :const:`DEFAULT_WEATHER_VARS`.
    timeout:
        HTTP request timeout in seconds.
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

    Returns a new DataFrame with weather columns added.
    """
    if weather_df.is_empty() or "hour" not in weather_df.columns:
        return timegrid_df

    if "Data Time" not in timegrid_df.columns:
        logger.warning("timegrid DataFrame has no 'Data Time' column; skipping enrichment")
        return timegrid_df

    # Parse "Data Time" (format: "DD/MM/YYYY HH24:MI:SS") and extract hour
    parsed = timegrid_df.with_columns(
        pl.col("Data Time")
        .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False)
        .dt.hour()
        .alias("__hour__")
    )

    # Prefix weather columns (except the join key 'hour')
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


def _build_empty_weather_frame(variables: list[str]) -> pl.DataFrame:
    """Return an empty DataFrame with the expected weather schema."""
    schema: dict[str, type[pl.DataType]] = {"hour": pl.Int64}
    for var in variables:
        schema[var] = pl.Float64
    return pl.DataFrame(schema=schema)


def get_coordinates_from_cis(cis_df: pl.DataFrame) -> tuple[float, float] | None:
    """Extract a representative (latitude, longitude) from the CIS DataFrame.

    Uses the median of all LAT / LONG values.  Returns ``None`` if the
    required columns are missing or empty.
    """
    # Look for LAT / LONG columns (case-insensitive)
    lat_col = _find_column(cis_df, "LAT")
    lon_col = _find_column(cis_df, "LONG")

    if lat_col is None or lon_col is None:
        logger.warning("CIS DataFrame missing LAT / LONG columns")
        return None

    lat_series = cis_df[lat_col].drop_nulls()
    lon_series = cis_df[lon_col].drop_nulls()

    if lat_series.is_empty() or lon_series.is_empty():
        return None

    # Cast to float in case they come as strings
    lat_median = float(lat_series.cast(pl.Float64).median())
    lon_median = float(lon_series.cast(pl.Float64).median())

    return lat_median, lon_median


def _find_column(df: pl.DataFrame, name: str) -> str | None:
    """Find a column by name ignoring case."""
    for col in df.columns:
        if str(col).strip().upper() == name.upper():
            return col
    return None
