"""Database connection helpers for the ARAUCARIA daily pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

import oracledb
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

DatabaseTarget = Literal["orca", "cis", "geo"]
ConfigKey = Literal["ORCA", "CIS", "GEO"]

oracledb.defaults.fetch_lobs = False

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

_TARGET_CONFIG_KEYS: dict[DatabaseTarget, ConfigKey] = {
    "orca": "ORCA",
    "cis": "CIS",
    "geo": "GEO",
}


def _load_config() -> dict:
    """Load the JSON config file used for database credentials."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}. "
            "Create it from config_example.json."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_db_config(target: DatabaseTarget = "orca") -> dict[str, str]:
    """Read and validate the requested DB config from config.json."""
    config_key = _TARGET_CONFIG_KEYS.get(target)
    if config_key is None:
        raise ValueError(f"Invalid database target: {target}. Use 'orca', 'cis', or 'geo'.")

    raw_config = _load_config()
    oracle_block = raw_config.get("oracle")
    if not isinstance(oracle_block, dict):
        raise ValueError("Invalid config.json: missing top-level 'oracle' object.")

    db_config = oracle_block.get(config_key)
    if not isinstance(db_config, dict):
        raise ValueError(
            f"Invalid config.json: missing oracle.{config_key} configuration."
        )

    service_name = db_config.get("service_name") or db_config.get("service")
    normalized = {
        "user": db_config.get("user"),
        "password": db_config.get("password"),
        "host": db_config.get("host"),
        "port": db_config.get("port"),
        "service_name": service_name,
    }

    missing = [
        key
        for key, value in normalized.items()
        if value is None or str(value).strip() == ""
    ]

    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required config values for target '{target}': {joined}. "
            f"Check {CONFIG_PATH.name}."
        )

    return {
        "user": str(normalized["user"]),
        "password": str(normalized["password"]),
        "host": str(normalized["host"]),
        "port": str(normalized["port"]),
        "service_name": str(normalized["service_name"]),
    }



def create_engine(target: DatabaseTarget = "orca") -> Engine:
    """Create a new SQLAlchemy engine connected to ORCA, CIS, or GEO Oracle DB."""
    config = _read_db_config(target)

    user = quote_plus(config["user"])
    password = quote_plus(config["password"])
    host = config["host"]
    port = config["port"]
    service = config["service_name"]

    url = f"oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={service}"

    return _sa_create_engine(
        url,
        pool_pre_ping=True,
        poolclass=NullPool,
    )
