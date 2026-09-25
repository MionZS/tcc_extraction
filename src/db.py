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


import os

def _load_env_file() -> None:
    """Load key-value pairs from .env if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _load_config() -> dict:
    """Load the JSON config file used for database credentials."""
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_db_config(target: DatabaseTarget = "orca") -> dict[str, str]:
    """Read and validate the requested DB config from env vars or config.json."""
    _load_env_file()
    config_key = _TARGET_CONFIG_KEYS.get(target)
    if config_key is None:
        raise ValueError(f"Invalid database target: {target}. Use 'orca', 'cis', or 'geo'.")

    # 1. Try environment variables
    env_user = os.getenv(f"{config_key}_USER") or os.getenv(f"ORACLE_{config_key}_USER")
    env_pass = os.getenv(f"{config_key}_PASSWORD") or os.getenv(f"ORACLE_{config_key}_PASSWORD")
    env_host = os.getenv(f"{config_key}_HOST") or os.getenv(f"ORACLE_{config_key}_HOST")
    env_port = os.getenv(f"{config_key}_PORT") or os.getenv(f"ORACLE_{config_key}_PORT")
    env_service = (
        os.getenv(f"{config_key}_SERVICE_NAME")
        or os.getenv(f"{config_key}_SERVICE")
        or os.getenv(f"ORACLE_{config_key}_SERVICE_NAME")
        or os.getenv(f"ORACLE_{config_key}_SERVICE")
    )

    if env_user and env_pass and env_host and env_port and env_service:
        return {
            "user": env_user,
            "password": env_pass,
            "host": env_host,
            "port": str(env_port),
            "service_name": env_service,
        }

    # 2. Fall back to config.json
    raw_config = _load_config()
    oracle_block = raw_config.get("oracle", {}) if isinstance(raw_config, dict) else {}
    db_config = oracle_block.get(config_key, {}) if isinstance(oracle_block, dict) else {}

    service_name = db_config.get("service_name") or db_config.get("service") or env_service
    normalized = {
        "user": env_user or db_config.get("user"),
        "password": env_pass or db_config.get("password"),
        "host": env_host or db_config.get("host"),
        "port": env_port or db_config.get("port"),
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
            f"Set environment variables ({config_key}_USER, etc.) or check {CONFIG_PATH.name}."
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
