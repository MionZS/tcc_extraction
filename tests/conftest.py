"""conftest.py — Fixtures compartilhadas para testes do TCC pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Cria um CSV de exemplo com cabeçalho e 3 linhas."""
    path = tmp_path / "araucaria_cis_20260615.csv"
    path.write_text(
        "UC;NIO;SITUACAO\n"
        "UC001;NIO001;ATIVA\n"
        "UC002;NIO002;ATIVA\n"
        "UC003;NIO003;INATIVA\n",
        encoding="utf-8-sig",
    )
    return path


@pytest.fixture
def dated_cis_dir(tmp_path: Path) -> Path:
    """Cria diretório CIS com arquivos datados."""
    d = tmp_path / "raw" / "CIS"
    d.mkdir(parents=True)
    for i in range(1, 11):
        dt = date(2026, 6, 15 - (10 - i))  # 6/6 to 6/15
        p = d / f"araucaria_cis_{dt:%Y%m%d}.csv"
        p.write_text(
            "UC;NIO;SITUACAO\nUC001;NIO001;ATIVA\n",
            encoding="utf-8-sig",
        )
    return d


@pytest.fixture
def dated_orca_dir(tmp_path: Path) -> Path:
    """Cria diretório ORCA com arquivos datados."""
    d = tmp_path / "raw" / "ORCA"
    d.mkdir(parents=True)
    for i in range(1, 11):
        dt = date(2026, 6, 15 - (10 - i))
        p = d / f"araucaria_mdm_{dt:%Y%m%d}.csv"
        p.write_text(
            "UC;NIO;DATA_LEITURA;VALOR_KWH\nUC001;NIO001;2026-06-10;123.4\n",
            encoding="utf-8-sig",
        )
    return d


@pytest.fixture
def output_tree(tmp_path: Path) -> Path:
    """Cria a árvore completa de output esperada."""
    for sub in ["raw/CIS", "raw/ORCA", "refined/reports", "raw/CIS/daily_cadastrados"]:
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path
