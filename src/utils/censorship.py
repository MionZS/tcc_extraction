"""
src/utils/censorship.py
-------------------------------------------------------------------------------
Funções de anonimização e censura de dados sensíveis (LGPD e Compliance COPEL).
Censura identificadores únicos de clientes como UC (Unidade Consumidora) e NIO.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re


def censor_uc_id(uc: str | int | None, keep_prefix: int = 3, keep_suffix: int = 2) -> str:
    """
    Censura o número da UC preservando apenas o início e o fim para depuração.
    Exemplo:
        '100000070' -> '100****70'
        '100027687' -> '100****87'
    Se o identificador for curto ou nulo, retorna uma máscara genérica.
    """
    if uc is None:
        return "UC-ANON"
    s = str(uc).strip()
    if not s or s == "-":
        return "-"
    if len(s) <= (keep_prefix + keep_suffix):
        return f"{s[:2]}***"
    
    masked_len = len(s) - keep_prefix - keep_suffix
    return f"{s[:keep_prefix]}{'*' * masked_len}{s[-keep_suffix:]}"


def pseudonymize_uc(uc: str | int | None, prefix: str = "UC") -> str:
    """
    Gera um pseudônimo reprodutível e totalmente descaracterizado via hash.
    Exemplo:
        '100000070' -> 'UC-A8F2'
        '100027687' -> 'UC-3D19'
    """
    if uc is None:
        return f"{prefix}-XXXX"
    s = str(uc).strip()
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:4].upper()
    return f"{prefix}-{h}"

