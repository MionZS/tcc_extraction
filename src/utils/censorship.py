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
from typing import Iterable


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


def censor_nio_id(nio: str | int | None, keep_prefix: int = 3, keep_suffix: int = 2) -> str:
    """Censura o identificador NIO removendo zeros à esquerda e mascarando dígitos centrais.

    Exemplo:
        '0043570463' -> 'NIO-435***63'
        '42312054'   -> 'NIO-423***54'
    """
    if nio is None:
        return "-"
    s = str(nio).strip()
    if not s or s == "-":
        return "-"
    stripped = s.lstrip("0") or s
    if len(stripped) <= (keep_prefix + keep_suffix):
        return f"NIO-{stripped[:2]}***"
    masked_len = len(stripped) - keep_prefix - keep_suffix
    return f"NIO-{stripped[:keep_prefix]}{'*' * masked_len}{stripped[-keep_suffix:]}"


def build_feeder_anonymizer(feeders: Iterable[str | None]) -> dict[str, str]:
    """Cria mapeamento determinístico de alimentadores reais para códigos neutros (Circuito F-01, ...)."""
    neutral_passthru = {
        "outros": "Outros",
        "sem alimentador": "Sem Alimentador",
        "none": "Sem Alimentador",
        "-": "-",
    }
    clean_real: set[str] = set()
    mapping: dict[str, str] = {}
    for raw in feeders:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        low = s.lower()
        if low in neutral_passthru:
            mapping[s] = neutral_passthru[low]
        else:
            clean_real.add(s)

    for idx, feeder_name in enumerate(sorted(clean_real), start=1):
        mapping[feeder_name] = f"Circuito F-{idx:02d}"
    return mapping


def sanitize_public_text(text: str, feeder_map: dict[str, str] | None = None) -> str:
    """Remove menções identificadoras (concessionária, município, alimentadores e caminhos locais)."""
    if not text:
        return text
    out = text

    # Substituir alimentadores do maior para o menor para evitar colisões parciais
    if feeder_map:
        for raw_feeder in sorted(feeder_map.keys(), key=len, reverse=True):
            anon_feeder = feeder_map[raw_feeder]
            if raw_feeder and raw_feeder != anon_feeder:
                out = re.sub(re.escape(raw_feeder), anon_feeder, out, flags=re.IGNORECASE)

    # Remover caminhos locais absolutos e usuário do sistema
    out = re.sub(r"[A-Za-z]:\\+Projects\\+tcc_extraction\\+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[A-Za-z]:/+Projects/+tcc_extraction/+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[A-Za-z]:\\+Uni\\+tcc_pipeline_simple\\+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[A-Za-z]:/+Uni/+tcc_pipeline_simple/+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\be806586\b", "usuario_local", out, flags=re.IGNORECASE)

    # Anonimizar município e distribuidora
    replacements = [
        (r"COPEL\s+ARAUC[ÁA]RIA", "REDE DE DISTRIBUIÇÃO (SUÍTE OOT)"),
        (r"Copel\s+Arauc[áa]ria", "Distribuidora (Município Avaliado)"),
        (r"araucaria_cis", "cis_municipio"),
        (r"ARAUC[ÁA]RIA", "MUNICÍPIO AVALIADO"),
        (r"Arauc[áa]ria", "Município Avaliado"),
        (r"araucaria", "municipio_avaliado"),
        (r"\bCOPEL\b", "DISTRIBUIDORA"),
        (r"\bCopel\b", "Distribuidora"),
        (r"\bcopel\b", "distribuidora"),
        (r"Óleos\s+Brasil|Oleos\s+Brasil", "Circuito F-XX"),
    ]
    for pat, repl in replacements:
        out = re.sub(pat, repl, out)

    return out


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

