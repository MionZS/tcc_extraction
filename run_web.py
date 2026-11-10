"""
Script para inicialização imediata do Cockpit Analítico AMI Web (estilo VETOR / SanPlat AMI).
Abre automaticamente o navegador padrão do operador em http://localhost:8050.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Adiciona raiz do projeto ao sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_web_cockpit import run_web_cockpit

if __name__ == "__main__":
    run_web_cockpit(port=8050, open_browser=True)

