"""Main Entry Point for TCC Extraction, Model Training & TUI Application.

Usage:
    uv run main.py
"""

from __future__ import annotations

import sys
from src.tui.tui_app import main_tui, show_environment_warning


def main() -> None:
    # Display environment check warning if not run via uv or venv
    show_environment_warning()
    
    # Run interactive TUI
    main_tui()


if __name__ == "__main__":
    main()
