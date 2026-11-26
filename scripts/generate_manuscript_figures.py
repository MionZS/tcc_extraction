"""CLI reproduzível para geração das figuras, metadados e fragmentos LaTeX do TCC.

Uso:
    python scripts/generate_manuscript_figures.py
    python scripts/generate_manuscript_figures.py --only ml
    python scripts/generate_manuscript_figures.py --only cases
    python scripts/generate_manuscript_figures.py --only platform
    python scripts/generate_manuscript_figures.py --skip-playwright
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.tcc_evidence.manuscript_figures import (
    MANUSCRIPT_FIG_DIR,
    SUBDIRS,
    _convert_rich_svgs_to_pdf,
    _load_suite_runs,
    generate_all_manuscript_figures,
    generate_controlled_ml_panels,
    generate_platform_interface_captures,
    generate_real_case_study_figures,
    write_latex_snippets_and_catalog,
)
from src.utils.censorship import build_feeder_anonymizer


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _collect_existing_metas() -> list[dict]:
    metas: list[dict] = []
    for key in ("ml", "cases", "platform"):
        subdir = SUBDIRS[key]
        if subdir.exists():
            for meta_file in sorted(subdir.glob("*.meta.json")):
                metas.append(json.loads(meta_file.read_text(encoding="utf-8")))
    return metas


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Gera figuras reproduzíveis, metadados e snippets LaTeX para TCC_AMI_Consolidado.pdf."
    )
    parser.add_argument(
        "--only",
        choices=["article", "all", "ml", "cases", "platform"],
        default="article",
        help="Artigo com dados reais (padrão). all e ml reproduzem o material didático histórico.",
    )
    parser.add_argument(
        "--skip-playwright",
        action="store_true",
        help="Pula as capturas da plataforma via Rich SVG + Playwright Chrome.",
    )
    args = parser.parse_args(argv)

    if args.only == "article":
        from scripts.generate_final_article_figures import generate_article_figures

        generate_article_figures()
        return 0

    if args.only == "all" and not args.skip_playwright:
        res = generate_all_manuscript_figures()
        print(f"Catálogo gerado em: {res['catalog']}")
        return 0

    for p in SUBDIRS.values():
        p.mkdir(parents=True, exist_ok=True)

    run_dirs = _load_suite_runs()
    oot_df = pl.read_parquet(run_dirs["oot_multiview"] / "decisions.parquet")
    raw_feeders = [
        str(x)
        for x in oot_df.get_column("meta__feeder").drop_nulls().unique().to_list()
        if str(x).strip()
    ]
    feeder_map = build_feeder_anonymizer(raw_feeders)

    if args.only in ("all", "ml"):
        print("[ML] Gerando painéis metodológicos controlados (Seção 5.1)...")
        generate_controlled_ml_panels()

    if args.only in ("all", "cases"):
        print("[CASES] Gerando séries e reconciliações das UCs reais mascaradas...")
        generate_real_case_study_figures(run_dirs, feeder_map)

    if args.only in ("all", "platform") and not args.skip_playwright:
        print("[PLATFORM] Gerando TUI Rich SVG/PDF e capturas Playwright do Cockpit Web...")
        generate_platform_interface_captures(run_dirs, feeder_map)
        _convert_rich_svgs_to_pdf(SUBDIRS["platform"])

    all_metas = _collect_existing_metas()
    write_latex_snippets_and_catalog(run_dirs, all_metas)
    catalog_path = MANUSCRIPT_FIG_DIR / "figure_catalog_manifest.json"
    print(f"[CONCLUÍDO] {len(all_metas)} metadados indexados em {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
