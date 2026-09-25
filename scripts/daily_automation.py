"""CLI Entrypoint for Daily Automation, Regression Testing, and Health Auditing.

Modes:
- 'health-check': Inspect data lake partitions and generate daily_health_report.md
- 'test-only': Execute unit tests and regressions programmatically
- 'feeder': Execute extraction and modeling pipeline for a specific feeder
- 'full': Run health-check + tests + feeder pipeline

Usage:
    python scripts/daily_automation.py --mode health-check
    python scripts/daily_automation.py --mode test-only
    python scripts/daily_automation.py --mode feeder --feeder-name Fonte_Nova
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.automation.daily_runner import DailyAutomationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Automation & Health Auditing CLI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["health-check", "test-only", "feeder", "full"],
        default="health-check",
        help="Operation mode (default: health-check)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory",
    )
    parser.add_argument(
        "--feeder-name",
        type=str,
        default="Fonte_Nova",
        help="Feeder name for 'feeder' or 'full' mode",
    )
    parser.add_argument(
        "--feeder-geo-id",
        type=int,
        default=6352460,
        help="Feeder GEO ID (default: 6352460 - Fonte Nova)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print(f"DAILY PIPELINE AUTOMATION: Mode [{args.mode.upper()}]")
    print("=" * 60)

    runner = DailyAutomationRunner(base_dir=args.output_dir)

    # 1. Health Check
    if args.mode in ("health-check", "full"):
        print("\n[*] Running Data Lake Health Audit...")
        result = runner.run_daily_audit()
        print(f"[+] Global Data Lake Health: {result.get('global_status')}")

    # 2. Test execution
    if args.mode in ("test-only", "full"):
        print("\n[*] Running Test Suite programmatically via pytest...")
        import pytest
        test_args = ["tests", "-q", "--disable-warnings"]
        ret = pytest.main(test_args)
        if ret == 0:
            print("[+] All automated tests passed successfully.")
        else:
            print(f"[-] Automated tests finished with returncode {ret}.")

    # 3. Feeder pipeline execution
    if args.mode in ("feeder", "full"):
        print(f"\n[*] Triggering Feeder Pipeline for '{args.feeder_name}' (GEO ID {args.feeder_geo_id})...")
        from scripts.run_feeder_pipeline import main as run_feeder
        old_argv = sys.argv
        sys.argv = [
            "run_feeder_pipeline.py",
            "--feeder-name", args.feeder_name,
            "--feeder-geo-id", str(args.feeder_geo_id),
            "--output-dir", str(args.output_dir),
            "--train-model",
        ]
        try:
            run_feeder()
        finally:
            sys.argv = old_argv

    print("\n" + "=" * 60)
    print("AUTOMATION WORKFLOW COMPLETED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
