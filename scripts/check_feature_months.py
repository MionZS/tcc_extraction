import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

for m in [8, 9, 10, 11]:
    d = Path(f"output/features/uc_day_features/feature_set_version=v1/report_year=2026/report_month={m:02d}")
    if d.exists():
        days = sorted([p.name.replace("report_day=", "") for p in d.iterdir() if p.is_dir()])
        print(f"Month {m:02d}: {len(days)} days ({days[0]} to {days[-1]})")
    else:
        print(f"Month {m:02d}: not found")
