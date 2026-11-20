import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

nov_dir = Path("output/features/uc_day_features/feature_set_version=v1/report_year=2026/report_month=11")
print(f"Days in {nov_dir}:")
if nov_dir.exists():
    for d in sorted(nov_dir.iterdir()):
        files = list(d.glob("*.parquet"))
        print(f"  {d.name}: {len(files)} files")
else:
    print("Not found")

# Also check measurements for November 2026
for sub in ["ami_interval", "ami_registers", "ami_instantaneous"]:
    m_dir = Path(f"output/measurements/{sub}/report_year=2026/report_month=11")
    if m_dir.exists():
        days = sorted([d.name for d in m_dir.iterdir()])
        print(f"Measurements {sub} Nov days: {days}")
