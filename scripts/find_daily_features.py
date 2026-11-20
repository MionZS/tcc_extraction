import os
from pathlib import Path

for root, dirs, files in os.walk("src"):
    for f in files:
        if f.endswith(".py"):
            p = Path(root) / f
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "build_daily_features" in text or "extract_day" in text or "uc_day_features" in text:
                print("Found in:", p)
