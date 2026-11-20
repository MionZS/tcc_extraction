import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

# search for how uc_day_features are written in src/pipeline.py
p = Path("src/pipeline.py")
if p.exists():
    text = p.read_text(encoding="utf-8")
    for line in text.splitlines():
        if any(k in line for k in ["day_features", "extract_features", "build_features", "uc_day_features"]):
            print("pipeline.py:", line)
