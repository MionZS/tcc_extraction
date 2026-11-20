import os
from pathlib import Path

for root, dirs, files in os.walk("src"):
    for f in files:
        if f.endswith(".py"):
            p = Path(root) / f
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "classified_ucs_latest" in text or "classified_ucs" in text:
                print("src:", p)

for root, dirs, files in os.walk("scripts"):
    for f in files:
        if f.endswith(".py"):
            p = Path(root) / f
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "classified_ucs_latest" in text or "classified_ucs" in text:
                print("scripts:", p)
