import os
from pathlib import Path

for root, dirs, files in os.walk("src/features"):
    for f in files:
        print(Path(root) / f)
