import re
from pathlib import Path

text = Path("src/datasets/retroactive_cleaner.py").read_text(encoding="utf-8")
for m in re.finditer(r"def ([a-zA-Z0-9_]+)\(", text):
    print("def", m.group(1))
