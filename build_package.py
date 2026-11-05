"""Regenerate the pai-lang package from the project root.

Run:  python build_package.py

pai.py, plugins/ai.py and PAI_GUIDE.md in this folder are GENERATED copies of the
originals in the project root. Edit the originals, then run this to refresh the
package — never hand-edit the copies here, or they'll silently drift.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BANNER = "# GENERATED from ../{name} by build_package.py — edit the original, not this copy.\n"


def copy(rel, banner=True):
    src = ROOT / rel
    dst = HERE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    if banner and rel.endswith(".py"):
        text = BANNER.format(name=rel) + text
    dst.write_text(text, encoding="utf-8")
    print(f"generated {rel}")


for rel in ("pai.py", "plugins/ai.py", "PAI_GUIDE.md"):
    copy(rel, banner=not rel.endswith(".md"))
print("pai-lang package is up to date.")
