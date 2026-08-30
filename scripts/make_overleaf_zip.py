"""Package paper/ as an Overleaf-ready zip. OWNER: Person C.

    python scripts/make_overleaf_zip.py

Writes pemwe-paper-overleaf.zip containing only what the build needs, with forward-slash
paths (a zip written on Windows with backslash separators fails to unpack correctly on
Overleaf). Build artefacts and the notes file are excluded -- REFERENCES.md documents what
each citation is for and is not part of the document.
"""
import sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "pemwe-paper-overleaf.zip"

def main():
    sync = ROOT / "scripts" / "sync_paper_figures.py"
    files = [PAPER / n for n in ("main.tex", "refs.bib", "README.md")]
    files += sorted((PAPER / "figures").glob("*.pdf"))
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"missing: {missing}"); return 1
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = f.relative_to(PAPER).as_posix()      # forward slashes, always
            z.write(f, arc)
            print(f"  + {arc}")
    print(f"\n-> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"   run `python {sync.relative_to(ROOT).as_posix()} --check` first if figures changed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
