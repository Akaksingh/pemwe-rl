"""Copy the generated figures into paper/figures/ and report drift. OWNER: Person C.

    python scripts/sync_paper_figures.py           # copy + report
    python scripts/sync_paper_figures.py --check   # report only, non-zero if stale

`results/figures/` is the canonical output of `pemwe.plots` (CONTRACTS.md 4).
`paper/figures/` is a copy, and it exists for one reason: a LaTeX project has to be
self-contained. Overleaf, arXiv and most submission systems take a single directory tree
with no parent, so `\\graphicspath{{../results/figures/}}` cannot resolve there.

A copy can go stale, which is worse than no copy -- a paper that silently builds against
last week's figures is a wrong paper that compiles. Hence --check, and hence the explicit
report of which files differ rather than a silent overwrite.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "figures"
DST = ROOT / "paper" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if stale")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"no {SRC.relative_to(ROOT)} -- run scripts/make_paper_figures.py first")
        return 1
    DST.mkdir(parents=True, exist_ok=True)

    stale, new, same = [], [], []
    for f in sorted(SRC.glob("*.pdf")):
        t = DST / f.name
        if not t.exists():
            new.append(f.name)
        elif not filecmp.cmp(f, t, shallow=False):
            stale.append(f.name)
        else:
            same.append(f.name)

    # a figure in the paper with no source is a figure nobody can regenerate
    orphan = [t.name for t in sorted(DST.glob("*.pdf")) if not (SRC / t.name).exists()]

    for n in new:
        print(f"  NEW    {n}")
    for n in stale:
        print(f"  STALE  {n}")
    for n in orphan:
        print(f"  ORPHAN {n}  (in paper/figures but not generated -- delete or regenerate)")
    print(f"  {len(same)} up to date")

    if args.check:
        bad = len(new) + len(stale) + len(orphan)
        print("\npaper/figures is CURRENT" if bad == 0 else
              f"\npaper/figures is STALE: {bad} file(s) differ -- run without --check")
        return 1 if bad else 0

    for n in new + stale:
        shutil.copy2(SRC / n, DST / n)
    if new or stale:
        print(f"\ncopied {len(new) + len(stale)} file(s) into paper/figures/")
    else:
        print("\nnothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
