"""Self-overlap check: does the submitted text repeat long runs from files published in the repo?

Not a substitute for Turnitin. It answers one specific question: if a similarity checker indexes the
public repository, which passages of the paper would it match, and how long are they?
"""
import re
import glob
import os
from pypdf import PdfReader

PDF = "D:/D Drive/rl/paper/build/INDIACom2027_Track2_paper_named.pdf"
ROOT = "D:/D Drive/rl/"


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


paper_words = norm(" ".join((p.extract_text() or "") for p in PdfReader(PDF).pages)).split()
print("paper words:", len(paper_words))

N = 8  # an 8-word run repeated verbatim is what a checker highlights
paper_ngrams = {}
for i in range(len(paper_words) - N + 1):
    paper_ngrams.setdefault(" ".join(paper_words[i:i + N]), []).append(i)

# every text file published in the repo, excluding the paper sources themselves
targets = []
for pat in ("*.md", "audit/*.md", "paper/*.md", "notes/*.md", "briefs/*.md", "kaggle/*.md", "results_batch/*.md"):
    targets += glob.glob(ROOT + pat)
targets = [t for t in targets if os.path.basename(t) not in ("main_track2.tex",)]

hits = {}
for f in targets:
    try:
        words = norm(open(f, encoding="utf-8", errors="replace").read()).split()
    except Exception:
        continue
    for i in range(len(words) - N + 1):
        g = " ".join(words[i:i + N])
        if g in paper_ngrams:
            hits.setdefault(os.path.relpath(f, ROOT), set()).add(g)

print("\nfiles in the repo sharing >=%d-word runs with the paper:" % N)
if not hits:
    print("  none")
total = set()
for f, gs in sorted(hits.items(), key=lambda kv: -len(kv[1])):
    print("  %-44s %3d matching runs" % (f, len(gs)))
    total |= gs

# how much of the paper is covered by those runs
covered = set()
for g in total:
    for i in paper_ngrams[g]:
        covered.update(range(i, i + N))
print("\nshare of paper words inside a repeated run: %.1f%%" % (100.0 * len(covered) / max(len(paper_words), 1)))
