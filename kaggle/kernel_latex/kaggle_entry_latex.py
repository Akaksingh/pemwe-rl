"""Kaggle kernel (CPU): compile paper/main_track2.tex with TeX Live and report page count, errors and page images.

Push with `kaggle kernels push -p kaggle/kernel_latex`. Builds two PDFs from the same source: the named version
(main_track2.pdf) and an anonymised copy for double-blind review (main_track2_anon.pdf, produced by switching the
\\anonfalse flag to \\anontrue). Output: /kaggle/working/latex_out/ (PDFs, logs, page PNGs, summary.txt).
The repo clone is removed afterwards.
"""
import re
import shutil
import subprocess
from pathlib import Path

URL = "https://github.com/Akaksingh/pemwe-rl.git"
WORK = Path("/kaggle/working")
REPO = WORK / "repo"
OUT = WORK / "latex_out"
OUT.mkdir(parents=True, exist_ok=True)
summary = []


def sh(cmd, cwd=None, check=False):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return r


shutil.rmtree(REPO, ignore_errors=True)
sh(f"git clone --depth 1 {URL} {REPO}", check=True)
summary.append("commit: " + sh(f"git -C {REPO} rev-parse HEAD").stdout.strip())

pkgs = ("texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended "
        "texlive-publishers texlive-science texlive-bibtex-extra poppler-utils")
r = sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends " + pkgs)
summary.append(f"apt install exit code: {r.returncode}")
print("apt exit", r.returncode, flush=True)

paper = REPO / "paper"
# anonymised copy: the source has \anonfalse; switch it to \anontrue
src = (paper / "main_track2.tex").read_text(encoding="utf-8")
BS = chr(92)
assert BS + "anonfalse" in src
(paper / "main_track2_anon.tex").write_text(src.replace(BS + "anonfalse", BS + "anontrue", 1), encoding="utf-8")


def build(name):
    for cmd in (f"pdflatex -interaction=nonstopmode {name}", f"bibtex {name}",
                f"pdflatex -interaction=nonstopmode {name}", f"pdflatex -interaction=nonstopmode {name}"):
        r = sh(cmd, cwd=paper)
        summary.append(f"[{name}] {cmd}: exit {r.returncode}")
    log = (paper / f"{name}.log").read_text(errors="replace") if (paper / f"{name}.log").exists() else ""
    m = re.findall(rf"Output written on {name}.pdf \((\d+) pages?", log)
    summary.append(f"[{name}] PAGES: " + (m[-1] if m else "no PDF produced"))
    errs = [l for l in log.splitlines() if l.startswith("!")]
    summary.append(f"[{name}] LaTeX errors: {len(errs)}")
    summary += errs[:20]
    blg = (paper / f"{name}.blg").read_text(errors="replace") if (paper / f"{name}.blg").exists() else ""
    bib_problems = [l for l in blg.splitlines() if l.startswith("Warning") or "missing a field" in l or "rror" in l]
    summary.append(f"[{name}] bibtex problems: {len(bib_problems)}")
    summary += bib_problems[:20]
    undefined = sorted(set(re.findall(r"Citation `([^']+)' .*undefined", log)) | set(re.findall(r"Reference `([^']+)' .*undefined", log)))
    summary.append(f"[{name}] undefined citations/references: " + (", ".join(undefined) if undefined else "none"))
    for ext in ("pdf", "log", "blg", "bbl"):
        if (paper / f"{name}.{ext}").exists():
            shutil.copy(paper / f"{name}.{ext}", OUT / f"{name}.{ext}")


build("main_track2")
sh("pdftoppm -r 80 -png main_track2.pdf page", cwd=paper)
for png in sorted(paper.glob("page-*.png")):
    shutil.copy(png, OUT / png.name)
build("main_track2_anon")
sh("pdftotext -layout main_track2_anon.pdf anon.txt", cwd=paper)
anon_txt = (paper / "anon.txt").read_text(errors="replace") if (paper / "anon.txt").exists() else ""
leaks = [w for w in ("Akaksha", "Singh", "Nishant", "Gupta", "du.ac.in", "Delhi", "github", "Akaksingh") if w.lower() in anon_txt.lower()]
summary.append("anonymised PDF still contains: " + (", ".join(leaks) if leaks else "no author name, email, affiliation or repo link"))
(OUT / "summary.txt").write_text("\n".join(summary))
print("\n".join(summary[:80]), flush=True)
shutil.rmtree(REPO, ignore_errors=True)
