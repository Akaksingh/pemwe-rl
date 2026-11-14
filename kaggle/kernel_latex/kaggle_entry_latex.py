"""Kaggle kernel (CPU): compile paper/main_track2.tex with TeX Live and report the page count and any errors.

Push with `kaggle kernels push -p kaggle/kernel_latex`. Output: /kaggle/working/latex_out/
(main_track2.pdf, main_track2.log, main_track2.blg, summary.txt). The repo clone is removed afterwards.
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
print("apt exit", r.returncode, r.stderr[-800:], flush=True)

paper = REPO / "paper"
for cmd in ("pdflatex -interaction=nonstopmode main_track2", "bibtex main_track2",
            "pdflatex -interaction=nonstopmode main_track2", "pdflatex -interaction=nonstopmode main_track2"):
    r = sh(cmd, cwd=paper)
    summary.append(f"{cmd}: exit {r.returncode}")
    print(cmd, "->", r.returncode, flush=True)

log = (paper / "main_track2.log").read_text(errors="replace") if (paper / "main_track2.log").exists() else ""
m = re.findall(r"Output written on main_track2.pdf \((\d+) pages?", log)
summary.append("PAGES: " + (m[-1] if m else "no PDF produced"))
errs = [l for l in log.splitlines() if l.startswith("!") or "Warning" in l and ("undefined" in l or "Overfull" in l and "pt" in l and float(re.findall(r"([\d.]+)pt", l)[0]) > 20)]
summary.append(f"errors/serious warnings: {len(errs)}")
summary += errs[:60]
for f in ("main_track2.pdf", "main_track2.log", "main_track2.blg", "main_track2.bbl"):
    if (paper / f).exists():
        shutil.copy(paper / f, OUT / f)
(OUT / "summary.txt").write_text("\n".join(summary))
print("\n".join(summary[:80]), flush=True)
shutil.rmtree(REPO, ignore_errors=True)
