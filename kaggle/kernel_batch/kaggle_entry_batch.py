"""Kaggle kernel (CPU only): validate the batched simulator on the real held-out days and benchmark it.

Push with `kaggle kernels push -p kaggle/kernel_batch`. Output: /kaggle/working/batch_results/
(bench_batch.csv, bench_batch.meta.json, pytest.log, environment.txt, lscpu.txt).
"""
import shutil
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/Akaksingh/pemwe-rl.git"
WORK = Path("/kaggle/working")
REPO = WORK / "repo"
OUT = WORK / "batch_results"
OUT.mkdir(parents=True, exist_ok=True)

shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", URL, str(REPO)], check=True)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("commit:", commit, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium", "pyyaml", "pytest"], check=True)

(OUT / "lscpu.txt").write_text(subprocess.run(["lscpu"], capture_output=True, text=True).stdout)
(OUT / "environment.txt").write_text(
    f"repo commit: {commit}\npython: {sys.version}\n" + subprocess.run(
        [sys.executable, "-c", "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"],
        capture_output=True, text=True).stdout)

# 1. correctness: the batched simulator against the scalar environment (synthetic and extreme profiles)
t = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_batch.py"], cwd=REPO, capture_output=True, text=True)
(OUT / "pytest.log").write_text(t.stdout + t.stderr)
print(t.stdout[-1500:], flush=True)

# 2. benchmark and real-data validation on the 72 held-out days
b = subprocess.run([sys.executable, "scripts/bench_batch.py", "--real", "--repeats", "3", "--out", str(OUT / "bench_batch.csv")],
                   cwd=REPO)
print("bench exit code:", b.returncode, "pytest exit code:", t.returncode, flush=True)
shutil.rmtree(REPO, ignore_errors=True)
sys.exit(t.returncode or b.returncode)
