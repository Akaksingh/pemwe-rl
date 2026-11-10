"""Kaggle kernel entry point: clone the public repo, run the follow-up experiments, keep only the results zip.

Push with `kaggle kernels push -p kaggle/kernel`. Edit MODE between pushes: "smoke" (pipeline check, ~15 min)
or "full" (the real run). Output = /kaggle/working/kaggle_results.zip plus run.log.
"""
import shutil
import subprocess
import sys
from pathlib import Path

MODE = "full"           # "smoke" or "full"
URL = "https://github.com/Akaksingh/pemwe-rl.git"
WORK = Path("/kaggle/working")
REPO = WORK / "repo"

shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", URL, str(REPO)], check=True)
print("commit:", subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), flush=True)
subprocess.run(["nvidia-smi"])
extra = ["--smoke", "--utd", "0.125"] if MODE == "smoke" else ["--max-hours", "8"]
rc = subprocess.run([sys.executable, str(REPO / "kaggle" / "run_experiments.py"), "--phase", "all", *extra]).returncode
print("run_experiments exit code:", rc, flush=True)
# keep the saved output small: results zip only (the zip already contains the new policies and CSVs)
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(WORK / "out", ignore_errors=True)
sys.exit(rc)
