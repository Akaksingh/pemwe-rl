# Batched (episode-parallel) simulator: validation and benchmark

* Code: `src/pemwe/batch.py`; tests: `tests/test_batch.py` (8 tests, vs the scalar `PEMWEEnv`); benchmark: `scripts/bench_batch.py`;
  analysis: `scripts/analyze_batch.py` (Hockney fit, figure `results/figures/fig_batch.pdf`, `batch_summary.json`).
* Produced by Kaggle kernel `pemwe-batch` (CPU only; `kaggle/kernel_batch/`), repo commit in `environment.txt`
  (d19f14e), 4 vCPU Intel Xeon 2.2 GHz (`lscpu.txt`), Linux, Python 3.13, NumPy 2.1.3, on the real 72 held-out days.
* Same-machine ratios only: scalar environment and batched simulator ran back to back. Absolute rates describe that machine,
  not the workstation of the other results (which was no longer available).
* `pytest.log`: the 8 tests pass on Linux. `bench_batch.meta.json`: validation of 24 random grid episodes against the
  scalar environment, maximum relative difference 0.0.
* A laptop trial run (Windows, noisy: the same batch took 5.7 s and 23.7 s) was not kept and is not used.
