cd ~/pemwe-rl
export PYTHONPATH=src
PY=.venv/bin/python
S=logs/hpc_status.txt
while ! grep -q HPC_DONE $S 2>/dev/null; do sleep 20; done
date +"phase2 start %F %T  load: $(uptime)" >> $S
OMP_NUM_THREADS=1 $PY scripts/bench_hpc_env.py > logs/hpc_env2.log 2>&1
date +"env bench v2 done %T" >> $S
$PY scripts/bench_hpc_multi.py > logs/hpc_multi.log 2>&1
date +"multi bench done %T" >> $S
echo PHASE2_DONE >> $S
