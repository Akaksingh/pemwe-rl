cd ~/pemwe-rl
export PYTHONPATH=src
PY=.venv/bin/python
S=logs/hpc_status.txt
date +"start %F %T" > $S
echo "load: $(uptime)" >> $S
echo "gpu procs: $(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tr '\n' ';')" >> $S
bash .hw.sh > results_hpc/hardware_specifications.txt 2>&1
rm -f results_hpc/hpc_scaling.csv
OMP_NUM_THREADS=1 $PY .bench_env.py > logs/hpc_env.log 2>&1
date +"env bench done %T" >> $S
$PY .bench_learner.py > logs/hpc_learner.log 2>&1
date +"learner bench done %T" >> $S
export OMP_NUM_THREADS=4
for g in 0.99 0.95; do
  $PY -m pemwe.train --algo sac --seed 0 --override reward.w2=0.1 --override train.sac.train_freq=32 \
     --override train.sac.gradient_steps=1024 --override train.sac.gamma=$g --total-timesteps 300000 \
     --run-id pilot_utd1_g${g}_seed0 > logs/pilot_g$g.log 2>&1 &
done
wait
date +"pilots done %T" >> $S
{ for g in 0.99 0.95; do
    $PY -m pemwe.evaluate --model models/pilot_utd1_g${g}_seed0.zip --policy sac \
        --run-id pilot_utd1_g${g}_seed0 --split train --out results_pilot 2>&1 | grep -v -i warn
  done; $PY .diag2.py; } > logs/pilot_g_eval.txt 2>&1
date +"eval done %T" >> $S
echo HPC_DONE >> $S
