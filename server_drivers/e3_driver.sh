cd ~/pemwe-rl
export PYTHONPATH=src OMP_NUM_THREADS=2
PY=.venv/bin/python
S=logs/hpc_status.txt
# wait for round 2 to finish AND for the chosen gamma to be written by hand
while ! grep -q PHASE3_DONE $S 2>/dev/null || [ ! -s .e3_gamma ]; do sleep 20; done
GAMMA=$(cat .e3_gamma)
date +"E3 start gamma=$GAMMA %T" >> $S
# longest jobs first; 4 at a time on the GPU (measured best aggregate throughput)
cat > .e3_jobs.txt <<'EOJ'
1 1024 0
1 1024 1
0.5 512 0
0.5 512 1
0.125 128 0
0.125 128 1
0.03125 32 0
0.03125 32 1
EOJ
export GAMMA
cat .e3_jobs.txt | xargs -P 4 -L 1 bash -c '
  utd=$0; gs=$1; seed=$2
  .venv/bin/python -m pemwe.train --algo sac --seed $seed --override reward.w2=0.1 --override train.sac.train_freq=32 \
     --override train.sac.gradient_steps=$gs --override train.sac.gamma=$GAMMA --device cuda --total-timesteps 300000 \
     --progress-csv results_e3/e3_utd${utd}_s${seed}.csv --progress-every 10000 --ckpt-every 50000 \
     --run-id e3_utd${utd}_s${seed} > logs/e3_utd${utd}_s${seed}.log 2>&1'
date +"E3 train done %T" >> $S
export OMP_NUM_THREADS=1
$PY scripts/run_scenario_grid.py --grid nominal --split train --day-stride 5 --workers 14 --out results_pilot/e3_grid.csv \
   --models models/ckpt/e3_utd*_steps.zip > logs/e3_eval.log 2>&1
date +"E3 eval done %T" >> $S
echo E3_DONE >> $S
