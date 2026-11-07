cd ~/pemwe-rl
export PYTHONPATH=src OMP_NUM_THREADS=2
PY=.venv/bin/python
S=logs/hpc_status.txt
while ! grep -qx E3_DONE $S 2>/dev/null; do sleep 20; done
export GAMMA=$(cat .e3_gamma)
mkdir -p results_e5
date +"E5 start gamma=$GAMMA %T" >> $S
cat > .e5_jobs.txt <<'EOJ'
20.0 0
20.0 1
12.9 0
12.9 1
5.32 0
5.32 1
1.41 0
1.41 1
EOJ
cat .e5_jobs.txt | xargs -P 4 -L 1 bash -c '
  w=$0; seed=$1
  .venv/bin/python -m pemwe.train --algo sac --seed $seed --override reward.w2=$w --override train.sac.train_freq=32 \
     --override train.sac.gradient_steps=1024 --override train.sac.gamma=$GAMMA --device cuda --total-timesteps 300000 \
     --progress-csv results_e5/e5_w2-${w}_s${seed}.csv --progress-every 10000 \
     --run-id e5_w2-${w}_s${seed} > logs/e5_w2-${w}_s${seed}.log 2>&1'
date +"E5 train done %T" >> $S
export OMP_NUM_THREADS=1
$PY scripts/run_scenario_grid.py --grid both --split test --workers 14 --out results_e5/grid_e5_test.csv \
   --models models/e5_w2-*_s*.zip > logs/e5_eval.log 2>&1
date +"E5 eval done %T" >> $S
echo E5_DONE >> $S
