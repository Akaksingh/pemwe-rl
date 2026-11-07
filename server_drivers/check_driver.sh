cd ~/pemwe-rl
export PYTHONPATH=src OMP_NUM_THREADS=2
PY=.venv/bin/python
S=logs/check_status.txt
date +"check start %T" > $S
cat > .check_jobs.txt <<'EOJ'
20.0 0
20.0 1
12.9 0
12.9 1
EOJ
cat .check_jobs.txt | xargs -P 4 -L 1 bash -c '
  w=$0; seed=$1
  .venv/bin/python -m pemwe.train --algo sac --seed $seed --override reward.w2=$w --override train.sac.train_freq=32 \
     --override train.sac.gradient_steps=32 --override train.sac.gamma=0.95 --device cuda --total-timesteps 300000 \
     --progress-csv results_check/chk_utd32_w2-${w}_s${seed}.csv --progress-every 10000 \
     --run-id chk_utd32_w2-${w}_s${seed} > logs/chk_w2-${w}_s${seed}.log 2>&1'
date +"check train done %T" >> $S
export OMP_NUM_THREADS=1
$PY scripts/run_scenario_grid.py --grid nominal --split train --day-stride 5 --workers 14 --out results_check/grid_check_train.csv \
   --models models/chk_utd32_w2-20.0_s0.zip models/chk_utd32_w2-20.0_s1.zip models/chk_utd32_w2-12.9_s0.zip models/chk_utd32_w2-12.9_s1.zip \
            models/e5_w2-20.0_s0.zip models/e5_w2-20.0_s1.zip models/e5_w2-12.9_s0.zip models/e5_w2-12.9_s1.zip > logs/check_eval.log 2>&1
date +"check eval done %T" >> $S
echo CHECK_DONE >> $S
