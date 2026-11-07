cd ~/shinrai
export PYTHONPATH=src OMP_NUM_THREADS=2
PY=.venv/bin/python
S=logs/e3b_status.txt
date +"e3b start %T" > $S
cat > .e3b_jobs.txt <<'EOJ'
1 1024 0
1 1024 1
0.5 512 0
0.5 512 1
0.25 256 0
0.25 256 1
0.125 128 0
0.125 128 1
EOJ
cat .e3b_jobs.txt | xargs -P 4 -L 1 bash -c '
  utd=$0; gs=$1; seed=$2
  .venv/bin/python -m pemwe.train --algo sac --seed $seed --override reward.w2=20.0 --override train.sac.train_freq=32 \
     --override train.sac.gradient_steps=$gs --override train.sac.gamma=0.95 --device cuda --total-timesteps 300000 \
     --progress-csv results_e3b/e3b_utd${utd}_s${seed}.csv --progress-every 10000 --ckpt-every 50000 \
     --run-id e3b_utd${utd}_s${seed} > logs/e3b_utd${utd}_s${seed}.log 2>&1'
date +"e3b train done %T" >> $S
export OMP_NUM_THREADS=1
$PY scripts/run_scenario_grid.py --grid nominal --split train --day-stride 5 --workers 14 --extra-slew 0.0005 0.001 0.0015 0.002 \
   --out results_e3b/grid_e3b_train.csv --models models/ckpt/e3b_utd*_steps.zip > logs/e3b_eval.log 2>&1
date +"e3b eval done %T" >> $S
echo E3B_DONE >> $S
