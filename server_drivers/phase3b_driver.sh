cd ~/pemwe-rl
export PYTHONPATH=src
PY=.venv/bin/python
S=logs/hpc_status.txt
while pgrep -u USER -f bench_hpc_diag.py > /dev/null; do sleep 10; done
date +"diag done %T" >> $S

# ---- round 2 pilots: ALL FOUR on the GPU (measured: 4 GPU procs ~330 upd/s aggregate vs ~272 for 2 GPU + 3 CPU)
OV="--override reward.w2=0.1 --override train.sac.train_freq=32 --override train.sac.gradient_steps=1024"
export OMP_NUM_THREADS=2
$PY -m pemwe.train --algo sac --seed 0 $OV --override train.sac.gamma=0.9  --device cuda --total-timesteps 300000 --run-id pilot2_g0.9_s0 > logs/p2_a.log 2>&1 &
$PY -m pemwe.train --algo sac --seed 0 $OV --override train.sac.gamma=0.8  --device cuda --total-timesteps 300000 --run-id pilot2_g0.8_s0 > logs/p2_b.log 2>&1 &
$PY -m pemwe.train --algo sac --seed 1 $OV --override train.sac.gamma=0.95 --device cuda --total-timesteps 300000 --run-id pilot2_g0.95_s1 > logs/p2_c.log 2>&1 &
$PY -m pemwe.train --algo sac --seed 0 $OV --override train.sac.gamma=0.95 --override reward.h2_scale_kg=0.1 --override reward.deg_scale_uv=1.0 --device cuda --total-timesteps 300000 --run-id pilot2_g0.95_rs10_s0 > logs/p2_d.log 2>&1 &
date +"round2 train start %T" >> $S
wait
date +"round2 train done %T" >> $S

export OMP_NUM_THREADS=1
$PY scripts/run_scenario_grid.py --grid nominal --split train --workers 12 --out results_pilot/grid_round2.csv \
   --models models/pilot_utd1_w2-0.1_seed0.zip models/pilot_utd1_w2-0.1_seed1.zip models/pilot_utd1_g0.99_seed0.zip \
            models/pilot_utd1_g0.95_seed0.zip models/pilot2_g0.9_s0.zip models/pilot2_g0.8_s0.zip \
            models/pilot2_g0.95_s1.zip models/pilot2_g0.95_rs10_s0.zip > logs/p2_eval.log 2>&1
$PY scripts/summarize_pilots.py --csv results_pilot/grid_round2.csv --w2 0.1 >> logs/p2_eval.log 2>&1
date +"round2 eval done %T" >> $S
echo PHASE3_DONE >> $S
