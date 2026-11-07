cd ~/pemwe-rl
export PYTHONPATH=src OMP_NUM_THREADS=4
PY=.venv/bin/python
OV1="--override reward.w2=0.1"
OV2="--override train.sac.train_freq=32"
OV3="--override train.sac.gradient_steps=1024"
date +"start %F %T" > logs/pilot_status.txt
for s in 0 1; do
  $PY -m pemwe.train --algo sac --seed $s $OV1 $OV2 $OV3 --total-timesteps 300000 \
      --run-id pilot_utd1_w2-0.1_seed$s > logs/pilot_s$s.log 2>&1 &
done
wait
date +"train done %F %T" >> logs/pilot_status.txt
{
  echo "=== EVAL on TRAIN split (in-sample, all train days), w2=0.1 ==="
  for s in 0 1; do
    $PY -m pemwe.evaluate --model models/pilot_utd1_w2-0.1_seed$s.zip --policy sac \
        --run-id pilot_utd1_w2-0.1_seed$s --split train --out results_pilot 2>&1 | grep -v -i warn
  done
  echo "--- hand-written controllers, same days"
  $PY -m pemwe.evaluate --policy baseline_naive --run-id naive_as_published --split train --out results_pilot 2>&1 | grep -v -i warn
  $PY -m pemwe.evaluate --policy baseline_naive --run-id no_threshold_const_plus1 --split train --out results_pilot \
      --override baseline.p_idle_threshold_frac=0.0 2>&1 | grep -v -i warn
  $PY -m pemwe.evaluate --policy baseline_ramplimited --run-id ramplimited --split train --out results_pilot 2>&1 | grep -v -i warn
} > logs/pilot_eval.txt 2>&1
date +"eval done %F %T" >> logs/pilot_status.txt
echo PILOT_DONE >> logs/pilot_status.txt
