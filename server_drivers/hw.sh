echo "date: $(date -Is)"; echo "host: $(hostname)"; uname -a; lsb_release -ds
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|CPU max MHz|L2 cache|L3 cache"
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
free -h | head -2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
~/pemwe-rl/.venv/bin/python -c "import sys,torch,numpy,gymnasium,stable_baselines3 as s;print('python',sys.version.split()[0],'| torch',torch.__version__,'| cuda runtime',torch.version.cuda,'| numpy',numpy.__version__,'| gymnasium',gymnasium.__version__,'| sb3',s.__version__)"
echo "git HEAD: $(git -C ~/pemwe-rl rev-parse HEAD)  (working tree has uncommitted edits: $(git -C ~/pemwe-rl status --short | grep -v '^??' | wc -l) tracked files)"
