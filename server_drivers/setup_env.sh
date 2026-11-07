set -e
~/anaconda3/bin/conda create -p ~/pemwe-rl/.venv python=3.12 -y
PIP=~/pemwe-rl/.venv/bin/pip
$PIP install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
$PIP install --no-cache-dir numpy pandas scipy matplotlib pyyaml pyarrow tensorboard requests pytest
$PIP install --no-cache-dir "stable-baselines3==2.9.0" "gymnasium==1.3.0" || $PIP install --no-cache-dir stable-baselines3 gymnasium
~/pemwe-rl/.venv/bin/python -c "import torch,gymnasium,stable_baselines3 as s;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'gym',gymnasium.__version__,'sb3',s.__version__)"
echo SETUP_DONE
