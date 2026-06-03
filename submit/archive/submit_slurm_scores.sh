#!/bin/bash

# ------------------------------------------------------------
#  sbatch submit_slurm_scores.sh
# ------------------------------------------------------------

#SBATCH --job-name=SRJ_LundNet_score
#SBATCH -p GPU                     # GPU partition
#SBATCH -N1                        # single node
#SBATCH -n8                        # 8 CPU cores
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --constraint='a100|l40s|v100'
#SBATCH --mem=50G                  # 50 GB RAM

# Email notification (optional)
#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL

# Log files
#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%j.%x.out

# Submit by: sbatch submit_slurm_score.sh

# ------------------------------------------------------------
#  Environment setup
# ------------------------------------------------------------

echo "===> Current directory before cd:"
pwd

# Move into SRJ project folder
cd /home/yuanda/srj/lundtoptagger
echo "===> Switched directory to:"
pwd

echo "===> Hostname:"
hostname

echo "===> Activating conda environment..."
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo "Activated env: $CONDA_DEFAULT_ENV"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
#  Run scoring (supports multi-model evaluation)
# ------------------------------------------------------------

CONFIG_FILE="configs/config_make_scores_SRJ.yaml"

echo ""
echo "===> Starting SRJ scoring..."
echo "Using config: $CONFIG_FILE"

python test_make_scores_SRJ.py "$CONFIG_FILE"

echo "===> Job finished."
