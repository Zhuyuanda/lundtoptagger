#!/bin/bash

# ------------------------------------------------------------
#  SLURM config
# ------------------------------------------------------------

#SBATCH --job-name=SRJ_LundNet_train
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
#SBATCH --output=/home/yuanda/srj/lundtoptagger/lundlog/slurm-%j.%x.out

# Submit by sbatch submit_slurm_train.sh

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
#  Run training
# ------------------------------------------------------------

echo ""
echo "===> Starting SRJ training..."
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

echo "===> Job finished."
