#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=48:00:00

cd /home/yuanda/srj/lundtoptagger
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

python weight_ONLY_TRAINS_LRJ.py "$CONF"
