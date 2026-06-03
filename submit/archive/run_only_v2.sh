#!/bin/bash
#SBATCH --job-name=SRJ_b75_fixed
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
##SBATCH --exclude=compute-gpu-0-0,compute-gpu-0-5
#SBATCH --output=/home/yuanda/srj/lundtoptagger/lundlog/slurm-%j.lund_b75.out

cd /home/yuanda/srj/lundtoptagger
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

python weight_ONLY_TRAINS_LRJ_fixed.py /home/yuanda/srj/lundtoptagger/configs/auto_generated_fixed/config_SRJ_lund_b75.yaml