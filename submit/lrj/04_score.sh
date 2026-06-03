#!/bin/bash
# Submission: sbatch submit/lrj/04_score.sh

#SBATCH --job-name=LRJ_score
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|l40s'
#SBATCH --mem=64G
#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%j.LRJ_Score.out

echo "===> Started at: $(date)"
cd /home/yuanda/srj/lundtoptagger

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

python test_make_scores_LRJ.py configs/config_make_scores_LRJ.yaml

echo "===> Job finished at: $(date)"
