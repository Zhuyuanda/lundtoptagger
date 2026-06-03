#!/bin/bash
# Submission: sbatch submit/lrj/02_preprocess.sh

#SBATCH --job-name=LRJ_preprocess
#SBATCH --partition=RCIF
#SBATCH --exclude=compute-0-21
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=25G
#SBATCH --export=ALL
#SBATCH --output=/home/yuanda/srj/lundtoptagger/preprocess_log/lrj_pt-%j.out
#SBATCH --error=/home/yuanda/srj/lundtoptagger/preprocess_log/lrj_pt-%j.err
#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL

echo "===> Time: $(date)"
echo "===> Hostname: $(hostname)"
echo "===> Job ID: ${SLURM_JOB_ID}"

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

cd /home/yuanda/srj/lundtoptagger
python preprocess_LRJ_CPU.py configs/config_preprocess_LRJ.yaml

echo "===> Job finished at $(date)"
