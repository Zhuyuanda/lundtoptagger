#!/bin/bash

# Submission command:   sbatch submit_preprocess_lrj.sh

#SBATCH --job-name=lrj_preprocess_pt
#SBATCH --partition=RCIF
#SBATCH --exclude=compute-0-21
#SBATCH -N 1
#SBATCH -n 8                    # 8个核心足以处理数据读取和权重计算
#SBATCH --mem=25G              # 增加到 128GB：因为计算全局 pT 权重需要一次性加载所有样本的 pT 和 Label 数组
#SBATCH --export=ALL

# 建议创建专门的 lrj 目录存放日志
#SBATCH --output=/home/yuanda/srj/lundtoptagger/preprocess_log/lrj_pt-%j.out
#SBATCH --error=/home/yuanda/srj/lundtoptagger/preprocess_log/lrj_pt-%j.err

#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL

echo "===> Time: $(date)"
echo "===> Hostname: $(hostname)"
echo "===> Job ID: ${SLURM_JOB_ID}"

# ------------------------------------------
# Environment Setup
# ------------------------------------------
echo "===> Activating environment"
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo "Using environment: $CONDA_DEFAULT_ENV"

# ------------------------------------------
# Run Preprocess
# ------------------------------------------
cd /home/yuanda/srj/lundtoptagger

# 调用刚才为你优化过的仅处理 pT flattening 的脚本
echo "===> Running LRJ pT-only preprocess script..."
python preprocess_LRJ_CPU.py configs/config_preprocess_LRJ.yaml

echo "===> Job finished at $(date)"