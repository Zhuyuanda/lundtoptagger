#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8   # 增加 CPU 核心以加速 torch.load
#SBATCH --gres=gpu:1
#SBATCH --mem=200G          # 针对 5.9 亿节点，申请 200G 比较稳妥
#SBATCH --time=48:00:00     # 数据量巨大，建议给足 48 小时防止超时

cd /home/yuanda/srj/lundtoptagger
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

python weight_ONLY_TRAINS_LRJ_fixed.py "$CONF"