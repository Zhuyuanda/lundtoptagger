#!/bin/bash

# ------------------------------------------------------------
# SBATCH 配置 - 针对 LRJ 打分任务优化
# ------------------------------------------------------------

#SBATCH --job-name=LRJ_MultiTagger_Score
#SBATCH -p GPU                      # GPU 分区 [cite: 965]
#SBATCH -N 1                        # 单节点 [cite: 966]
#SBATCH -n 8                        # 8 个 CPU 核心 [cite: 967]
#SBATCH --gres=gpu:1                # 请求 1 块 GPU [cite: 968]
#SBATCH --constraint='a100|l40s'    # 建议优先使用显存更大的 A100 [cite: 969]
#SBATCH --mem=64G                   # 增加到 64GB RAM，防止合并大半径喷流 ROOT 时 OOM [cite: 970]

# 日志文件路径
#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%j.LRJ_Score.out [cite: 973]

# ------------------------------------------------------------
# 环境设置
# ------------------------------------------------------------

echo "===> Started at: $(date)"
cd /home/yuanda/srj/lundtoptagger [cite: 978]

echo "===> 正在激活 ATLAS 环境..."
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

# ------------------------------------------------------------
# 运行打分 (支持多种 LundNet+X 组合)
# ------------------------------------------------------------

# 使用你新准备的 LRJ 配置文件
CONFIG_FILE="configs/config_make_scores_LRJ.yaml"

echo "===> Running LRJ scoring..."
echo "Config: $CONFIG_FILE"
echo "Target Models: LundNet, b50, b75, ParT, GN3X, ParT_alone, GN3X_alone"

# 运行适配了 LundNet_General 和 Standalone 分支的脚本
python test_make_scores_LRJ.py "$CONFIG_FILE"

echo "===> Job finished at: $(date)"