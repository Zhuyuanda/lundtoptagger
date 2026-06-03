#!/bin/bash

# Submission command:   sbatch submit_slurm_make_data_lrj.sh

#SBATCH --job-name=make_data_lrj_top
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -n 2              # 优化点 1: 减少 CPU 核心请求 (MakeData 主要是单核或低并发)
#SBATCH --mem=64G         # 优化点 2: 大幅降低内存请求 (64G 足以处理 10000 喷注的 Batch)

# 优化点 3: 限制并发运行数 (加上 %5，表示 25 个任务中最多同时运行 5 个)
# 这能有效防止多个任务同时读写磁盘导致的 I/O 堵塞，也能让其他同学的任务插空运行
#SBATCH --array=0-24%5

#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%A_%a.out

# ------------------------------------------
# Environment and Paths
# ------------------------------------------
cd /home/yuanda/srj/lundtoptagger
echo "Now in $(pwd)"
hostname

echo "Activating environment..."
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

# ------------------------------------------
# Parameters
# ------------------------------------------
main_config="configs/config_make_data_LRJ.yaml"
signal_config="configs/config_signal_LRJ.yaml"
event_fraction_idx=${SLURM_ARRAY_TASK_ID}

# 优化点 4: 随机延迟启动 (防止所有任务在同一秒打开几百个 ROOT 文件)
sleep $((RANDOM % 30))

echo "Running index: $event_fraction_idx"

# ------------------------------------------
# Run MakeData (LRJ Version)
# ------------------------------------------
python Make_data_LRJ.py "$main_config" --override \
    signal_config_file="$signal_config" \
    id="TopTagging_LRJ" \
    signal="top" \
    event_fraction_idx="$event_fraction_idx"

echo "Job done."