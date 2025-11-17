#!/bin/bash

# 提交方式: sbatch submit_slurm_make_data.sh

# job name
#SBATCH --job-name=make_data_110

# 队列
#SBATCH -p RCIF

# 节点与资源
#SBATCH -N1
#SBATCH -n4
#SBATCH --mem=35G

# 数组任务: 6 个输入 × 2 份 (70%, 30%) = 12 个任务, 最多同时跑 6 个
#SBATCH --array=0-1

# 邮件通知
#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL

# log 文件
#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%j.%a.%x.out

# 重要：路径必须加引号，且 * 必须保留不被 bash 提前展开
input_paths=("/home/yuanda/parent/parent.root")

ids=(test)
signals=(srj)

NUM_INPUTS=1
NUM_EVENT_FRACTIONS=2   # 两份事件切分: 70%, 30%

cd /home/yuanda/srj/lundtoptagger
echo "Now in $(pwd)"
hostname

echo "Activating environment..."
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo "Conda env: $CONDA_DEFAULT_ENV"

# 计算索引
event_fraction_idx=$(( SLURM_ARRAY_TASK_ID / NUM_INPUTS ))
input_set_idx=$(( SLURM_ARRAY_TASK_ID % NUM_INPUTS ))

path_to_rootfiles="${input_paths[$input_set_idx]}"
id="${ids[$input_set_idx]}"
signal="${signals[$input_set_idx]}"

# 主配置
main_config="configs/config_make_data_SRJ.yaml"
signal_config="configs/config_signal_SRJ.yaml"

echo ""
echo "path_to_rootfiles: $path_to_rootfiles"
echo "id: $id"
echo "signal: $signal"
echo "event_fraction_idx: $event_fraction_idx"
echo "Using main config: $main_config"
echo "Using signal config: $signal_config"

# 最终运行命令
python Make_data_SRJ.py "$main_config" --override \
    signal_config_file="$signal_config" \
    out_dir="/home/yuanda/dep/{id}{frac}" \
    path_to_rootfiles="$path_to_rootfiles" \
    id="$id" \
    signal="$signal" \
    event_fraction_idx="$event_fraction_idx"
