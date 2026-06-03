#!/bin/bash

# 路径定义 (保持一致)
SUBMIT_DIR=$(cd "$(dirname "$0")"; pwd)
BASE_DATA_PATH="/share/lustre/yzhu/IOP_DATA/preprocessed/train/processed_*.pt"
BASE_SAVE_DIR="/share/lustre/yzhu/IOP_DATA/models"
CONFIG_DIR="/home/yuanda/srj/lundtoptagger/configs/auto_generated"
LOG_DIR="/home/yuanda/srj/lundtoptagger/lundlog"

# 剩下的三个任务
experiments=(
    "lund_b75:2"
    "lund_b50:2"
    "lund_part:2"
)

for exp in "${experiments[@]}"; do
    IFS=":" read -r MODE DIM <<< "$exp"
    EXP_SAVE_DIR="${BASE_SAVE_DIR}/exp_${MODE}"
    mkdir -p $EXP_SAVE_DIR
    
    # 设定：不挑卡，BS 设为 1024 保证 V100 也能跑
    CONSTRAINT="a100|l40s|v100"
    BS=2048 

    CONF_FILE="${CONFIG_DIR}/config_train_${MODE}.yaml"
    cat <<EOF > $CONF_FILE
data:
  path_to_trainfiles: "$BASE_DATA_PATH"
  path_to_save: "$EXP_SAVE_DIR"
architecture:
  choose_model: "LundNet_General"
  training_mode: "$MODE"
  extra_dim: $DIM
  batch_size: $BS
  learning_rate: 0.001
  n_epochs: 35
  patience: 5
  test_size: 0.1
gpu: 0
num_workers: 4
EOF

    sbatch --job-name="LRJ_${MODE}" \
           --export=ALL,CONF="$CONF_FILE" \
           --constraint="$CONSTRAINT" \
           --output="${LOG_DIR}/slurm-%j.${MODE}.out" \
           "${SUBMIT_DIR}/submit_slurm_template.sh"

    echo ">>> Submitted $MODE with BS=$BS on ANY available GPU."
done