#!/bin/bash

# 路径定义
SUBMIT_DIR=$(cd "$(dirname "$0")"; pwd)
BASE_DATA_PATH="/share/lustre/yzhu/IOP_DATA/preprocessed/train/processed_*.pt"
BASE_SAVE_DIR="/share/lustre/yzhu/IOP_DATA/models_fixed"
CONFIG_DIR="/home/yuanda/srj/lundtoptagger/configs/auto_generated_fixed"
LOG_DIR="/home/yuanda/srj/lundtoptagger/lundlog"

mkdir -p $CONFIG_DIR
mkdir -p $LOG_DIR

experiments=(
    "lund_only:1"
    "lund_gn3x:10"
    "lund_b75:2"
    "lund_b50:2"
    "lund_part:2"
)

for exp in "${experiments[@]}"; do
    IFS=":" read -r MODE DIM <<< "$exp"
    EXP_SAVE_DIR="${BASE_SAVE_DIR}/exp_${MODE}_SRJ"
    mkdir -p $EXP_SAVE_DIR
    
    # 1. 核心逻辑：检查 A100 是否有空位
    # 如果有 idle 或 mix 的 A100，我们就申请 A100
    # 如果没有，我们就不加 constraint，让调度器随便分配一个能跑的 GPU (V100/L40s)
    A100_NODES=$(sinfo -p GPU -o "%N %t %f" | grep -i "a100" | grep -E "idle|mix" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//')
    
    if [ -n "$A100_NODES" ]; then
        # 如果有 A100 空闲，指定这些节点，确保秒跑
        SPECIFIC_NODES="--nodelist=$A100_NODES"
        BS=4096
        echo ">>> [Check] Found available A100 nodes: $A100_NODES. Targeting them..."
    else
        # 如果 A100 全满，不加任何限制，有什么跑什么（不 PD 的关键）
        SPECIFIC_NODES="" 
        BS=1024  # 降低 BS 以兼容可能分到的 V100 (16GB)
        echo ">>> [Check] A100 is busy. Falling back to ANY available GPU to avoid PD."
    fi

    # 2. 动态生成 YAML
    CONF_FILE="${CONFIG_DIR}/config_SRJ_${MODE}.yaml"
    cat <<EOF > $CONF_FILE
data:
  path_to_trainfiles: "$BASE_DATA_PATH"
  path_to_save: "$EXP_SAVE_DIR"
architecture:
  choose_model: "LundNet_General"
  training_mode: "$MODE"
  extra_dim: $DIM
  batch_size: $BS
  learning_rate: 0.0004
  n_epochs: 35
  patience: 8
  test_size: 0.1
gpu: 0
num_workers: 4  # 降低 worker 数减少 CPU 争抢，有助于不 PD
EOF

    # 3. 提交任务
    # 注意：删除了 --constraint，改用动态的 $SPECIFIC_NODES
    # 如果没有 A100，$SPECIFIC_NODES 为空，调度器会直接分发到 idle 的 V100/L40s 上
    sbatch --job-name="SRJ_${MODE}" \
           $SPECIFIC_NODES \
           --export=ALL,CONF="$CONF_FILE" \
           --output="${LOG_DIR}/slurm-%j.${MODE}.out" \
           "${SUBMIT_DIR}/submit_slurm_template.sh"

    echo ">>> Submitted $MODE with BS=$BS."
    sleep 2
done