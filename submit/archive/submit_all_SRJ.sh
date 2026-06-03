#!/bin/bash

# ==========================================================
# 路径定义 (请确保这些路径在你的服务器上是正确的)
# ==========================================================
SUBMIT_DIR=$(cd "$(dirname "$0")"; pwd)
BASE_DATA_PATH="/share/lustre/yzhu/IOP_DATA/preprocessed/train/processed_*.pt"
BASE_SAVE_DIR="/share/lustre/yzhu/IOP_DATA/models_fixed"
CONFIG_DIR="/home/yuanda/srj/lundtoptagger/configs/auto_generated_fixed"
LOG_DIR="/home/yuanda/srj/lundtoptagger/lundlog"

mkdir -p $CONFIG_DIR
mkdir -p $LOG_DIR

# 实验矩阵: "模式名称:额外特征维度"
# 逻辑：extra_dim = 物理特征数 + 1 (Ntrk)
experiments=(
    "lund_only:1"
    "lund_gn3x:10"
    "lund_b75:2"
    "lund_b50:2"
    "lund_part:2"
)

# ==========================================================
# 核心操作：清理旧的 GPU 任务 (不影响 RCIF 分区的 make_dat)
# ==========================================================
echo ">>> Cleaning up existing GPU partition jobs for user: $USER..."
scancel -p GPU -u $USER

# ==========================================================
# 循环提交新任务
# ==========================================================
for exp in "${experiments[@]}"; do
    IFS=":" read -r MODE DIM <<< "$exp"
    EXP_SAVE_DIR="${BASE_SAVE_DIR}/exp_${MODE}_SRJ"
    mkdir -p $EXP_SAVE_DIR
    
    # 策略：BS=1024 是 V100/L40S/A100 通用的安全值
    BS=1024 

    # 1. 动态生成 YAML 配置文件
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
num_workers: 2
EOF

    # 2. 提交任务：彻底去掉 --constraint，让调度器自由分配
    sbatch --job-name="SRJ_lund" \
           --export=ALL,CONF="$CONF_FILE" \
           --output="${LOG_DIR}/slurm-%j.${MODE}.out" \
           "${SUBMIT_DIR}/submit_slurm_template.sh"

    echo ">>> Submitted $MODE with BS=$BS (Instant Start Strategy)."
    sleep 1
done

echo -e "\n--- All 5 GPU tasks submitted. Checking status: ---"
squeue -u $USER | grep SRJ