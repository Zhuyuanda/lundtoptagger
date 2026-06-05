#!/bin/bash
# Submission: bash submit/lrj/03_train_all.sh

SUBMIT_DIR=$(cd "$(dirname "$0")"; pwd)
REPO_ROOT="$(cd "$SUBMIT_DIR/../.." && pwd)"
BASE_DATA_PATH="/share/lustre/yzhu/IOP_DATA/preprocessed/train/processed_*.pt"
BASE_SAVE_DIR="/share/lustre/yzhu/IOP_DATA/models_lrj"
CONFIG_DIR="$REPO_ROOT/configs/generated"
LOG_DIR="$REPO_ROOT/lundlog"

mkdir -p "$CONFIG_DIR" "$LOG_DIR"

experiments=(
    "lund_only:1"
    "lund_gn3x:10"
    "lund_part:2"
    # GN2v01 official b-WPs (FixedCutBEff_XX, top-3 SRJs by pT in cone)
    "lund_bWP77:2"
    "lund_bWP85:2"
    # GN2v01 official c-WPs (FixedCutCEff_XX, top-2 SRJs, 77% b-veto embedded)
    "lund_cWP30:2"
    "lund_cWP50:2"
)

for exp in "${experiments[@]}"; do
    IFS=":" read -r MODE DIM <<< "$exp"
    EXP_SAVE_DIR="${BASE_SAVE_DIR}/exp_${MODE}"
    mkdir -p "$EXP_SAVE_DIR"

    BS=1024
    CONF_FILE="${CONFIG_DIR}/config_LRJ_lund_${MODE}.yaml"
    cat <<EOF > "$CONF_FILE"
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
  optimizer_schedule: [8, 16]
gpu: 0
num_workers: 0
EOF

    sbatch --job-name="LRJ_train_${MODE}" \
           --export=ALL,CONF="$CONF_FILE" \
           --output="${LOG_DIR}/slurm-%j.${MODE}.out" \
           "${SUBMIT_DIR}/03_train_worker.sh"

    echo ">>> Submitted LRJ_train_${MODE} (BS=$BS)"
    sleep 1
done

echo -e "\n--- Submitted 5 LRJ training jobs ---"
squeue -u "$USER" | grep LRJ_train || true
