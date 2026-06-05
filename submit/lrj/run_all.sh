#!/bin/bash
# ============================================================
# LRJ LundNet — One-click pipeline (11-part split + 30 models)
#
# Usage:
#   bash submit/lrj/run_all.sh          # top tagger (default)
#   bash submit/lrj/run_all.sh W        # W tagger
#   bash submit/lrj/run_all.sh top W    # both signals sequentially
#
# Pipeline per signal:
#   [1] make_data  – SLURM array 0-10 (10 train parts 7% each + 1 test 30%)
#   [2] preprocess – CPU job, train_part{0..9}/ + test/
#   [3+4] per model: train (GPU) → score (GPU)
#
# Model count:
#   top: lund_only×6 (LC N=5..10) + toptrans + bWP×5 + fb×5 = 17
#   W:   lund_only×6 (LC N=5..10) + wtrans   + cWP×3 + fc×3 = 13
#   Total: 30 models × 2 jobs (train+score) = 60 GPU jobs
# ============================================================

set -euo pipefail

# Allow running both signals: run_all.sh top W
SIGNALS=("${@:-top}")

for SIGNAL in "${SIGNALS[@]}"; do
    case "$SIGNAL" in
        top) DATA_ID="TopTagging_LRJ" ;;
        W)   DATA_ID="WTagging_LRJ"   ;;
        *)   echo "Unknown signal: $SIGNAL  (choose 'top' or 'W')"; exit 1 ;;
    esac

    SUBMIT_DIR=$(cd "$(dirname "$0")"; pwd)
    REPO_ROOT=$(cd "$SUBMIT_DIR/../.." && pwd)
    CONFIG_DIR="$REPO_ROOT/configs/generated"
    LOG_DIR="$REPO_ROOT/lundlog"

    # ---- Hypatia data paths ----
    DATA_ROOT="/share/lustre/yzhu/IOP_DATA"
    MEANSTD_JSON="${DATA_ROOT}/part0_7percent/meanstd_part0.json"
    PREPROC_DIR="${DATA_ROOT}/preprocessed_${DATA_ID}"
    MODELS_BASE="${DATA_ROOT}/models_lrj_${DATA_ID}"
    SCORE_BASE="${DATA_ROOT}/score_lrj_${DATA_ID}"
    TEST_ROOT_GLOB="${DATA_ROOT}/part10_30percent/data_${DATA_ID}_batch*_ln_kT_cut_None.root"
    TEST_GRAPH_GLOB="${PREPROC_DIR}/test/processed_graphs_${DATA_ID}_batch*_ln_kT_cut_None_with_pt.pt"

    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PREPROC_DIR" "$MODELS_BASE" "$SCORE_BASE"

    echo "============================================================"
    echo "  LRJ pipeline: SIGNAL=${SIGNAL}  DATA_ID=${DATA_ID}"
    echo "  Repo:  $REPO_ROOT"
    echo "  Logs:  $LOG_DIR"
    echo "============================================================"

    # ==========================================================
    # [1] make_data  (array 0-10: idx 0-9 = train parts, 10 = test)
    # ==========================================================
    JOB_MAKE=$(sbatch --parsable \
        --job-name="LRJ1_make_${SIGNAL}" \
        --output="${LOG_DIR}/slurm-%A_%a.make_${SIGNAL}.out" \
        --export=ALL,SIGNAL="${SIGNAL}",DATA_ID="${DATA_ID}" \
        "${SUBMIT_DIR}/01_make_data.sh"
    )
    echo "[1] make_data  → JOB ${JOB_MAKE}  (array 0-10, 11 tasks)"

    # ==========================================================
    # [2] preprocess  (CPU, after all make_data tasks)
    # ==========================================================
    PREP_CONF="${CONFIG_DIR}/config_preprocess_${DATA_ID}.yaml"

    # Build train_parts YAML block (10 parts, idx 0-9)
    TRAIN_PARTS_BLOCK=""
    for i in $(seq 0 9); do
        TRAIN_PARTS_BLOCK+="    - \"${DATA_ROOT}/part${i}_7percent/graphs_${DATA_ID}_batch*_ln_kT_cut_None_with_pt.pt\"\n"
    done

    printf "data:\n  train_parts:\n%s  test_graphs:\n    - \"%s\"\n  out_dir:      \"%s\"\n  meanstd_json: \"%s\"\n\nflatten_pt:        true\nn_bins_pt:         100\nstreaming_flatten: true\n" \
        "$(echo -e "$TRAIN_PARTS_BLOCK")" \
        "${DATA_ROOT}/part10_30percent/graphs_${DATA_ID}_batch*_ln_kT_cut_None_with_pt.pt" \
        "${PREPROC_DIR}" \
        "${MEANSTD_JSON}" \
        > "$PREP_CONF"

    JOB_PREP=$(sbatch --parsable \
        --job-name="LRJ2_prep_${SIGNAL}" \
        --partition=RCIF \
        --exclude=compute-0-21 \
        --nodes=1 \
        --ntasks=8 \
        --mem=25G \
        --dependency=afterok:"${JOB_MAKE}" \
        --output="${LOG_DIR}/slurm-%j.prep_${SIGNAL}.out" \
        --wrap="
            cd '${REPO_ROOT}'
            source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
            conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
            python preprocess_LRJ_CPU.py '${PREP_CONF}'
        "
    )
    echo "[2] preprocess → JOB ${JOB_PREP}  (after ${JOB_MAKE})"

    # ==========================================================
    # [3+4] Submit one (train → score) pair per model
    # ==========================================================
    # submit_model MODE DIM N_PARTS [FILTER_ATTR]
    submit_model() {
        local MODE=$1
        local DIM=$2
        local N_PARTS=$3
        local FILTER_ATTR=${4:-""}

        local TAG="${MODE:5}"          # strip "lund_"
        local EXP_SAVE="${MODELS_BASE}/exp_${MODE}/nparts${N_PARTS}"
        local SCORE_DIR="${SCORE_BASE}/exp_${MODE}/nparts${N_PARTS}"
        mkdir -p "$EXP_SAVE" "$SCORE_DIR"

        # --- Training config ---
        local TRAIN_CONF="${CONFIG_DIR}/train_${DATA_ID}_${MODE}_np${N_PARTS}.yaml"
        local FILTER_LINE=""
        [[ -n "$FILTER_ATTR" ]] && FILTER_LINE="filter_attr: \"${FILTER_ATTR}\""

        cat > "$TRAIN_CONF" <<YAML
data:
  preproc_dir: "${PREPROC_DIR}"
  path_to_save: "${EXP_SAVE}"
architecture:
  choose_model:       "LundNet_General"
  training_mode:      "${MODE}"
  extra_dim:          ${DIM}
  batch_size:         1024
  learning_rate:      0.0004
  n_epochs:           35
  patience:           8
  test_size:          0.1
  optimizer_schedule: [8, 16]
n_parts: ${N_PARTS}
shards_per_window: 20
gpu: 0
num_workers: 0
${FILTER_LINE}
YAML

        local JID_TRAIN
        JID_TRAIN=$(sbatch --parsable \
            --job-name="Tr_${TAG}_n${N_PARTS}" \
            --partition=GPU \
            --nodes=1 --ntasks=1 --cpus-per-task=8 \
            --gres=gpu:1 --mem=200G --time=48:00:00 \
            --dependency=afterok:"${JOB_PREP}" \
            --export=ALL,CONF="${TRAIN_CONF}" \
            --output="${LOG_DIR}/slurm-%j.tr_${MODE}_np${N_PARTS}.out" \
            "${SUBMIT_DIR}/03_train_worker.sh"
        )

        # --- Score config (per model) ---
        local SCORE_CONF="${CONFIG_DIR}/score_${DATA_ID}_${MODE}_np${N_PARTS}.yaml"
        local CKPT="${EXP_SAVE}/best_model_${MODE}.pt"

        cat > "$SCORE_CONF" <<YAML
data:
  paths_to_test_file_root:
    - "${TEST_ROOT_GLOB}"
  paths_to_test_file_graphs:
    - "${TEST_GRAPH_GLOB}"
  path_to_outdir: "${SCORE_DIR}"
  sample: "${DATA_ID}"
  kT_cut: "None"

test:
  batch_size: 1024
  scores_branch_name: "LRJ_{tag}_score"
  models_to_run:
    - tag:       "${TAG}"
      extra_dim: ${DIM}
      ckpt:      "${CKPT}"
YAML

        local JID_SCORE
        JID_SCORE=$(sbatch --parsable \
            --job-name="Sc_${TAG}_n${N_PARTS}" \
            --partition=GPU \
            --nodes=1 --ntasks=1 --cpus-per-task=8 \
            --gres=gpu:1 --constraint='a100|l40s' \
            --mem=64G \
            --dependency=afterok:"${JID_TRAIN}" \
            --output="${LOG_DIR}/slurm-%j.sc_${MODE}_np${N_PARTS}.out" \
            --wrap="
                cd '${REPO_ROOT}'
                source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
                conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
                python test_make_scores_LRJ.py '${SCORE_CONF}'
            "
        )

        printf "    %-28s  np=%-2s  train=%-8s score=%s\n" \
            "${MODE}" "${N_PARTS}" "${JID_TRAIN}" "${JID_SCORE}"
        ALL_SCORE_IDS+=("$JID_SCORE")
    }

    ALL_SCORE_IDS=()
    echo "[3+4] Submitting train+score jobs (after prep ${JOB_PREP})..."

    # ---- lund_only learning curve (N=5..10) ----
    for N in 5 6 7 8 9 10; do
        submit_model "lund_only" 1 "${N}"
    done

    if [[ "$SIGNAL" == "top" ]]; then
        # ---- Top tagger experiments (N=10) ----
        submit_model "lund_toptrans" 2 10
        submit_model "lund_bWP65"    2 10
        submit_model "lund_bWP70"    2 10
        submit_model "lund_bWP77"    2 10
        submit_model "lund_bWP85"    2 10
        submit_model "lund_bWP90"    2 10
        submit_model "lund_fb65"     1 10 "has_reco_b_WP65"
        submit_model "lund_fb70"     1 10 "has_reco_b_WP70"
        submit_model "lund_fb77"     1 10 "has_reco_b_WP77"
        submit_model "lund_fb85"     1 10 "has_reco_b_WP85"
        submit_model "lund_fb90"     1 10 "has_reco_b_WP90"
    else
        # ---- W tagger experiments (N=10) ----
        submit_model "lund_wtrans"   2 10
        submit_model "lund_cWP10"    2 10
        submit_model "lund_cWP30"    2 10
        submit_model "lund_cWP50"    2 10
        submit_model "lund_fc10"     1 10 "has_reco_c_WP10"
        submit_model "lund_fc30"     1 10 "has_reco_c_WP30"
        submit_model "lund_fc50"     1 10 "has_reco_c_WP50"
    fi

    N_MODELS=${#ALL_SCORE_IDS[@]}
    echo ""
    echo "============================================================"
    echo "  SIGNAL=${SIGNAL}: ${N_MODELS} models submitted"
    echo "  make=${JOB_MAKE}  prep=${JOB_PREP}"
    echo "  Score job IDs: ${ALL_SCORE_IDS[*]}"
    echo "  Monitor: squeue -u \$USER"
    echo "============================================================"
    echo ""
done
