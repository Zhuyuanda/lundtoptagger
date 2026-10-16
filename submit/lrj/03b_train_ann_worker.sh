#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|l40s|v100'
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300

# ANN (GMM adversary) training worker. Expects env:
#   CONF       — training yaml (do_combined_training: true, lambda_parameter set)
#   SEED_DIR   — directory holding the converged classifier
#                (best_model_${MODE}.pt + scale_factor.json)
# Chains itself across 48h windows until done_${MODE}_ann_lam${LAM}.flag exists.

REPO_ROOT=/home/yuanda/lundtoptagger
LOG_DIR=${REPO_ROOT}/lundlog

cd $REPO_ROOT
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/lustre/yzhu/conda/envs/torch26_py39

PATH_TO_SAVE=$(python -c "import yaml; c=yaml.safe_load(open('${CONF}')); print(c['data']['path_to_save'])")
MODE=$(python -c "import yaml; c=yaml.safe_load(open('${CONF}')); print(c['architecture']['training_mode'])")
LAM=$(python -c "import yaml; c=yaml.safe_load(open('${CONF}')); print('%g' % c['architecture']['lambda_parameter'])")
TAG="${MODE}_ann_lam${LAM}"

# Seed the classifier snapshot at job start (not submit time), so a baseline
# retraining that converged while we sat in the queue is picked up.
mkdir -p "$PATH_TO_SAVE"
if [ -n "${SEED_DIR:-}" ]; then
    if [ ! -f "${PATH_TO_SAVE}/best_model_${MODE}.pt" ]; then
        cp "${SEED_DIR}/best_model_${MODE}.pt" "${PATH_TO_SAVE}/" || exit 1
        echo ">>> Seeded classifier from ${SEED_DIR} ($(date))"
    fi
    [ -f "${PATH_TO_SAVE}/scale_factor.json" ] || cp "${SEED_DIR}/scale_factor.json" "${PATH_TO_SAVE}/"
fi
# Mark the classifier stage as done so the trainer jumps to the ANN stages
touch "${PATH_TO_SAVE}/done_${MODE}.flag"

resubmit() {
    echo ">>> Signal received — checking ANN continuation..." >&2
    if [ ! -f "${PATH_TO_SAVE}/done_${TAG}.flag" ]; then
        if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
            kill -TERM "$PYTHON_PID"; wait "$PYTHON_PID" 2>/dev/null
        fi
        CONT_NAME="ANN_${TAG}_cont"
        case "${SLURM_JOB_NAME:-}" in v3_*) CONT_NAME=$SLURM_JOB_NAME ;; esac
        sbatch --job-name="${CONT_NAME}" \
            --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
            --gres=gpu:1 --constraint='a100|l40s|v100' \
            --mem="${SLURM_MEM_PER_NODE:-122880}M" --time=48:00:00 \
            --export=ALL,CONF="${CONF}",SEED_DIR="${SEED_DIR:-}" \
            --output="${LOG_DIR}/slurm-%j.ann_${TAG}_cont.out" \
            "${BASH_SOURCE[0]}"
    fi
}
trap 'resubmit; exit 0' SIGUSR1 SIGTERM

PY_START=$SECONDS
python weight_ONLY_TRAINS_LRJ.py "$CONF" &
PYTHON_PID=$!
wait $PYTHON_PID
PYTHON_EXIT=$?

if [ $PYTHON_EXIT -ne 0 ] && [ ! -f "${PATH_TO_SAVE}/done_${TAG}.flag" ]; then
    echo ">>> Python exited ($PYTHON_EXIT) without ANN done flag — resubmitting..." >&2
    # A crash within minutes of start is an environment/config error that a
    # retry cannot fix; resubmitting would loop (one job every few seconds).
    if [ $((SECONDS - PY_START)) -lt 600 ]; then
        echo ">>> Python failed after $((SECONDS - PY_START))s - not resubmitting (fast failure)." >&2
        exit 1
    fi
    resubmit
fi
