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

REPO_ROOT=/home/yuanda/lundtoptagger
LOG_DIR=${REPO_ROOT}/lundlog

cd $REPO_ROOT
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/lustre/yzhu/conda/envs/torch26_py39

PATH_TO_SAVE=$(python -c "import yaml; c=yaml.safe_load(open('${CONF}')); print(c['data']['path_to_save'])" 2>/dev/null)
MODE=$(python -c "import yaml; c=yaml.safe_load(open('${CONF}')); print(c['architecture']['training_mode'])" 2>/dev/null)

resubmit() {
    echo ">>> Signal received — checking if continuation needed..." >&2
    if [ -n "$PATH_TO_SAVE" ] && [ -n "$MODE" ] && [ ! -f "${PATH_TO_SAVE}/done_${MODE}.flag" ]; then
        # Kill Python gracefully so it finishes saving the current checkpoint
        if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
            echo ">>> Sending SIGTERM to Python (pid=$PYTHON_PID) ..." >&2
            kill -TERM "$PYTHON_PID"
            wait "$PYTHON_PID" 2>/dev/null
        fi
        # Name continuation after the experiment dir (e.g. Tr_former_cWP50_W_cont)
        # so baseline and LundFormer chains are distinguishable in the queue.
        EXP_TAG=$(basename "$(dirname "$PATH_TO_SAVE")")   # exp_lund_xxx
        EXP_TAG=${EXP_TAG#exp_lund_}
        case "$PATH_TO_SAVE" in
            *WTagging*|*/models_W/*|*_W_80*)       CH=W ;;
            *TopTagging*|*/models_Top/*|*_Top_*)   CH=top ;;
            *)            CH=x ;;
        esac
        JOB_NAME="Tr_${EXP_TAG}_${CH}_cont"
        case "${SLURM_JOB_NAME:-}" in v3_*) JOB_NAME=$SLURM_JOB_NAME ;; esac
        echo ">>> Submitting continuation job: ${JOB_NAME}" >&2
        sbatch \
            --job-name="${JOB_NAME}" \
            --partition=GPU \
            --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
            --gres=gpu:1 --constraint='a100|l40s|v100' \
            --exclude="${EXCL:-compute-gpu-0-3,compute-gpu-0-5}" \
            --mem="${SLURM_MEM_PER_NODE:-122880}M" --time=48:00:00 \
            --export=ALL,CONF="${CONF}" \
            --output="${LOG_DIR}/slurm-%j.${JOB_NAME}.out" \
            "${BASH_SOURCE[0]}"
    else
        echo ">>> Training already converged or no config info, skipping resubmit." >&2
    fi
}

trap 'resubmit; exit 0' SIGUSR1 SIGTERM

PY_START=$SECONDS
python weight_ONLY_TRAINS_LRJ.py "$CONF" &
PYTHON_PID=$!
wait $PYTHON_PID
PYTHON_EXIT=$?

# Normal exit: check if done, resubmit if not converged
if [ $PYTHON_EXIT -ne 0 ] && [ ! -f "${PATH_TO_SAVE}/done_${MODE}.flag" ]; then
    echo ">>> Python exited with code $PYTHON_EXIT without done flag — resubmitting..." >&2
    # A crash within minutes of start is an environment/config error that a
    # retry cannot fix; resubmitting would loop (one job every few seconds).
    if [ $((SECONDS - PY_START)) -lt 600 ]; then
        echo ">>> Python failed after $((SECONDS - PY_START))s - not resubmitting (fast failure)." >&2
        exit 1
    fi
    resubmit
fi
