#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-23:30:00
#SBATCH --job-name=09_gate_v3
#SBATCH --output=/share/lustre/yzhu/IOP_DATA/logs_lrj/slurm-%j.gate_v3.out

# v3 watchdog. Releases on done flags (not on Slurm exit, which also happens
# at walltime rollover):
#   baseline converged      -> its ANN / DisCo jobs (seeded from that baseline)
#   all baselines + md done -> scoring + evaluation (round 1)
#   all downstream done     -> scoring + evaluation (round 2), then exit
# Health check: a released model that is not done and has no queued/running
# job (same job name; v3 continuations keep it) is resubmitted, at most
# MAX_RETRY consecutive times (the count resets once a job runs > 2 h);
# after that it is given up and logged as FAILED.

source /home/yuanda/lundtoptagger/submit/lrj/v3_models.sh
MAX_RETRY=3
cd $REPO

in_queue() { squeue -h -u yuanda -n "$1" -o %i | grep -q .; }

# Seconds the model's current job has been running (0 if not running).
run_secs() {
    local st=$(squeue -h -u yuanda -n "$1" -t R -o %S | head -1)
    [ -n "$st" ] && echo $(( $(date +%s) - $(date -d "$st" +%s) )) || echo 0
}

# Storage outages make every job die at start-up (python env lives on Lustre);
# retries spent then would give up healthy models.
env_ok() {
    timeout -s KILL 300 bash -c "source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh \
        && conda activate /share/lustre/yzhu/conda/envs/torch26_py39 \
        && python -c 'import torch, torch_geometric, yaml' && ls '$V3' >/dev/null" >/dev/null 2>&1
}

# ensure_running jobname conf worker seed flagfile -> 0 if done or given up
ensure_running() {
    local jn=$1 conf=$2 worker=$3 seed=$4 flag=$5
    [ -f "$flag" ] && return 0
    [ -f "$V3/.failed_$jn" ] && return 0
    local rf="$V3/.retry_$jn"
    if in_queue "$jn"; then
        # MAX_RETRY counts consecutive failures: 2 h of running is progress
        [ -f "$rf" ] && [ "$(run_secs "$jn")" -gt 7200 ] && rm -f "$rf"
        return 1
    fi
    sleep 60   # continuation may be in flight
    if in_queue "$jn" || [ -f "$flag" ]; then return 1; fi
    if ! env_ok; then
        echo "$(date) $jn not queued, but storage/env unhealthy -> waiting (no retry spent)"
        return 1
    fi
    local n=$(cat "$rf" 2>/dev/null || echo 0)
    if [ "$n" -ge $MAX_RETRY ]; then
        echo "$(date) FAILED: $jn gave up after $n consecutive retries"
        touch "$V3/.failed_$jn"
        return 0
    fi
    echo $((n + 1)) > "$rf"
    echo "$(date) $jn not done and not queued -> resubmit #$((n + 1)): $(v3_sub $conf $jn $worker $seed)"
    return 1
}

# Continuations submitted by already-running workers come from their Slurm
# spool copy (8 CPUs); shrink them while pending.
shrink_pending() {
    squeue -h -u yuanda -t PD -o '%i %j %C' | while read -r i n c; do
        case "$n" in v3_*)
            if [ "$c" -gt $CPUS ]; then
                scontrol update JobId=$i NumCPUs=$CPUS MinCPUsNode=$CPUS CPUsPerTask=$CPUS \
                    && echo "$(date) $n ($i): CPUs $c -> $CPUS"
            fi ;;
        esac
    done
}

score_round() {  # round
    local js je
    js=$(sbatch --parsable --job-name="v3_score_r$1" \
        --output="$LOG/slurm-%j.v3_score_r$1.out" $REPO/submit/lrj/10_score_v3.sh)
    je=$(sbatch --parsable --job-name="v3_eval_r$1" --dependency=afterok:$js \
        --output="$LOG/slurm-%j.v3_eval_r$1.out" $REPO/submit/lrj/11_eval_v3.sh)
    echo "$(date) scoring round $1 -> score $js, eval $je"
}

while true; do
    shrink_pending
    n_base=0
    for b in "${BASE[@]}"; do
        read -r ch exp tag jn pubdir <<<"$b"
        ensure_running "$jn" "train_${ch}_v3_${exp}.yaml" 03_train_worker.sh "" \
            "$(exp_dir $ch $exp)/done_$tag.flag" && n_base=$((n_base + 1))
    done

    n_down=0
    for d in "${DOWN[@]}"; do
        read -r ch base exp tag jn worker <<<"$d"
        seed=$(exp_dir $ch $base)
        conf="train_${ch}_v3_${exp}.yaml"
        if [ ! -f "$V3/.gate_$jn" ]; then
            if [ -f "$seed/done_lund_only.flag" ]; then
                echo "$(date) $ch $base converged -> $jn job $(v3_sub $conf $jn $worker $seed)"
                touch "$V3/.gate_$jn"
            fi
            continue
        fi
        ensure_running "$jn" "$conf" "$worker" "$seed" \
            "$(exp_dir $ch $exp)/done_$tag.flag" && n_down=$((n_down + 1))
    done
    # downstream of a given-up baseline can never start: count it as finished
    for d in "${DOWN[@]}"; do
        read -r ch base exp tag jn worker <<<"$d"
        for b in "${BASE[@]}"; do
            read -r bch bexp btag bjn bpub <<<"$b"
            if [ "$bch" = "$ch" ] && [ "$bexp" = "$base" ] && [ -f "$V3/.failed_$bjn" ] \
               && [ ! -f "$V3/.gate_$jn" ]; then
                n_down=$((n_down + 1))
            fi
        done
    done

    if [ $n_base -eq ${#BASE[@]} ] && [ ! -f "$V3/.gate_score_r1" ]; then
        score_round 1; touch "$V3/.gate_score_r1"
    fi
    if [ $n_base -eq ${#BASE[@]} ] && [ $n_down -eq ${#DOWN[@]} ]; then
        score_round 2
        echo "$(date) All v3 models finished. Watchdog exiting."
        exit 0
    fi
    if [ $SECONDS -gt 165600 ]; then   # 46 h
        echo "$(date) base $n_base/${#BASE[@]} down $n_down/${#DOWN[@]} - resubmitting watchdog."
        sbatch "$REPO/submit/lrj/09_gate_v3.sh"
        exit 0
    fi
    sleep 300
done
