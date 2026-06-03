#!/bin/bash

LOG_DIR="/home/yuanda/srj/lundtoptagger/lundlog"
SSH_OPT="-q -o ConnectTimeout=2 -o StrictHostKeyChecking=no"

while true; do
    clear
    echo "==============================================================================="
    echo "  LRJ LundNet Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================================="

    echo -e "\n[1. SLURM Queue Status]"
    squeue -u "$USER" | grep -E "JOBID|LRJ_train" || true

    echo -e "\n[2. Memory Usage (Real RSS from Node Kernel)]"
    printf "%-10s %-15s %-12s %-10s %-10s\n" "JOBID" "NODE" "RES_MEM" "LIMIT" "USAGE%"
    for jobid in $(squeue -u "$USER" -h -t R -o "%i %j" | grep LRJ_train | awk '{print $1}'); do
        node=$(squeue -j "$jobid" -h -o %N)
        res_gb=$(ssh $SSH_OPT "$node" "ps -u $USER -o rss,command | grep weight_ONLY_TRAINS_LRJ.py | grep -v grep" | awk '{sum+=$1} END {if(sum) printf "%.2f", sum/1024/1024; else print "0"}')
        usage_pct=$(awk "BEGIN {print ($res_gb/200)*100}")
        printf "%-10s %-15s %-12sG %-10s %-10.1f%%\n" "$jobid" "$node" "$res_gb" "200G" "$usage_pct"
    done

    echo -e "\n[3. Latest Log Progress]"
    ls -t "${LOG_DIR}"/slurm-*.out 2>/dev/null | head -n 5 | while read -r logfile; do
        fname=$(basename "$logfile")
        last_line=$(tail -n 1 "$logfile" | cut -c 1-80)
        echo -e "  > $fname: $last_line"
    done

    echo -e "\n[4. Health Check]"
    oom_count=$(grep -l "Out of memory" "${LOG_DIR}"/slurm-*.out 2>/dev/null | wc -l)
    if [ "$oom_count" -gt 0 ]; then
        echo "!! WARNING: $oom_count jobs detected OOM error !!"
    else
        echo "  Check: No OOM detected."
    fi

    echo -e "\n==============================================================================="
    echo "  Ctrl+C to exit. Refreshing every 60s..."
    sleep 60
done
