#!/bin/bash

# 定义日志目录
LOG_DIR="/home/yuanda/srj/lundtoptagger/lundlog"

# 解决 SSH 报错的小技巧：添加 -o ConnectTimeout=2 快速跳过不通的节点
# 并在查询时静默 host key 警告
SSH_OPT="-q -o ConnectTimeout=2 -o StrictHostKeyChecking=no"

while true; do
    clear
    echo "==============================================================================="
    echo "  LundNet Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================================="
    
    # 1. 显示 SLURM 队列状态
    echo -e "\n[1. SLURM Queue Status]"
    squeue -u $USER | grep -E "JOBID|SRJ_lund"
    
    # 2. 检查内存占用 (通过 SSH 穿透查询真实 RSS)
    echo -e "\n[2. Memory Usage (Real RSS from Node Kernel)]"
    printf "%-10s %-15s %-12s %-10s %-10s\n" "JOBID" "NODE" "RES_MEM" "LIMIT" "USAGE%"
    for jobid in $(squeue -u $USER -h -t R -o %i -n SRJ_lund); do
        node=$(squeue -j $jobid -h -o %N)
        # 获取物理内存占用 (GB)
        res_gb=$(ssh $SSH_OPT $node "ps -u $USER -o rss,command | grep weight_ONLY_TRAINS_LRJ_fixed.py | grep -v grep" | awk '{sum+=$1} END {if(sum) printf "%.2f", sum/1024/1024; else print "0"}')
        
        # 计算百分比 (基于 200G)
        usage_pct=$(awk "BEGIN {print ($res_gb/200)*100}")
        
        # 根据占用率改变颜色 (可选)
        printf "%-10s %-15s %-12sG %-10s %-10.1f%%\n" "$jobid" "$node" "$res_gb" "200G" "$usage_pct"
    done

    # 3. 检查日志最后一行 (进度追踪)
    echo -e "\n[3. Latest Log Progress]"
    # 按修改时间排序日志，查看最新的 5 个
    ls -t ${LOG_DIR}/slurm-*.out 2>/dev/null | head -n 5 | while read logfile; do
        fname=$(basename $logfile)
        last_line=$(tail -n 1 "$logfile" | cut -c 1-80) # 限制长度防止刷屏
        echo -e "  > $fname: $last_line"
    done

    # 4. 预警：检查是否存在 OOM 或崩溃
    echo -e "\n[4. Health Check]"
    oom_count=$(grep -l "Out of memory" ${LOG_DIR}/slurm-*.out 2>/dev/null | wc -l)
    if [ "$oom_count" -gt 0 ]; then
        echo "!! WARNING: $oom_count jobs detected OOM error !!"
    else
        echo "  Check: No OOM detected."
    fi

    echo -e "\n==============================================================================="
    echo "  Ctrl+C to exit. Refreshing every 60s..."
    sleep 60
done