#!/bin/bash
# lundtag_compare v3: submit baselines + flat-mass (md) models on the pubnote
# H5 shards, then start the gate that releases ANN/DisCo and scoring.
#
#   bash submit/lrj/08_submit_v3.sh
#
# Configs: configs/v3/train_{Top,W}_v3_<model>.yaml (make_v3_configs.py)
# Models : /share/lustre/yzhu/IOP_DATA/v3_pub/models_{Top,W}/exp_<model>/nparts1
#
# To stop a v3 training chain: touch its done flag first, then scancel
# (a plain scancel triggers the worker's continuation trap).

source /home/yuanda/lundtoptagger/submit/lrj/v3_models.sh
mkdir -p $LOG $V3
cd $REPO

for b in "${BASE[@]}"; do
    read -r ch exp tag jn pubdir <<<"$b"
    d=$(exp_dir $ch $exp); mkdir -p "$d"
    # same training shards as pubnote -> identical sig/bkg scale factor
    [ -f "$d/scale_factor.json" ] || cp "$pubdir/exp_lund_only/nparts1/scale_factor.json" "$d/"
    echo "$ch $exp -> $(v3_sub train_${ch}_v3_${exp}.yaml $jn 03_train_worker.sh)"
done

echo "gate -> $(sbatch --parsable $REPO/submit/lrj/09_gate_v3.sh)"
