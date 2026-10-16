#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|l40s|v100'
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Score pubnote test shards with every finished v3 model (+ pubnote LundNet).
# Columns already present in scores_{ch}.h5 are skipped, so reruns only add
# newly finished models.

REPO=/home/yuanda/lundtoptagger
P=/share/lustre/yzhu/pubnote
V3=/share/lustre/yzhu/IOP_DATA/v3_pub
cd $REPO
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/lustre/yzhu/conda/envs/torch26_py39
set -e

python -W ignore tools/score_h5_multi.py \
    --test-dir $P/test_LRJ_W_801859_20M \
    --test-meanstd $P/preprocessed_LRJ_W_801859_20M/train/meanstd.json \
    --out $V3/scores_W.h5 \
    --configs configs/v3/train_W_v3_*.yaml \
              pubnote=configs/pubnote/train_LRJ_W_801859_20M_lund_only.yaml \
              pubnote_md=configs/pubnote/train_LRJ_W_802017_20M_lund_only.yaml

python -W ignore tools/score_h5_multi.py \
    --test-dir $P/test_LRJ_Top_20M \
    --test-meanstd $P/preprocessed_LRJ_Top_20M/train/meanstd.json \
    --out $V3/scores_Top.h5 \
    --configs configs/v3/train_Top_v3_*.yaml \
              pubnote=configs/pubnote/train_LRJ_Top_20M_lund_only.yaml
