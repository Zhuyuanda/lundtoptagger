#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# v3 evaluation: per-pT-bin 50% WPs, pT-parameterized inclusive ROC, mass
# JSD and Pareto, physics weights matched from the raw test_1M.h5.

REPO=/home/yuanda/lundtoptagger
V3=/share/lustre/yzhu/IOP_DATA/v3_pub
OUT=/home/yuanda/pubplot/lundtag_compare/v3
cd $REPO
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/lustre/yzhu/conda/envs/torch26_py39
set -e

python -W ignore plot_lundtag_compare_binnedwp.py --scores-h5 $V3/scores_W.h5 \
    --channel W --sig-def WZ    --outdir $OUT/W_WZ
python -W ignore plot_lundtag_compare_binnedwp.py --scores-h5 $V3/scores_W.h5 \
    --channel W --sig-def Wonly --outdir $OUT/W_Wonly
python -W ignore plot_lundtag_compare_binnedwp.py --scores-h5 $V3/scores_Top.h5 \
    --channel top --outdir $OUT/Top
