#!/bin/bash
# Submission: sbatch submit/lrj/01_make_data.sh
# W tagging: SIGNAL=W DATA_ID=WTagging_LRJ sbatch --export=ALL,SIGNAL,W,DATA_ID submit/lrj/01_make_data.sh
# (also add 801859 ROOT to path_to_rootfiles via config_make_data_LRJ.yaml or --override)

#SBATCH --job-name=LRJ_make_data
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --array=0-24%5
#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%A_%a.out

SIGNAL=${SIGNAL:-top}
DATA_ID=${DATA_ID:-TopTagging_LRJ}

cd /home/yuanda/srj/lundtoptagger
echo "Now in $(pwd)"
hostname
echo "Signal: $SIGNAL | Dataset id: $DATA_ID"

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

main_config="configs/config_make_data_LRJ.yaml"
signal_config="configs/config_signal_LRJ.yaml"
event_fraction_idx=${SLURM_ARRAY_TASK_ID}

sleep $((RANDOM % 30))
echo "Running index: $event_fraction_idx"

python Make_data_LRJ.py "$main_config" --override \
    signal_config_file="$signal_config" \
    id="$DATA_ID" \
    signal="$SIGNAL" \
    event_fraction_idx="$event_fraction_idx"

echo "Job done."
