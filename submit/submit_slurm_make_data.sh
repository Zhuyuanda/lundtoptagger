#!/bin/bash

# Submission command:   sbatch submit_slurm_make_data.sh

#SBATCH --job-name=make_data_dijet
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -n4
#SBATCH --mem=256G

# Your config has 25 event_fraction slices
#SBATCH --array=0-24

#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL

#SBATCH --output=/home/yuanda/srj/lundtoptagger/log/slurm-%j.%a.%x.out


# ------------------------------------------
# Environment and Paths
# ------------------------------------------
cd /home/yuanda/srj/lundtoptagger
echo "Now in $(pwd)"
hostname

echo "Activating environment..."
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo "Conda env: $CONDA_DEFAULT_ENV"


# ------------------------------------------
# Parameters
# ------------------------------------------
main_config="configs/config_make_data_SRJ.yaml"
signal_config="configs/config_signal_SRJ.yaml"

event_fraction_idx=${SLURM_ARRAY_TASK_ID}

echo "Using main config:   $main_config"
echo "Using signal config: $signal_config"
echo "event_fraction_idx:  $event_fraction_idx"


# ------------------------------------------
# Run MakeData
# ------------------------------------------
python Make_data_SRJ.py "$main_config" --override \
    signal_config_file="$signal_config" \
    id="dijet" \
    signal="srj" \
    event_fraction_idx="$event_fraction_idx"

echo "Job done."
