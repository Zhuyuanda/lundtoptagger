#!/bin/bash
#SBATCH --job-name=srj_preprocess
#SBATCH --partition=RCIF        # CPU 分区
# exclude nodes that do NOT mount /share/lustre properly
#SBATCH --exclude=compute-0-21
#SBATCH -N 1
#SBATCH -n 8                   # 16 CPU cores
#SBATCH --mem=50G              # 50GB (kde flattening)
#SBATCH --export=ALL
#SBATCH --output=/home/yuanda/srj/lundtoptagger/preprocess_log/slurm-%j.out
#SBATCH --error=/home/yuanda/srj/lundtoptagger/preprocess_log/slurm-%j.err
#SBATCH --mail-user=ucaphue@ucl.ac.uk
#SBATCH --mail-type=ALL
# submit by sbatch submit_preprocess.sh

echo "===> Hostname:"
hostname

echo "===> Activating environment"
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo "Using environment: $CONDA_DEFAULT_ENV"

echo "===> Running CPU preprocess script..."
cd /home/yuanda/srj/lundtoptagger

python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

echo "===> Job finished."
