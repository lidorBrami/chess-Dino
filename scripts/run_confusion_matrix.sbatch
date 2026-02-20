#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --job-name=conf_mat
#SBATCH --output=logs/conf_mat-%j.out
#SBATCH --error=logs/conf_mat-%j.err
#SBATCH --gpus=1
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate detrex

python report/generate_confusion_matrix.py
