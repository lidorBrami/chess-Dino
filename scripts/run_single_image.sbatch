#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --job-name=single_img
#SBATCH --output=logs/single_img-%j.out
#SBATCH --error=logs/single_img-%j.err
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

python3 report/generate_single_image_pred.py
