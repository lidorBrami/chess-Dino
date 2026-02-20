#!/bin/bash
#SBATCH --time=0-00:15:00
#SBATCH --job-name=eval_noood
#SBATCH --output=logs/eval_noood-%j.out
#SBATCH --error=logs/eval_noood-%j.err
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

cd inference
python eval_fen_no_ood.py
