#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --job-name=fen_fig
#SBATCH --output=logs/fen_fig-%j.out
#SBATCH --error=logs/fen_fig-%j.err
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

python report/generate_fen_example.py
