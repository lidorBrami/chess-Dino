#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --job-name=chess_infer
#SBATCH --output=logs/infer-%j.out
#SBATCH --error=logs/infer-%j.err
#SBATCH --gpus=1
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "============================================"
echo "Job started at: $(date)"
echo "============================================"

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate detrex

python projects/dino/inference_chess.py \
    --image "$1" \
    --output "${2:-output/result.jpg}" \
    --threshold "${3:-0.3}"

echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"
