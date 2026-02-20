#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --job-name=ood_calib
#SBATCH --output=/home/lidorbr/detrex/logs/calibrate-%j.out
#SBATCH --error=/home/lidorbr/detrex/logs/calibrate-%j.err
#SBATCH --gpus=1
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "============================================"
echo "OOD Calibration started at: $(date)"
echo "============================================"

cd /home/lidorbr/detrex
source ~/.bashrc
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate detrex

python projects/dino/calibrate_ood.py \
    --config projects/dino/configs/dino_swin_large_chess_finetune.py \
    --checkpoint /home/lidorbr/detrex/output/dino_chess_finetune/model_final.pth \
    --output /home/lidorbr/detrex/output/ood_calibration.pth \
    --num-classes 13 \
    --feature-dim 256 \
    --min-score 0.7

echo "============================================"
echo "OOD Calibration finished at: $(date)"
echo "============================================"


