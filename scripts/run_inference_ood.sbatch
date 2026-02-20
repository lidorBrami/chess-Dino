#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --job-name=ood_infer
#SBATCH --output=/home/lidorbr/detrex/logs/infer_ood-%j.out
#SBATCH --error=/home/lidorbr/detrex/logs/infer_ood-%j.err
#SBATCH --gpus=1
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "============================================"
echo "OOD Inference started at: $(date)"
echo "============================================"

cd /home/lidorbr/detrex
source ~/.bashrc
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate detrex

# Run OOD inference on test image
# New logic:
#   - piece-threshold: high confidence detections are chess pieces
#   - ood-score-threshold: minimum score to consider as potential OOD
#   - ood-mahal-threshold: Mahalanobis distance above this = OOD
python projects/dino/inference_chess_ood.py \
    --image /home/lidorbr/detrex/data/test/test/frame_000640_jpg.rf.29da4092a040f7353293891bca8ae526.jpg \
    --output /home/lidorbr/detrex/output/frame_000640_ood_result.jpg \
    --piece-threshold 0.5 \
    --ood-score-threshold 0.01 \
    --ood-mahal-threshold 70.0 \
    --ood-calibration /home/lidorbr/detrex/output/ood_calibration.pth

echo "============================================"
echo "OOD Inference finished at: $(date)"
echo "============================================"

