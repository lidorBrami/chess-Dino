#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --job-name=single_pred
#SBATCH --output=logs/single_pred-%j.out
#SBATCH --error=logs/single_pred-%j.err
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

python3 -c "
import sys, os, numpy as np, torch
from PIL import Image
sys.path.insert(0, 'inference')
from predict_board import predict_board
from predict_board_no_ood import predict_board as predict_board_no_ood
from cfg import CLASS_TO_FEN

img_path = 'data/dino/val/game2/game8/frame_013240_jpg.rf.43c285e640b973e228d04a8aef2fe86c.jpg'
img = np.array(Image.open(img_path).convert('RGB'))

print('=== WITH OOD ===')
board_ood = predict_board(img)
for r in range(8):
    row = ''
    for c in range(8):
        v = board_ood[r][c].item()
        if v == 13: row += 'X '
        elif v == 12: row += '. '
        else: row += CLASS_TO_FEN.get(v, '?') + ' '
    print(row)

print()
print('=== WITHOUT OOD ===')
board_no = predict_board_no_ood(img)
for r in range(8):
    row = ''
    for c in range(8):
        v = board_no[r][c].item()
        if v == 13: row += 'X '
        elif v == 12: row += '. '
        else: row += CLASS_TO_FEN.get(v, '?') + ' '
    print(row)
"
