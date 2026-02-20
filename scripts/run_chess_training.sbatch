#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dino_chess
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "============================================"

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate detrex

python projects/dino/train_net.py \
    --config-file projects/dino/configs/dino_swin_large_chess_finetune.py \
    --num-gpus 2

echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"
