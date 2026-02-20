#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --job-name=abl_nopre
#SBATCH --output=logs/abl_nopre-%j.out
#SBATCH --error=logs/abl_nopre-%j.err
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate detrex

python projects/dino/train_net.py \
    --config-file projects/dino/configs/dino_swin_large_chess_ablation_no_pretrain.py \
    --num-gpus 2
