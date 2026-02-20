#!/bin/bash
#SBATCH --time 2-00:00:00                       ### 2 days time limit
#SBATCH --job-name dino_chess                   ### name of the job
#SBATCH --output /home/lidorbr/detrex/sbatch_logs/log-%J.out    ### output log
#SBATCH --error /home/lidorbr/detrex/sbatch_logs/log-%J.err     ### error log

#SBATCH --gpus=1                                ### number of GPUs
#SBATCH --partition=rtx4090
#SBATCH --account=azencot
#SBATCH --qos=azencot
#SBATCH --mem=32G                               ### amount of RAM memory
#SBATCH --cpus-per-task=4                       ### number of CPU cores

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
source ~/.bashrc
conda activate detrex

cd /home/lidorbr/detrex

python tools/train_net.py \
    --config-file projects/dino/configs/dino_swin_large_chess_finetune.py \
    --num-gpus 1

echo 'done'
echo `date`

