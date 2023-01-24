#!/bin/bash

#need step_size first, then noise level , then system, then letter. i.e. sbatch train_general_original.slurm 1 0.0 KS a


#SBATCH --job-name=train     # Job name
#SBATCH --account=amath
#SBATCH --partition=ckpt            #gpu-rtx6k
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kjohnst@uw.edu    # Where to send mail.  Set this to your email address
#SBATCH --time=01-00:00:00            # Wall time limit (days-hrs:min:sec)
#SBATCH --output=train%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

echo "job name = $SLURM_JOB_NAME"
echo "job number = $SLURM_JOB_ID"


srun python -u train_model_original.py -ss $1 -n $2 -s $3 -l $4