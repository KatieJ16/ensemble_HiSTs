#!/bin/bash

#need noise level first, then system, then letter. i.e. sbatch train_general.slurm 0.0 KS a




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
#SBATCH --cpus-per-task=4

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

echo "job name = $SLURM_JOB_NAME"
echo "job number = $SLURM_JOB_ID"
echo "noise = $1"
echo "system = $2"
echo "letter = $3"

echo "srun python -u train_general.py -n $1 -s $2 -l $3"

srun python -u train_general.py -n $1 -s $2 -l $3