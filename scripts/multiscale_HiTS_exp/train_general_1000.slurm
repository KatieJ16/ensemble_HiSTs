#!/bin/bash
#SBATCH --job-name=train_general_vanderpol_10      # Job name
#SBATCH --account=amath
#SBATCH --partition=ckpt            #gpu-rtx6k
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kjohnst@uw.edu    # Where to send mail.  Set this to your email address
#SBATCH --time=02-00:00:00            # Wall time limit (days-hrs:min:sec)
#SBATCH --output=train_egernal_vanderpol%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

srun python -u train_general_1000.py
