#!/bin/bash



#SBATCH --job-name=train     # Job name
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kjohnst@uw.edu    # Where to send mail.  Set this to your email address
#SBATCH --time=00-02:00:00            # Wall time limit (days-hrs:min:sec)
#SBATCH --output=train%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

echo "job name = $SLURM_JOB_NAME"
echo "job number = $SLURM_JOB_ID"


srun python -u post_original.py -n $1 -s $2 -l $3