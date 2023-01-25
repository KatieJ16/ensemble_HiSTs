#!/bin/bash
#SBATCH --job-name=make_data      # Job name
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kjohnst@uw.edu    # Where to send mail.  Set this to your email address
#SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically first among nodes and then among sockets within a node
#SBATCH --time=01:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=make_data%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

srun python generate_data_lorenz.py