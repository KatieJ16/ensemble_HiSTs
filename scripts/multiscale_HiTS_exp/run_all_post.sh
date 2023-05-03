#!/bin/bash

#inputs need to be noise system letter smallest_step

system=$1
letter=$2
smallest_step=$3

for noise in 0.0 0.05 0.1 0.2
do
   echo sbatch post_both.slurm $noise $system $letter $smallest_step

    #run the depends method
    sbatch post_both.slurm $noise $system $letter $smallest_step

done