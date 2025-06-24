#!/bin/bash
#SBATCH -p genx
#SBATCH -c 4
#SBATCH --mem=8GB
#SBATCH --time=0:15
#SBATCH --array=0-39


# Try to avoid this. Only doing this for educational purposes.

# Add 1 to the task id to word with sed's 1-indexing
task_id=$((SLURM_ARRAY_TASK_ID + 1))
# Grab line n from the disbatch_script
sed_arg="${task_id}q;d"
command=$(sed "$sed_arg" disbatch_script)
# Run it
eval "$command"