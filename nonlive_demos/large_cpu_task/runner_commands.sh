# The disBatch module is not loaded by default:
module load disBatch
mkdir disbatch_logs

# Run the job with a pool of 40 cpu cores
# Assumes cwd is disbatch-tutorial/nonlive_demos/large_cpu_task
# sbatch -p genx -c 40 --mem=64GB -t 0-1 disBatch disbatch_script -p disbatch_logs/ -t 10
# Argument breakdown:
# -p genx: use the genx partition (non-exclusive allocations to cpu-only nodes for small-ish jobs)
# -c 40: request 40 cpu cores
# --mem=64GB: request 64GB of memory
# -t 0-1: request a maximum of 1 hour of walltime  (days-hours)
# {disBatch disbatch_script -p disbatch_logs/ -t 10}: command to run on the allocated resources
#  -p disbatch_logs/: path to the directory where disBatch will store its logs
#  -t 10: maximum number of tasks to run in parallel (This should be automatically set to the number of cores requested, but I had trouble getting that to work on genx)


# Run the job with a pool of 80+ cpu cores in an exclusive allocation
# Assumes cwd is disbatch-tutorial/nonlive_demos/large_cpu_task
sbatch -p gen -t 0:15 disBatch disbatch_script -p disbatch_logs/ -c 4
# Argument breakdown:
# -p gen: use the gen partition (exclusive allocations to cpu-only nodes for larger jobs)
# -t 0:15: request a maximum of 15 min of walltime (hours:minutes)
# {disBatch disbatch_script -p disbatch_logs/}: command to run on the allocated resources
#  -p disbatch_logs/: path to the directory where disBatch will store its logs
#  -c 4: Each task needs 4 cpu cores. disBatch will see the total number of cores available in the allocation and run at most C/4 tasks in parallel,