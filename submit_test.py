#! /bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 1:00:00

# This is where you specify the number of cores
# you want. It starts at 0 so if you 4 cores,
# enter 0-3
#SBATCH --array=0-5
#SBATCH -o %a.out
#SBATCH -e %a.err

# This is to get e-mail notifications
# when the jobs start and end
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=lisacm@princeton.edu

# Name of the job as it appears in squeue
#SBATCH --job-name=test

#! DELLA
# Load python 2.7
module load python 
module load anaconda

# The first argument is the total number of cores you
# want but we get it from a SLURM environment variable
python routine.py $((SLURM_ARRAY_TASK_MAX+1)) $SLURM_ARRAY_TASK_ID