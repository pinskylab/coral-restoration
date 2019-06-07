#! /bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH --partition=stf
#SBATCH --account=stf

# This is to get e-mail notifications
# when the jobs start and end
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ldefilip@uw.edu

# Name of the job as it appears in squeue
#SBATCH --job-name=restoration

# The first argument is the total number of cores you
# want but we get it from a SLURM environment variable
python routine.py