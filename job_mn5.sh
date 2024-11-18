#!/bin/bash
#SBATCH -A bsc32 
#SBATCH -n 1
#SBATCH -c 112
#SBATCH --time 2:00:00
#SBATCH -J NINO
#SBATCH --output /gpfs/scratch/bsc32/bsc032259/output/job_output/MPI_job%J.out
#SBATCH --error /gpfs/scratch/bsc32/bsc032259/output/job_output/MPI_job%J.err
#SBATCH -q gp_debug

# inter=$1

source /gpfs/home/bsc/bsc032259/mambaforge/bin/activate
conda activate esmvaltool
# conda activate esmvaltool_plus_others
#SBATCH -q gp_bsces


esmvaltool run /gpfs/scratch/bsc32/bsc032259/jmindlin/recipe_nino_trends.yml --config_file /gpfs/scratch/bsc32/bsc032259/config_files/config-user.yml
