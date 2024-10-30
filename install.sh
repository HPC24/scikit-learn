#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024
 
#SBATCH --job-name=environment_setup
#SBATCH --time=0-00:05:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7900MB

#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

CONDA_ENV="sklearn-env"

module purge
module add miniconda3
module add cmake/3.27.9
module add ninja/1.11.1
module add openblas/0.3.26

source deactivate
conda activate ${CONDA_ENV}

echo "Using Conda Environment ${CONDA_ENV}"

find . -exec touch {} +

pip3 install --editable . \
	    --verbose \
	    --no-build-isolation \
	     --config-settings editable-verbose=true



