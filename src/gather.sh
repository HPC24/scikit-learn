#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024

#SBATCH --job-name=gathering
#SBATCH --time=0-00:05:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=7900MB

#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

# Command to move all the files from the different build directories into the ./out directory 
# find -wholename "*out/*timings.txt" -exec cp {} ./out \;

PYTHON_FILE="./gather.py"
CONDA_ENV="sklearn-env"
BUILD_DIR_PREFIX="./build"
DATA_DIR_NAME="out"
VISUALIZATION_DIR="./Visualization"

BUILD_DIRS=$(ls -d ${BUILD_DIR_PREFIX}* | xargs realpath)
#DATA_DIRS=$(echo "${BUILD_DIRS}" | sed "s/\([^ ]*\)/\1\/${DATA_DIR_NAME}/g")
#DATA_DIRS="${DATA_DIRS} $(realpath sklearn_timings)"
DATA_DIRS="out"
echo "Creating Visualization directory"

mkdir -p ${VISUALIZATION_DIR}

# Slurm specific module loading 
module purge 
module add slurm 
module add miniconda3 
source deactivate 

echo "Activating conda environment ${CONDA_ENV}"
source activate ${CONDA_ENV}

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Maximal threads per process: $SLURM_CPUS_PER_TASK"
echo "Current working directory is `pwd`"


echo "Pulling data from following Data Directories: ${DATA_DIRS}"

python ${PYTHON_FILE} --data_directory ${DATA_DIRS} --output_directory ${VISUALIZATION_DIR}

echo "Finished data gathering"
echo "Results located in ${VISUALIZATION_DIR}"








