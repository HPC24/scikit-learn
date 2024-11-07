#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024
 
#SBATCH --job-name=profiling
#SBATCH --time=0-04:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=7900MB

#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

$CONDA_ENV="sklearn-env"

# Threads used for the Profiling
PROFILING_THREADS=6

IMPLEMENTATION="assign_centroids"

SIMD="_simd"

# VTune Parameters
PROFILING_RESULTS_DIR="vtune_results"
ANALYSIS_TYPE="hotspots"

OUTPUT_FILE="sklearn${SIMD}_${IMPLEMENTATION}"

# Paths
EXECUTABLE="timing.py"

C_COMPILER_FLAGS="-march=native -mtune=native"

export C_COMPILER_FLAGS=${C_COMPILER_FLAGS}

VTUNE_OUTPUT_DIRECTORY=${PROFILING_RESULTS_DIR}_${OUTPUT_FILE}_OMP_${PROFILING_THREADS}
OUTPUT_DIR="${VTUNE_OUTPUT_DIRECTORY}/out"

echo "Creating Output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Creating Vtune directory: ${VTUNE_OUTPUT_DIRECTORY}"
mkdir -p  "${VTUNE_OUTPUT_DIRECTORY}"

# Start with slurm specific commands
module purge
module add slurm
module add miniconda3
module add intel-oneapi-vtune/2024.1.0

source deactivate

echo "Activating ${CONDA_ENV}"
source activate ${CONDA_ENV}


echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Maximal threads per process: $SLURM_CPUS_PER_TASK"
echo "Current working directory is `pwd`" 


echo "Starting Vtune"
echo "Collecting: ${ANALYSIS_TYPE}"
echo "OMP_NUM_THREADS: ${PROFILING_THREADS}"

export OMP_NUM_THREADS=${PROFILING_THREADS}

vtune -collect ${ANALYSIS_TYPE} \
    -result-dir ${VTUNE_OUTPUT_DIRECTORY} \
    -- ${EXECUTABLE} \
    --output_dir ${OUTPUT_DIR} \
    --output_file ${OUTPUT_FILE} \
    -- ${IMPLEMENTATION}

echo "finished profiling"


