#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024
 
#SBATCH --job-name=profiling
#SBATCH --time=0-08:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=7900MB

#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

OUTPUT_DIR="./out"
OUTPUT_FILE="sklearn"
BUILD_DIR=$(realpath build_pybind)

CONDA_ENV="sklearn-env"
PYTHON_FILE="src/timing.py"

# which implementation of the lloyd algorithm that is used inside sklearn
IMPLEMENTATION="use_assign_centroids"

if [ ${IMPLEMENTATION} == "use_assign_centroids" ]; then
    echo "Using custom assign_centroids function"
elif [ ${IMPLEMENTATION} == "use_assign_centroids_gemm" ]
     echo "Using custom assing_centroids_gemm function"
else
    echo "Using standard scikit-learn implementation"
fi

# slurm specific 
module purge
module add slurm 
module add miniconda3
source deactivate

echo "using conda environment ${CONDA_ENV}"
source activate ${CONDA_ENV}

echo "Creating output directory for timings"
mkdir -p ${OUTPUT_DIR}


echo "Starting timing of sklearn KMeans implementation for up to ${SLURM_CPUS_PER_TASK}"

for N in $(seq 1 ${SLURM_CPUS_PER_TASK});
do
    export OMP_NUM_THREADS=${N}
    echo "Starting timing for ${N} OMP_NUM_THREADS"

    python ${PYTHON_FILE} --output_dir ${OUTPUT_DIR} --output_file ${OUTPUT_FILE} --${IMPLEMENTATION}

done

echo "Finished sklearn KMeans timing"

