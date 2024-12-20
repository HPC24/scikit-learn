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
BUILD_DIR="/home/kurs_2024_sose_hpc/kurs_2024_sose_hpc_11/scikit-learn"

CONDA_ENV="sklearn-env"
PYTHON_FILE="timing.py"
CWD=$(pwd)

# which implementation of the lloyd algorithm that is used inside sklearn
IMPLEMENTATION=""

C_COMPILER_FLAGS="-march=native -mtune=native"
export C_COMPILER_FLAGS=${C_COMPILER_FLAGS}

if [ ${IMPLEMENTATION} == "assign_centroids" ]; then
    echo "Using custom assign_centroids function"
elif [ ${IMPLEMENTATION} == "assign_centroids_gemm" ]; then
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

echo "Uninstalling build dir: ${BUILD_DIR}"
cd /home/kurs_2024_sose_hpc/kurs_2024_sose_hpc_11/scikit-learn
rm -r build

echo "Generating meson_options.txt file"
echo "Using C_COMPILER_FLAGS: ${C_COMPILER_FLAGS}"
echo "option('C_COMPILER_FLAGS', type: 'string', value: '${C_COMPILER_FLAGS}', description: 'Custom C compiler flags for the project')" > meson_options.txt

echo "Uninstalling scikit-learn"
pip uninstall scikit-learn -y

echo "Installing scikit-learn"
pip install --editable . \
   --verbose --no-build-isolation \
   --config-settings editable-verbose=true

cd ${CWD}

echo "Importing KMeans to compile the necessary files"
python -c "from sklearn.cluster import KMeans"

echo "Starting timing of sklearn KMeans implementation for up to ${SLURM_CPUS_PER_TASK}"

for N in $(seq 1 ${SLURM_CPUS_PER_TASK});
do
    export OMP_NUM_THREADS=${N}
    echo "Starting timing for ${N} OMP_NUM_THREADS"

    python ${PYTHON_FILE} --output_dir ${OUTPUT_DIR} --output_file ${OUTPUT_FILE} --${IMPLEMENTATION}

done

echo "Finished sklearn KMeans timing"

