import pathlib
import argparse
import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


chunk_sizes = [4, 8 , 16, 32, 64, 128, 256, 512, 1024]
TIMING_ITERATIONS = 5
KMEANS_INIT="random"
KMEANS_N_INIT=1
N_CLUSTER = 10
TOL = 1e-9
MAX_ITER = 500
SEED = 42


def match_labels(labels_true, labels_pred):
    """
    Permute labels of labels_pred to match labels_true as much as possible.
    """
    cm = confusion_matrix(labels_true, labels_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {old_label: new_label for old_label, new_label in zip(col_ind, row_ind)}
    new_labels_pred = np.array([mapping[label] for label in labels_pred])
    return new_labels_pred


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir",
                        required = True,
                        help = "Path to the output directory where the results should be saved"
                        )
                        
    parser.add_argument("--output_file",
			required = True,
			help = "Name of the output file where the results will be saved"
			)
    
    parser.add_argument("--assign_centroids", 
                        action="store_true",
                        help="Set to True to use assign centroids Implementation")
    
    parser.add_argument("--assign_centroids_gemm", 
                        action="store_true",
                        help="Set to True to use assign centroids gemm Implementation")
    
    return parser.parse_args()


def fit_kmeans(kmeans, data) -> float:
    
    start_time = time.perf_counter()
    
    kmeans.fit(data)
    
    end_time = time.perf_counter()
    # get the fit time in ms 
    fit_time = (end_time - start_time) * 1000
    
    iterations = kmeans.n_iter_
    
    return fit_time, iterations
    
    
def main():
    
    # custom_cache_dir = "/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_11/data"
    custom_cache_dir = "/home/joshua/Projects/HPC_Project/data"
    cmd_args = parse_args()
    
    output_dir = pathlib.Path(cmd_args.output_dir)
    assert output_dir.is_dir(), f"{output_dir} does not exist" 
    
    compiler_flags = os.environ.get("C_COMPILER_FLAGS")
    
    if compiler_flags is None:
        
        print("No additional Compilation arguments set")
        
        if cmd_args.assign_centroids: 
            output_file = output_dir / (cmd_args.output_file + "_assign_centroids_timings.txt")
        elif cmd_args.assign_centroids_gemm:
            output_file = output_dir / (cmd_args.output_file + "_assign_centroids_gemm_timings.txt")
        else:
            output_file = output_dir / (cmd_args.output_file + "_timings.txt")
    else:
        
        print(f"Compilation flags: {compiler_flags}")
        
        if "-march=native" in compiler_flags and "-mtune=native" in compiler_flags:
            print("Architecture Optimization enabled")
            arch_opt = "_arch_opt"
            compiler_flags = "_".join([flag.replace("-", "") for flag in compiler_flags.split() if flag != "-march=native" and flag != "-mtune=native"])
        else:
            print("No Architecture Optimization enabled")
            arch_opt = "_no_arch_opt"
            
        if cmd_args.assign_centroids: 
            output_file = output_dir / (cmd_args.output_file + "_assign_centroids_" + compiler_flags + arch_opt + "_timings.txt")
        elif cmd_args.assign_centroids_gemm:
            output_file = output_dir / (cmd_args.output_file + "_assign_centroids_gemm_" + compiler_flags + arch_opt + "_timings.txt")
        else:
            output_file = output_dir / (cmd_args.output_file + compiler_flags + arch_opt + "_timings.txt")
        

    if output_file.exists():   
        print(f"{output_file} already exists")
    else:
        print(f"Creating output file: {output_file}")
        output_file.touch()
    
    timings = []
    iterations = []
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    
    print("Fetching MNIST dataset if it is not already loaded")
    
    mnist = fetch_openml('mnist_784', data_home = custom_cache_dir)
    data = mnist.data.to_numpy()
    
    print("Done")
    
    print("Initializing KMeans object with following Parameters:")
    print(f"N_CLUSTER: {N_CLUSTER}")
    print(f"INIT: {KMEANS_INIT}")
    print(f"N_INIT: {KMEANS_N_INIT}")
    print(f"TOLERANCE: {KMEANS_N_INIT}")
    print(f"MAX_ITER: {MAX_ITER}")
    print(f"SEED: {SEED}")

    print(f"Starting timing with {omp_num_threads} OMP_NUM_THREADS")
    
    for chunk_size in chunk_sizes:
        
        timings = []
        iterations = []
    
        for i in range(TIMING_ITERATIONS):
            
                kmeans = KMeans(n_clusters = N_CLUSTER, 
                        init = KMEANS_INIT, 
                        n_init = KMEANS_N_INIT, 
                        tol = TOL,
                        max_iter = MAX_ITER,
                        random_state = SEED,
                        use_assign_centroids= cmd_args.assign_centroids,
                        use_assign_centroids_gemm = cmd_args.assign_centroids_gemm,
                        chunk_size = chunk_size)
                
                time, iteration = fit_kmeans(kmeans, data = data)
                timings.append(time)
                iterations.append(iteration)
                
        iterations_second = [iteration / (time / 1000) for iteration, time in zip(iterations, timings)]
                
        kmeans_ref = kmeans = KMeans(n_clusters = N_CLUSTER, 
                        init = KMEANS_INIT, 
                        n_init = KMEANS_N_INIT, 
                        tol = TOL,
                        max_iter = MAX_ITER,
                        random_state = SEED)

        kmeans_ref.fit(data)

        true_labels = kmeans_ref.labels_
        my_labels = kmeans.labels_

        # Align labels_B to labels_A
        aligned_labels_B = match_labels(true_labels, my_labels)

        # Step 1: Compare the arrays element-wise
        matches = true_labels == aligned_labels_B  # This will create a boolean array

        # Step 2: Count the number of True values (i.e., where elements are equal)
        count_equal = np.sum(matches)  # Sum the boolean array to get the count of matches
        
        log_file = output_dir / (cmd_args.output_file + "_alignment_log.txt")
        with open(log_file, "a") as log:
            log.write(f"Chunk size: {chunk_size} ")
            log.write(f"Number of elements: {aligned_labels_B.size} ")
            log.write(f"Number of equal elements: {count_equal}\n")

                
        with open(output_file, "r") as file:
            
            content = file.read()
            
            if not content:
                file_is_empty = True
            else:
                file_is_empty = False
                
        print("Writing timings to file")
            
        with open(output_file, "a") as file:
            
            if file_is_empty:
                file.write("OMP_NUM_THREADS\tFIT_TIME\tNUM_ITERATIONS\tITERATIONS_SECOND\tCHUNK_SIZE\n")
                for fit_time, num_iterations, iteration_second in zip(timings, iterations, iterations_second):
                    file.write(f"{omp_num_threads}\t{fit_time}\t{num_iterations}\t{iteration_second}\t{chunk_size}\n")
            else:
                for fit_time, num_iterations, iteration_second in zip(timings, iterations, iterations_second):
                    file.write(f"{omp_num_threads}\t{fit_time}\t{num_iterations}\t{iteration_second}\t{chunk_size}\n")
                
        print("Finished writing to file")
    
if __name__ == "__main__":
    main()