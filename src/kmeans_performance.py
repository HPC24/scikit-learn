import logging
import argparse
import time
import os

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


N_DIGITS = 10 
TOLERANCE = 1e-24
DATA_DIRECTORY = "/home/joshua/Projects/HPC_Project/data"



def get_cmdargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_images",
        default=70_000,
        type=int,
        help="Number of images to use.",
    )
    parser.add_argument(
        "--max_iter",
        default=200,
        type=int,
        help="Number of k-Means interations",
    )
    parser.add_argument(
        "--num_reps",
        type=int,
        default=5,
        help="Number of repetitions of k-Means",
    )
    parser.add_argument(
        "--output_path",
        default="./out",
        help="Path where to place output files",
    )
    return parser.parse_args()


def main():
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Gettting command line arguments")
    cmd_args = get_cmdargs()
    
    logging.ingo("Loading Mnist Dataset")
    
    mnist = fetch_openml(
    name="mnist_784",
    cache=True,
    data_home=DATA_DIRECTORY,
    as_frame=False,
    )
    
    
    logging.info("Getting data and target and splitting into desired size")
    
    # normalize the data 
    data = mnist.data[:cmd_args.num_images] / 255
    target = mnist.target[:cmd_args.num_images]
    
    # How many Open mp threads were used 
    
    
    runtimes = []
    openmp_threads = os.environ["OMP_NUM_THREADS"]
    logging.info(f"Starting KMeans run with {cmd_args.num_reps} Repitions and {openmp_threads} OMP Threads")
    
    for i in range(cmd_args.num_reps):
        start_time = time.perf_counter()
        kmeans = KMeans(
                    n_clusters = N_DIGITS,
                    init = 'k-means++',
                    max_iter = cmd_args.max_iter,
                    verbose = 1,
                    random_state = 42,
                    tol = TOLERANCE
            ).fit(data)
        end_time = time.perf_counter()
        runtimes.append(end_time - start_time)
        logging.info(f"Time for Iteration Nr. {i + 1} = {runtimes[-1]:.4f} seconds")
        
    output_path = Path(cmd_args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame()

    




