import logging
import argparse
import pathlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def get_cmdargs():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_directory",
        required=True,
        nargs="*", #multiple arguments can be provided
        help="Path/Paths to the direcories with the benchmarks"
        )
    
    parser.add_argument(
        "--output_directory",
        required=False,
        help="Path to the directory where the images should be saved"
    )

    return parser.parse_args()

def combine_results(data_directories: list) -> list[pd.DataFrame]:

    dfs = []
    
    for directory in data_directories:
        
        directory = pathlib.Path(directory)
        assert directory.is_dir(), f"{directory} does not exist"
        assert any(directory.iterdir()), f"{directory} is empty"
        
        for file in directory.glob("*timings.txt"):
            print(f"Processing Data File: {file}")
            df = pd.read_csv(file, sep = "\t").assign(parameters = lambda df_: [str(file.stem).replace("_timings", "")] * df_.shape[0])
            print(df.columns)
            dfs.append(df)
            
    return pd.concat(dfs, axis = "rows")
            
        
def main():
    
    cmd_args = get_cmdargs()
    
    df = combine_results(cmd_args.data_directory)
    
#     color_mapping = {
#     'g++_O3_arch_opt': '#FF0000',  # Red
#     'g++_O3_no_archopt': '#0000FF',  # Blue
#     'icx_O3_arch_opt': '#FFD700',  # Gold
#     'icx_O3_no_archopt': '#008000',  # Green
#     'g++_O3_arch_opt_SIMD_512': '#00FFFF',  # Cyan
#     'icx_O3_arch_opt_SIMD_512': '#FF00FF',  # Magenta
#     'g++_O3_arch_opt_SIMD_512_NUMA': '#FFA500',  # Orange
#     'icx_O3_arch_opt_SIMD_512_NUMA': '#800080',  # Purple
#     'pybind': '#008080',  # Dark Teal
#     'sklearn': '#808080'   # Grey
# }
    
    # Create the pivot table for Speed up compared to single core performance
    piv_table = df.pivot_table(
        index = "OMP_NUM_THREADS", 
        values = "FIT_TIME", 
        columns = ["parameters", "CHUNK_SIZE"], 
        aggfunc="mean"
    )

    # T_n=1 / T_n
    piv_table = piv_table.div(piv_table.iloc[0], axis = "columns").pow(-1)
    #ideal line
    
    y_values = np.arange(1, piv_table.index[-1] + 1, 1)
    
    cmap = plt.get_cmap("Set1")
    
    # create plots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        sharex=True, 
        layout="constrained", 
        figsize=(15, 5)
    )
    
    
    piv_table.plot(
        kind = "line", 
        marker = "o", 
        grid = True,
        alpha=0.7,
        linewidth=2,
        markersize=8,
        ax = ax1,
        colormap = "viridis"
        #color = [color_mapping[label] for label in piv_table.columns]
    )
    
    ax1.plot(
        y_values, 
        y_values, 
        color='black', 
        linestyle='--', 
        label='ideal', 
        zorder=1
    )
    
    ax1.set_xlabel("Number of OpenMP threads")
    ax1.set_ylabel("Speed up $S_n$ w.r.t using single core")
    ax1.minorticks_on()
    ax1.set_xticks(y_values)
    ax1.set_yticks(y_values)
    ax1.legend()
    
    # calculate Compute performance (iterations/second)
    df_iterations = df.assign(iterations_second = lambda df_: df_["NUM_ITERATIONS"] / (df_["FIT_TIME"] / 1000))
    
    piv_table_iterations = df_iterations.pivot_table(
        index = "OMP_NUM_THREADS", 
        values = "iterations_second", 
        columns = ["parameters", "CHUNK_SIZE"], 
        aggfunc = "mean"
    )
    
    piv_table_iterations.plot(
        kind = "line", 
        marker = "o", 
        grid = True,
        alpha=0.7,
        linewidth=2,
        markersize=8,
        ax = ax2,
        colormap = "viridis" 
        #color = [color_mapping[label] for label in piv_table_iterations.columns]
    )
    
    ax2.set_xlabel("Number of OpenMP threads")
    ax2.set_ylabel("Compute performance (iterations/second)")
    ax2.minorticks_on()
    ax2.set_xticks(y_values)
    ax2.legend()
    
    if cmd_args.output_directory is not None:
        output_directory = pathlib.Path(cmd_args.output_directory)
        if not output_directory.exists():
        # Create the directory (and any necessary parent directories)
            output_directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory {output_directory} created.")
        else:
            print(f"Directory {output_directory} already exists.")
            
        save_file = output_directory / "KMeans_Performance.png"
        fig.savefig(save_file, bbox_inches="tight")
    else:
        plt.show()
        

if __name__ == "__main__":
    main()

    
    
    
    