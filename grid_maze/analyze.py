import pandas as pd
import matplotlib.pyplot as plt
import time
from environment import Environment
from agent import Agent

def statistics_with_obstacle_increase(heuristics, algorithms, plot=True):
    """
    Analyze the performance of different heuristic and algorithm combinations 
    as the number of obstacles increases in the environment.
    
    Args:
        heuristics (list): List of heuristic functions.
        algorithms (list): List of algorithm names (as strings).
        plot (bool): Whether to plot the results. Default is True.
        
    Returns:
        pd.DataFrame: A DataFrame containing the performance statistics.
    """
    MAX_OBSTACLES = 100
    ITERATIONS = 100

    # Initialize the DataFrame to store results
    df_obstacle_increase = pd.DataFrame(index=[i for i in range(MAX_OBSTACLES + 1)])

    for n_obstacles in range(MAX_OBSTACLES + 1):
        # Dictionary to accumulate values for each (heuristic, algorithm) pair
        results = {
            (heuristic, algorithm): {"time": 0, "iterations": 0, "path_length": 0}
            for heuristic in heuristics
            for algorithm in algorithms
        }
        completed_iterations = 0
        
        for _ in range(ITERATIONS):
            # Initialize the environment
            env = Environment(10, 10, n_obstacles)

            # Check if a basic configuration allows a valid path
            agent = Agent(heuristics[0], env)
            path, _ = agent.search(algorithms[0])
            if not path:
                # Skip iteration if no path is found
                continue

            completed_iterations += 1

            # Evaluate all combinations of heuristics and algorithms
            for heuristic, algorithm in results.keys():
                agent = Agent(heuristic, env)
                start = time.time()
                path, n_iterations = agent.search(algorithm)
                end = time.time()

                # Calculate metrics
                time_elapsed = end - start
                path_length = len(path) if path else 0

                # Accumulate values
                results[(heuristic, algorithm)]["time"] += time_elapsed
                results[(heuristic, algorithm)]["iterations"] += n_iterations
                results[(heuristic, algorithm)]["path_length"] += path_length

        if completed_iterations == 0:
            # Break the loop if no valid iterations were completed
            break

        # Compute averages and update the DataFrame
        for (heuristic, algorithm), metrics in results.items():
            heuristic_name = heuristic.__name__
            df_obstacle_increase.loc[n_obstacles, f"{algorithm}_{heuristic_name}_time"] = metrics["time"] / completed_iterations
            df_obstacle_increase.loc[n_obstacles, f"{algorithm}_{heuristic_name}_iterations"] = metrics["iterations"] / completed_iterations
            df_obstacle_increase.loc[n_obstacles, f"{algorithm}_{heuristic_name}_path_length"] = metrics["path_length"] / completed_iterations

    if plot:
        # Plot the results
        for metric in ["time", "iterations", "path_length"]:
            plt.figure(figsize=(8, 4))
            df_obstacle_increase[[col for col in df_obstacle_increase.columns if metric in col]].plot()
            plt.title(f"Obstacle Increase Analysis ({metric})")
            plt.xlabel("Number of Obstacles")
            plt.ylabel(metric.capitalize())
            plt.legend([col.replace(f"_{metric}", "") for col in df_obstacle_increase.columns if metric in col])
            plt.show()
            
    return df_obstacle_increase
