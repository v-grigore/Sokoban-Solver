import argparse
import random
from sokoban import (
    Box,
    Map,
    Player,
    gif
)
from sokoban.moves import *
from search_methods import Solver

import os
from concurrent.futures import ThreadPoolExecutor as tp

LRTA_STAR = 1
SIMULATED_ANNEALING = 2
SIMULATED_ANNEALING_V2 = 3

s1_data = []
s2_data = []

def create_graph(title, graph_type, sim_annealing=False, save_path=None):
    """
    Create a graph based on the collected data.
    
    Args:
        title (str): Base title for the graph
        graph_type (int): 
            1 for execution time
            2 for explored states
            3 for solved status
            4 for undo_moves (only valid if sim_annealing=True)
        sim_annealing (bool): 
            If True, use data from Simulated Annealing algorithm (s2_data)
            If False, use data from LRTA* algorithm (s1_data)
        save_path (str): Path to save the graph image, if None, the graph is displayed
    """
    import matplotlib.pyplot as plt
    
    # Special check for graph_type=4
    if graph_type == 4 and not sim_annealing:
        raise ValueError("For graph_type=4 (undo_moves), sim_annealing must be True")
    
    # Select the appropriate data source
    data = s2_data if sim_annealing else s1_data
    
    # Extract and filter data based on graph_type
    if graph_type == 1:  # Execution time
        filtered_data = data  # No filtering for execution time
        y_data = [item[1] for item in filtered_data]
        y_label = "Execution Time (seconds)"
        graph_title = f"{title} - Execution Time"
    elif graph_type == 2:  # Explored states
        filtered_data = [item for item in data if item[4]]  # Filter for solved maps
        y_data = [item[3] for item in filtered_data]
        y_label = "Explored States"
        graph_title = f"{title} - Explored States"
    elif graph_type == 3:  # Solved status
        filtered_data = data  # No filtering for solved status
        y_data = [1 if item[4] else 0 for item in filtered_data]
        y_label = "Solved (1=Yes, 0=No)"
        graph_title = f"{title} - Solved Status"
    elif graph_type == 4:  # Undo moves (iterations in s2)
        filtered_data = [item for item in data if item[4]]  # Filter for solved maps
        y_data = [item[2] for item in filtered_data]
        y_label = "Undo Moves"
        graph_title = f"{title} - Undo Moves"
    else:
        raise ValueError("Invalid graph type. Must be 1, 2, 3, or 4.")
    
    # Extract map names from the filtered data
    map_names = [item[0] for item in filtered_data]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(map_names, y_data)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}' if isinstance(height, float) else f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
                    
    plt.xlabel("Map Names")
    plt.ylabel(y_label)
    plt.title(graph_title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Either save or show the graph
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == '__main__':
    '''
    Set RUN_ALL to True to run all tests in the tests folder.
    Set RUN_ALL to False to run a specific test.
    '''
    RUN_ALL = False

    if not RUN_ALL:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Solve Sokoban puzzles using LRTA* or Simulated Annealing')
        parser.add_argument('algorithm', choices=['lrta*', 'simulated-annealing'], 
                            help='Algorithm to use for solving the puzzle')
        parser.add_argument('input_file', help='Input file containing the Sokoban map')
        parser.add_argument('--save-images', action='store_true', help='Save solution as images')
        
        args = parser.parse_args()
        
        random.seed(45)

        # Load map from input file
        map = Map.from_yaml(args.input_file)
        map_name = os.path.basename(args.input_file).split('.')[0]
        
        if args.algorithm == 'lrta*':
            print(f"Solving map: {map_name} with LRTA*")
            solution, iterations, explored_states, is_solved, states, elapsed_time = Solver(map).solve(LRTA_STAR)
            algorithm_dir = 'LRTA'
        else:  # simulated-annealing
            print(f"Solving map: {map_name} with Simulated Annealing")
            solution, iterations, explored_states, is_solved, states, elapsed_time = Solver(map).solve(SIMULATED_ANNEALING_V2)
            algorithm_dir = 'SIM_ANN'
        
        print(f"Found solution: {is_solved}")
        print(f"Number of iterations: {iterations}")
        print(f"Number of explored states: {explored_states}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        if args.save_images and is_solved:
            output_dir = f'images'
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving gif for map: {map_name}")
            gif.save_images(solution, f'{output_dir}/{map_name}')
            gif.create_gif(f'{output_dir}/{map_name}', f'{map_name}.gif', output_dir)

        exit(0)

    for file in os.listdir('tests'):
        map = Map.from_yaml(f"tests/{file}")
        map_name = file.split('.')[0]
        print(f"Solving map: {map_name} with LRTA*")

        solution, iterations, explored_states, is_solved, states, elapsed_time = Solver(map).solve(LRTA_STAR)
        print(f"Found solution: {is_solved}")
        print(f"Number of iterations: {iterations}")
        print(f"Number of explored states: {explored_states}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        s1_data.append((map_name, elapsed_time, iterations, explored_states, is_solved))

        save_gif = False
        if save_gif:
            print(f"Saving gif for map: {map_name}")
            gif.save_images(solution, f'images/LRTA/{map_name}')
            gif.create_gif(f'images/LRTA/{map_name}', f'{map_name}.gif', 'images/LRTA')

        print(f"Solving map: {map_name} with Simulated Annealing")

        solution, iterations, explored_states, is_solved, states, elapsed_time = Solver(map).solve(SIMULATED_ANNEALING_V2)
        print(f"Found solution: {is_solved}")
        print(f"Number of iterations: {iterations}")
        print(f"Number of explored states: {explored_states}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        s2_data.append((map_name, elapsed_time, iterations, explored_states, is_solved))

        save_gif = False
        if not is_solved:
            save_gif = False
        if save_gif:
            print(f"Saving gif for map: {map_name}")
            gif.save_images(solution, f'images/SIM_ANN/{map_name}')
            gif.create_gif(f'images/SIM_ANN/{map_name}', f'{map_name}.gif', 'images/SIM_ANN')

    for i in range(3):
        create_graph("LRTA*", i + 1, sim_annealing=False, save_path=f'images/graphs/lrta_{i + 1}.png')
        create_graph("SIMULATED_ANNEALING", i + 1, sim_annealing=True, save_path=f'images/graphs/sim_annealing_{i + 1}.png')

    create_graph("Simulated Annealing", 4, sim_annealing=True, save_path='images/graphs/sim_annealing_undo_moves.png')

