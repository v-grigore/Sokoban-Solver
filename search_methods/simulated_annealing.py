import os
import random
import time
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from sokoban.map import Map
from sokoban.moves import *
from queue import Queue
from numpy.random import default_rng
from .heuristics import min_A_star_distance

def out_of_bounds(x, y, map: Map) -> bool:
    ''' Check if the position is out of bounds '''
    return x < 0 or x >= map.length or y < 0 or y >= map.width

def get_neighbours(map: Map):
    ''' Returns the neighbours of the current state (no pull moves)'''
    neighbours = []
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    for neighbour in map.get_neighbours():
        if neighbour.undo_moves == map.undo_moves:
            neighbours.append(neighbour)
        if neighbour.undo_moves > 0:
            player_pos = (map.player.x, map.player.y)
            for box in map.boxes.values():
                for i in range(4):
                    if (box.x + dx[i], box.y + dy[i]) == player_pos:
                        opposite_side = (box.x - dx[i], box.y - dy[i])
                        if opposite_side in map.obstacles:
                            neighbours.append(neighbour)
                            break
                        if out_of_bounds(opposite_side[0], opposite_side[1], map):
                            neighbours.append(neighbour)
                            break
    return neighbours

def off_target_heuristic(map_state: Map, prev_state: Map) -> int:
    """Return a penalty if player moved a box off a target (player is on target)"""
    player_pos = (map_state.player.x, map_state.player.y)
    
    # Check if player moved box off a target
    if player_pos in map_state.targets and player_pos in prev_state.positions_of_boxes:
        return 25  # Penalty for moving a box off a target
    
    return 0

def is_box_move(prev: Map, current: Map):
    ''' Returns true if the player moved a box'''
    if (current.player.x, current.player.y) in prev.positions_of_boxes:
        return True
    return False

def get_box_moves(map: Map):
    ''' Returns the possible moves the player can make with the boxes'''
    box_moves = []
    solution = []
    solutions = []
    visited_states = set()
    queue = Queue()
    queue.put((map.copy(), solution))
    while not queue.empty():
        current_state, current_solution = queue.get()
        current_state_str = str(current_state)

        if current_state_str in visited_states:
            continue

        if current_state in box_moves:
            continue
        visited_states.add(current_state_str)

        for neighbour in current_state.get_neighbours():
            if is_box_move(current_state, neighbour) or neighbour.undo_moves > current_state.undo_moves:
                if str(neighbour) not in visited_states:
                    box_moves.append(neighbour)
                    solutions.append(current_solution + [neighbour])
                    visited_states.add(str(neighbour))
                continue
            queue.put((neighbour, current_solution + [neighbour]))

    return box_moves, solutions


def simulated_annealing(map: Map, heuristic, time_limit=300):
    """Simulated Annealing algorithm for Sokoban"""
    start_time = time.time()
    current_state = map.copy()
    best_state = current_state.copy()
    best_cost = float('inf')
    solution = []
    states = []
    states.append(current_state.copy())
    solution.append(current_state.copy())
    reset_threshold = map.width * map.length * 3
    iterations = 0
    reset_count = 0
    
    # Initial temperature and cooling rate
    temperature = 1000.0
    cooling_rate = 0.99

    def cost(state: Map, neighbour: Map) -> int:
        return heuristic(neighbour) + off_target_heuristic(neighbour, state)
    
    while not current_state.is_solved():
        if time.time() - start_time > time_limit:
            print("Time limit exceeded")
            break
        
        iterations += 1

        neighbours = get_neighbours(current_state)
        costs = [cost(current_state, neighbour) + 1 for neighbour in neighbours]
        costs = np.array(costs)
        probabilities = np.exp(-costs) / np.sum(np.exp(-costs))
        
        chosen_index = np.random.choice(len(neighbours), p=probabilities)
        next_state = neighbours[chosen_index]

        delta = cost(current_state, current_state) - costs[chosen_index]
        if delta < 0 or random.uniform(0, 1) < np.exp(-delta / temperature):
            pass
        else:
            next_state = current_state.copy()

        if len(solution) >= reset_threshold and heuristic(current_state) >= heuristic(solution[len(solution) - reset_threshold]):
            if random.uniform(0, temperature) > temperature:
                print(f"No progress made in the last {reset_threshold} iterations, resetting")
                print("Iterations: ", iterations)

                reset_count += 1
                if reset_count > 1000:
                    print("Reset count exceeded")
                    break

                current_state = map.copy()
                solution = [current_state.copy()]
                states.append(current_state)
                continue

        _create_figure(
            current_state,
            show=True,
            save_path=None,
            save_name=None,
            neighbours=neighbours,
            costs=costs,
            probabilities=probabilities,
            pause=True
        )
        
        current_state = next_state.copy()
        states.append(current_state.copy())
        solution.append(current_state.copy())
        temperature *= cooling_rate
    
    return solution, iterations, current_state.explored_states, current_state.is_solved(), states, time.time() - start_time

def simulated_annealing_v2(map: Map, heuristic, time_limit=300):
    """Simulated Annealing algorithm for Sokoban"""
    random.seed(45)
    rng = default_rng(45) 

    start_time = time.time()
    current_state = map.copy()
    best_state = current_state.copy()
    best_cost = float('inf')
    solution = []
    states = []
    states.append(current_state.copy())
    solution.append(current_state.copy())
    reset_threshold = map.width * map.length * 3
    iterations = 0
    it_reset = 0
    reset_count = 0
    
    # Initial temperature and cooling rate
    temperature = 1000.0
    cooling_rate = 0.99

    def get_cost(state: Map, prev: Map) -> int:
        return heuristic(state) + state.explored_states + (15 if state.undo_moves > prev.undo_moves else 0)
    
    while not current_state.is_solved():
        if time.time() - start_time > time_limit:
            print("Time limit exceeded")
            break
        
        iterations += 1
        it_reset += 1
        if iterations % 1000 == 0:
            print(f"Iterations: {iterations}")

        # neighbours = [neighbour for neighbour in current_state.get_neighbours() if cost(neighbour) != float('inf')]
        neighbours, solutions = get_box_moves(current_state)
        costs = [get_cost(neighbour, current_state) for neighbour in neighbours]

        costs = np.array(costs)
        sum = np.sum(np.exp(-costs))
        probabilities = np.exp(-costs) / sum if sum != 0 else np.ones(len(costs)) / len(costs)
        
        chosen_index = rng.choice(len(neighbours), p=probabilities)
        next_state = neighbours[chosen_index]
        next_solution = solutions[chosen_index]

        delta = costs[chosen_index] - get_cost(current_state, current_state)
        if delta < 0 or random.uniform(0, 1) < np.exp(-delta / temperature):
            pass
        else:
            next_state = current_state.copy()
            next_solution = [current_state.copy()]

        if it_reset >= reset_threshold and temperature < 0.1:
            print(f"No progress made in the last {reset_threshold} iterations, reheating")
            print("Iterations: ", iterations)

            reset_count += 1
            if reset_count > 1000:
                print("Reset count exceeded")
                break

            temperature = 1000.0
            it_reset = 0
            continue

        # if iterations > 1000:
        #     _create_figure(
        #         current_state,
        #         show=True,
        #         save_path=None,
        #         save_name=None,
        #         neighbours=neighbours,
        #         costs=costs,
        #         probabilities=probabilities,
        #         pause=True
        #     )

        player_pos = (current_state.player.x, current_state.player.y)
        target = (next_state.player.x, next_state.player.y)

        current_state = next_state.copy()
        solution = solution + next_solution
        states = states + next_solution
        temperature *= cooling_rate
    
    print("Undo moves: ", current_state.undo_moves)
    return solution, current_state.undo_moves, current_state.explored_states, current_state.is_solved(), states, time.time() - start_time


def _create_figure(
        state: Map, 
        show: bool = True, 
        save_path: Optional[str] = None, 
        save_name: Optional[str] = None,
        neighbours: Optional[list] = None,
        costs: Optional[list] = None,
        probabilities: Optional[list] = None,
        pause: bool = False
    ) -> None:
        fig, ax = plt.subplots()
        ax.imshow(state.map, cmap='viridis')

        marker_size = 10
        ax.invert_yaxis()

        width_labels = [x - 0.5 for x in range(state.width)]
        length_labels = [y - 0.5 for y in range(state.length)]

        ax.grid(True, which='major', color='black', linewidth=1.5)
        ax.set_xticks(width_labels)
        ax.set_yticks(length_labels)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        ax.plot(state.player.y, state.player.x, 'ro', markersize=1.5 * marker_size)

        if neighbours is not None and costs is not None and probabilities is not None:
            for map, cost, prob in zip(neighbours, costs, probabilities):
                nx, ny = map.player.x, map.player.y
                ax.text(
                    ny, nx,
                    f"{cost}",
                    color='green',
                    ha='center',
                    va='center',
                    fontsize=marker_size,
                    fontweight='bold'
                )
                ax.text(
                    ny, nx + 0.2,
                    f"{prob:.2f}",
                    color='blue',
                    ha='center',
                    va='center',
                    fontsize=marker_size,
                    fontweight='bold'
                )
            for map, cost in zip(neighbours, costs):
                nx, ny = map.player.x, map.player.y
                ax.text(
                    ny, nx,
                    f"{cost}",
                    color='green',
                    ha='center',
                    va='center',
                    fontsize=marker_size,
                    fontweight='bold'
                )

        for box in state.boxes.values():
            ax.plot(box.y, box.x, 'bs', markersize=marker_size)

        for target_x, target_y in state.targets:
            ax.plot(target_y, target_x, 'gx', markersize=marker_size)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            if save_name is None:
                save_name = 'default.png'
            if not save_name.endswith('.png'):
                save_name += '.png'
            fig.savefig(os.path.join(save_path, save_name))

        if show:
            if pause:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1)

        plt.close(fig)
        