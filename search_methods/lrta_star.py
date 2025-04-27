import os
import time
from typing import Optional

from matplotlib import pyplot as plt
from sokoban.map import Map
from sokoban.moves import *

def get_neighbours(map: Map):
    ''' Returns the neighbours of the current state (no pull moves)'''
    neighbours = []
    for neighbour in map.get_neighbours():
        if neighbour.undo_moves == 0:
            neighbours.append(neighbour)
    return neighbours

def off_target_heuristic(map_state: Map, prev_state: Map) -> int:
    """Return a penalty if player moved a box off a target (player is on target)"""
    player_pos = (map_state.player.x, map_state.player.y)
    
    # Check if player moved box off a target
    if player_pos in map_state.targets and player_pos in prev_state.positions_of_boxes:
        return 25  # Penalty for moving a box off a target
    
    return 0

def lrta_star(map: Map, heuristic, time_limit=30, max_iterations=1000000):
    ''' Solve sokoban using LRTA* algorithm '''
    current_state = map.copy()
    h = {}
    visit_count = {}
    states = [current_state.copy()]
    solution = [current_state.copy()]
    iterations = 0
    start_time = time.time()
    visited = set()
    reset_threshold = map.width * map.length * 3
    reset_count = 0
    prev_state = None
    last_box_move = 0
    ignored_positions = []

    def get_heuristic(state: Map) -> int:
        ''' Heuristic function for LRTA* '''
        state_str = str(state)
        h_cost = h[state_str] if state_str in h else 1 + heuristic(state)
        penalty = visit_count[state_str] if state_str in visit_count else 0
        return h_cost + penalty * 3
    
    def cut_similar_states(state: Map):
        '''Set value of similar states to infinity'''
        for x in range(state.length):
            for y in range(state.width):
                state.player.x = x
                state.player.y = y
                state_str = str(state)
                h[state_str] = float('inf')

    while not current_state.is_solved():
        elapesd_time = time.time() - start_time
        if elapesd_time > time_limit:
            print("Time limit exceeded")
            break

        iterations += 1
        if iterations > max_iterations:
            print("Max iterations exceeded")
            break
        if iterations % 10000 == 0:
            print(f"Iterations: {iterations}")
        if elapesd_time % 60 == 0:
            print('=' * 20)
            print(f"Elapsed time: {int(elapesd_time)} seconds")
            print('=' * 20)

        current_state_str = str(current_state)

        # if no progress is made, reset
        # if len(solution) >= reset_threshold and iterations >= last_box_move + reset_threshold:
        if len(solution) >= reset_threshold and heuristic(current_state) >= heuristic(solution[len(solution) - reset_threshold]):
            # print(f"No progress made in the last {reset_threshold} iterations, resetting")
            # print("Iterations: ", iterations)

            reset_count += 1
            if reset_count > 1000:
                print("Reset count exceeded")
                break

            # _create_figure(
            #     current_state,
            #     show=True,
            #     save_path=None,
            #     save_name=None,
            #     neighbours=neighbours,
            #     costs=costs,
            #     pause=True
            # )

            cut_similar_states(current_state)
            current_state = map.copy()
            solution = [current_state.copy()]
            states.append(current_state)
            last_box_move = iterations
            ignored_positions = []
            continue

        if current_state_str not in visited:
            visited.add(current_state_str)
            h[current_state_str] = heuristic(current_state)
            visit_count[current_state_str] = 1
        else:
            visit_count[current_state_str] += 1

        neighbours = get_neighbours(current_state)
        costs = [get_heuristic(neighbour) for neighbour in neighbours]
        min_cost = min(costs)

        h[current_state_str] = max(h[current_state_str], min_cost) + off_target_heuristic(current_state, prev_state)

        if h[current_state_str] == float('inf'):
            print("Deadlock detected")
            current_state = map.copy()
            solution = [current_state.copy()]
            states.append(current_state)
            break

        box_move = None
        for neighbour in neighbours:
            player_pos = (neighbour.player.x, neighbour.player.y)
            if player_pos in current_state.positions_of_boxes:
                box_move = neighbour
                break

        best_neighbour = neighbours[costs.index(min_cost)]
        if box_move:
            if box_move == best_neighbour:
                last_box_move = iterations
                ignored_positions = []
            if box_move != best_neighbour:
                if heuristic(box_move) == float('inf'):
                    ignored_positions.append((box_move.player.x, box_move.player.y))
                elif box_move in h and h[box_move] == float('inf'):
                    ignored_positions.append((box_move.player.x, box_move.player.y))

        # _create_figure(
        #     current_state,
        #     show=True,
        #     save_path=None,
        #     save_name=None,
        #     neighbours=neighbours,
        #     costs=costs
        # )

        prev_state = current_state.copy()
        current_state = best_neighbour.copy()
        solution.append(current_state)
        states.append(current_state)

    return solution, iterations, current_state.explored_states, current_state.is_solved(), states, time.time() - start_time

def _create_figure(
        state: Map, 
        show: bool = True, 
        save_path: Optional[str] = None, 
        save_name: Optional[str] = None,
        neighbours: Optional[list] = None,
        costs: Optional[list] = None,
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

        if neighbours is not None and costs is not None:
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
