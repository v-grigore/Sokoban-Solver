from itertools import permutations
from typing import List, Tuple
from sokoban.map import Map
from sokoban.moves import *
from heapq import heappop, heappush
from collections import defaultdict

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    '''Calculate the Manhattan distance between two positions'''
    '''This is a helper function for the heuristic functions'''
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def min_manhattan_distance(box_pos: Tuple[int, int], targets: List[Tuple[int, int]]) -> int:
    '''Calculate the minimum Manhattan distance from a box to any target'''
    return min(manhattan_distance(box_pos, target) for target in targets)

def min_A_star_distance(box_pos: Tuple[int, int], targets: List[Tuple[int, int]], map_state: Map) -> int:
    '''Calculate the minimum A* distance from a box to any target'''
    def out_of_bounds(x, y):
        return x < 0 or x >= map_state.length or y < 0 or y >= map_state.width

    def a_star(start: Tuple[int, int], goal: Tuple[int, int]) -> int:

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = manhattan_distance(start, goal)

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                return g_score[current]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if out_of_bounds(neighbor[0], neighbor[1]):
                    continue

                if neighbor in map_state.obstacles or neighbor in map_state.positions_of_boxes:
                    continue

                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return float('inf')  # Return infinity if no path is found

    return min(a_star(box_pos, target) for target in targets if target)

def sum_box_to_target_distances(map_state: Map) -> int:
    '''Calculate the sum of minimum distances from each box to matching target'''
    boxes = [(box.x, box.y) for box in map_state.boxes.values()]
    targets = [target for target in map_state.targets if target not in map_state.positions_of_boxes]

    if targets == []:
        return 0
    return sum(min_A_star_distance(box, map_state.targets, map_state) for box in boxes)

def sum_box_to_target_distances_v2(map_state: Map) -> int:
    '''Calculate the sum of minimum distances from each box to matching target'''
    targets = [target for target in map_state.targets]

    if targets == []:
        return 0
    
    min_distance = float('inf')
    
    for permutation in permutations(targets):
        sum_distance = 0
        if len(targets) == 1:
            permutation = targets
        for box, target in zip(map_state.positions_of_boxes, permutation):
            distance = min_A_star_distance(box, [target], map_state)
            sum_distance += distance

        min_distance = min(min_distance, sum_distance)  # Update the minimum distance

        # Calculate the sum of distances for this permutation

    return min_distance

def nearest_box_distance(map_state: Map, valid_positions) -> int:
    '''Find the distance to the nearest box that isn't on a target'''
    player_pos = (map_state.player.x, map_state.player.y)

    return min_A_star_distance(player_pos, valid_positions, map_state)
    
    # Return distance to closest box not on a target
    return min(manhattan_distance(player_pos, box) for box in boxes_not_on_targets)

def is_corner_deadlock(map_state: Map, box_pos: Tuple[int, int]) -> bool:
    """Check if a box is in a corner deadlock position"""
    x, y = box_pos
    
    # Check if box is in a corner
    left_blocked = (x, y-1) in map_state.obstacles or y == 0
    right_blocked = (x, y+1) in map_state.obstacles or y == map_state.width - 1
    up_blocked = (x+1, y) in map_state.obstacles or x == map_state.length - 1
    down_blocked = (x-1, y) in map_state.obstacles or x == 0
    
    if (left_blocked and up_blocked) or \
       (left_blocked and down_blocked) or \
       (right_blocked and up_blocked) or \
       (right_blocked and down_blocked):
        # Box is in a corner and not on a target
        if box_pos not in map_state.targets:
            return True
    
    return False

def deadlock_heuristic(map_state: Map) -> int:
    """Return infinity if any box is in a deadlock position, otherwise 0"""
    for box in map_state.positions_of_boxes:
        if box in map_state.targets:
            continue

        if is_corner_deadlock(map_state, box):
            return float('inf')
    
    return 0

def off_target_heuristic(map_state: Map, prev_state: Map) -> int:
    """Return a penalty if player moved a box off a target (player is on target)"""
    player_pos = (map_state.player.x, map_state.player.y)
    
    # Check if player moved box off a target
    if player_pos in map_state.targets and player_pos in prev_state.positions_of_boxes:
        return 15  # Penalty for moving a box off a target
    
    return 0

def H1(map_state: Map, ignored_positions=[]) -> int:
    """Heuristic function"""
    if map_state.is_solved():
        return 0

    def out_of_bounds(x, y):
        return x < 0 or x >= map_state.length or y < 0 or y >= map_state.width
    
    # Sum of distances from player to boxes not on targets
    dx = [-1, 1, 0, 0, 1, -1, 0, 0]
    dy = [0, 0, -1, 1, 0, 0, 1, -1]
    valid_positions = []
    for box in map_state.positions_of_boxes:
        for i in range(4):
            x = box[0] + dx[i]
            y = box[1] + dy[i]
            future_x = box[0] + dx[i+4]
            future_y = box[1] + dy[i+4]
            if (x, y) in ignored_positions:
                continue
            if out_of_bounds(x, y):
                continue
            if (x, y) in map_state.obstacles or (x, y) in map_state.positions_of_boxes:
                continue
            if out_of_bounds(future_x, future_y):
                continue
            if (future_x, future_y) in map_state.obstacles or (future_x, future_y) in map_state.positions_of_boxes:
                continue
            valid_positions.append((x, y))

    if valid_positions == []:
        return float('inf')

    player_to_box_distance = nearest_box_distance(map_state, valid_positions)
    
    # Sum of minimum distances from boxes to targets
    box_to_target_distance = sum_box_to_target_distances(map_state)
    
    # Deadlock check
    deadlock_penalty = deadlock_heuristic(map_state)
    
    return player_to_box_distance + box_to_target_distance * 10 + deadlock_penalty

def H2(map_state: Map, ignored_positions=[]) -> int:
    """Heuristic function"""
    if map_state.is_solved():
        return 0

    def out_of_bounds(x, y):
        return x < 0 or x >= map_state.length or y < 0 or y >= map_state.width
    
    # Sum of distances from player to boxes not on targets
    dx = [-1, 1, 0, 0, 1, -1, 0, 0]
    dy = [0, 0, -1, 1, 0, 0, 1, -1]
    valid_positions = []
    for box in map_state.positions_of_boxes:
        for i in range(4):
            x = box[0] + dx[i]
            y = box[1] + dy[i]
            future_x = box[0] + dx[i+4]
            future_y = box[1] + dy[i+4]
            if (x, y) in ignored_positions:
                continue
            if out_of_bounds(x, y):
                continue
            if (x, y) in map_state.obstacles or (x, y) in map_state.positions_of_boxes:
                continue
            if out_of_bounds(future_x, future_y):
                continue
            if (future_x, future_y) in map_state.obstacles or (future_x, future_y) in map_state.positions_of_boxes:
                continue
            valid_positions.append((x, y))

    if valid_positions == []:
        return float('inf')

    player_to_box_distance = nearest_box_distance(map_state, valid_positions)
    
    # Sum of minimum distances from boxes to targets
    box_to_target_distance = sum_box_to_target_distances_v2(map_state)
    
    # Deadlock check
    deadlock_penalty = deadlock_heuristic(map_state)
    
    return player_to_box_distance + box_to_target_distance * 10 + deadlock_penalty