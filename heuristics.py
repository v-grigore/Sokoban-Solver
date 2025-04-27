from sokoban.map import Map
from typing import List, Tuple, Dict

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def min_manhattan_distance(box_pos: Tuple[int, int], targets: List[Tuple[int, int]]) -> int:
    """Calculate minimum Manhattan distance from a box to any target"""
    return min(manhattan_distance(box_pos, target) for target in targets)

def sum_box_to_target_distances(map_state: Map) -> int:
    """Sum of minimum distances from each box to any target"""
    boxes = [(box.x, box.y) for box in map_state.boxes.values()]
    return sum(min_manhattan_distance(box, map_state.targets) for box in boxes)

def nearest_box_distance(map_state: Map) -> int:
    """Find distance to the nearest box that isn't on a target"""
    player_pos = (map_state.player.x, map_state.player.y)
    
    # Get boxes not on targets
    boxes_not_on_targets = [(box.x, box.y) for box in map_state.boxes.values() 
                           if (box.x, box.y) not in map_state.targets]
    
    # If all boxes are on targets, return 0
    if not boxes_not_on_targets:
        return 0
    
    # Return distance to closest box not on a target
    return min(manhattan_distance(player_pos, box) for box in boxes_not_on_targets)

def strategic_box_distance(map_state: Map) -> int:
    """Distance to the nearest box that needs strategic positioning"""
    player_pos = (map_state.player.x, map_state.player.y)
    boxes = [(box.x, box.y) for box in map_state.boxes.values()]
    targets = map_state.targets
    
    # Initialize with a large value
    best_score = float('inf')
    
    for box in boxes:
        # Skip boxes already on targets
        if box in targets:
            continue
        
        # Find the nearest target for this box
        nearest_target = min(targets, key=lambda t: manhattan_distance(box, t))
        
        # Calculate ideal push direction (simplistic approach)
        dx = 1 if nearest_target[0] > box[0] else (-1 if nearest_target[0] < box[0] else 0)
        dy = 1 if nearest_target[1] > box[1] else (-1 if nearest_target[1] < box[1] else 0)
        
        # Calculate strategic position(s) for player to push the box
        push_positions = []
        if dx != 0:
            push_positions.append((box[0] - dx, box[1]))
        if dy != 0:
            push_positions.append((box[0], box[1] - dy))
        
        # If we have push positions, consider the closest one
        if push_positions:
            closest_push_pos = min(push_positions, key=lambda p: manhattan_distance(player_pos, p))
            position_score = manhattan_distance(player_pos, closest_push_pos)
            
            # Consider both the proximity to the push position and how close the box is to target
            combined_score = position_score + 0.5 * manhattan_distance(box, nearest_target)
            
            if combined_score < best_score:
                best_score = combined_score
    
    return int(best_score) if best_score != float('inf') else 0

def sum_player_to_box_distances(map_state: Map) -> int:
    """Sum of distances from player to each box"""
    player_pos = (map_state.player.x, map_state.player.y)
    boxes = [(box.x, box.y) for box in map_state.boxes.values()]
    return sum(manhattan_distance(player_pos, box) for box in boxes)

def min_matching_cost(map_state: Map) -> int:
    """Minimum sum of distances when matching boxes to targets (greedy assignment)"""
    boxes = [(box.x, box.y) for box in map_state.boxes.values()]
    targets = map_state.targets.copy()
    
    total_distance = 0
    for box in boxes:
        if not targets:
            break
        # Find closest target
        closest_target_idx = min(range(len(targets)), 
                               key=lambda i: manhattan_distance(box, targets[i]))
        total_distance += manhattan_distance(box, targets[closest_target_idx])
        targets.pop(closest_target_idx)
    
    return total_distance

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
    boxes = [(box.x, box.y) for box in map_state.boxes.values()]
    for box in boxes:
        if is_corner_deadlock(map_state, box):
            return float('inf')
    return 0

def combined_heuristic(map_state: Map) -> int:
    """Combine multiple heuristics"""
    # Check for deadlocks first
    if deadlock_heuristic(map_state) == float('inf'):
        return float('inf')
    
    # Combine box-target matching with player strategic positioning
    h1 = min_matching_cost(map_state) 
    h2 = strategic_box_distance(map_state) * 2  # Weight the strategic positioning more heavily
    
    return h1 + h2
