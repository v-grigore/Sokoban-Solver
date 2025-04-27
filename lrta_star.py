from sokoban.map import Map
from sokoban.moves import *
from search_methods.solver import Solver
from typing import Dict, List, Tuple, Set, Optional

# Define direction mappings for dx and dy
dx = {0: -1, 1: 1, 2: 0, 3: 0}  # LEFT, RIGHT, UP, DOWN
dy = {0: 0, 1: 0, 2: -1, 3: 1}  # LEFT, RIGHT, UP, DOWN
import time
import random
import heuristics

class LRTAStar(Solver):
    def __init__(self, map_state: Map, heuristic_func=None, max_iterations=1000, time_limit=30) -> None:
        super().__init__(map_state)
        self.heuristic_func = heuristic_func if heuristic_func else heuristics.combined_heuristic
        self.max_iterations = max_iterations
        self.time_limit = time_limit  # time limit in seconds
        self.h_table = {}  # learned heuristic values
        self.deadlock_states = set()  # set of deadlock states to avoid
    
    def get_valid_moves(self, state: Map) -> List[int]:
        """Get valid moves from the current state, filtering out pull moves"""
        moves = state.filter_possible_moves()
        
        # Only allow basic moves and push moves
        push_moves = []
        for move in moves:
            if move < BOX_LEFT:  # Basic player movement
                push_moves.append(move)
            else:  # Box move
                # Check if this is a push move (player is behind the box)
                implicit_move = move - 4
                player_pos = (state.player.x, state.player.y)
                future_pos = state.player.get_future_position(implicit_move)
                
                # Check if there's a box at the future position (push)
                if future_pos in state.positions_of_boxes:
                    push_moves.append(move)
        
        return push_moves
    
    def evaluate(self, state: Map) -> float:
        """Get the heuristic value for a state, using the learned value if available"""
        state_str = str(state)
        
        # If we already know this is a deadlock state, return infinite cost
        if state_str in self.deadlock_states:
            return float('inf')
            
        if state_str not in self.h_table:
            h_value = self.heuristic_func(state)
            
            # Check for deadlock
            if h_value == float('inf'):
                self.deadlock_states.add(state_str)
                
            self.h_table[state_str] = h_value
            
        return self.h_table[state_str]
    
    def score_move(self, state: Map, move: int) -> float:
        """Score a move based on how it positions the player to push boxes toward goals"""
        next_state = state.copy()
        next_state.apply_move(move)
        
        # Base score is the heuristic value
        score = self.evaluate(next_state)
        if score == float('inf'):
            return float('inf')
        
        # Player position after move
        player_pos = (next_state.player.x, next_state.player.y)
        
        # Get boxes not on targets
        boxes_not_on_targets = [(box.x, box.y) for box in next_state.boxes.values() 
                               if (box.x, box.y) not in next_state.targets]
        
        # Bonus for box push moves that make progress toward targets
        if move >= BOX_LEFT:  # Box push move
            box_move = move - 4  # Convert to box movement direction
            box_x, box_y = None, None
            
            # Find which box was moved
            for box in next_state.boxes.values():
                if (box.x, box.y) not in state.positions_of_boxes:
                    box_x, box_y = box.x, box.y
                    break
            
            if box_x is not None and box_y is not None:
                # Check if the box is closer to any target after the move
                box_pos = (box_x, box_y)
                min_dist_before = min(heuristics.manhattan_distance((box_x - dx[box_move], box_y - dy[box_move]), target) 
                                    for target in state.targets)
                min_dist_after = min(heuristics.manhattan_distance(box_pos, target) 
                                    for target in state.targets)
                
                if min_dist_after < min_dist_before:
                    # Reward moves that bring boxes closer to targets
                    score -= 5  # Significantly reward progress toward targets
                
                # Extra reward if the box ends up on a target
                if box_pos in state.targets:
                    score -= 10
        
        # For all moves, evaluate how they position the player relative to boxes
        else:
            # If there are boxes not on targets, prioritize getting close to them
            if boxes_not_on_targets:
                # Specifically reward moves that bring player closer to boxes not on targets
                current_nearest = min(heuristics.manhattan_distance((state.player.x, state.player.y), box) 
                                     for box in boxes_not_on_targets)
                after_nearest = min(heuristics.manhattan_distance(player_pos, box) 
                                    for box in boxes_not_on_targets)
                
                if after_nearest < current_nearest:
                    score -= 3  # Reward getting closer to boxes not on targets
                
                # Further reward strategic positioning
                for box_pos in boxes_not_on_targets:
                    # Check if player is adjacent to a box after this move
                    if (abs(player_pos[0] - box_pos[0]) == 1 and player_pos[1] == box_pos[1]) or \
                       (abs(player_pos[1] - box_pos[1]) == 1 and player_pos[0] == box_pos[0]):
                        
                        # Find where the box would go if pushed
                        push_dir_x = box_pos[0] - player_pos[0]
                        push_dir_y = box_pos[1] - player_pos[1]
                        pushed_pos = (box_pos[0] + push_dir_x, box_pos[1] + push_dir_y)
                        
                        # Check if push is valid (not into wall or another box)
                        if pushed_pos not in state.obstacles and pushed_pos not in state.positions_of_boxes:
                            # Check if push brings box closer to target
                            min_dist_to_target = min(heuristics.manhattan_distance(box_pos, target) 
                                                   for target in state.targets)
                            min_dist_after_push = min(heuristics.manhattan_distance(pushed_pos, target) 
                                                    for target in state.targets)
                            
                            if min_dist_after_push < min_dist_to_target:
                                score -= 8  # Significant reward for strategic positioning
            
        return score
    
    def find_best_move(self, state: Map, valid_moves: List[int]) -> int:
        """Find the best move based on heuristic and strategic positioning"""
        best_score = float('inf')
        best_moves = []
        
        for move in valid_moves:
            score = self.score_move(state, move)
            
            if score < best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        
        # If multiple moves have the same score, choose one randomly
        return random.choice(best_moves) if best_moves else None
    
    def restart_from_current(self, current_state: Map, solution_moves: List[int]) -> Tuple[Map, List[int]]:
        """Restart from the initial state but keep learned heuristics"""
        # Restart from the initial state
        new_state = self.map.copy()
        return new_state, []  # Reset solution moves
    
    def solve(self) -> Optional[List[int]]:
        """Solve the Sokoban puzzle using optimized LRTA*"""
        start_time = time.time()
        current_state = self.map.copy()
        solution_moves = []
        
        # Set to track visited states to avoid infinite loops
        visited_states = set()
        cycle_count = 0
        max_cycle_count = 3
        
        iterations = 0
        while not current_state.is_solved() and iterations < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                print(f"Time limit of {self.time_limit} seconds reached.")
                break
            
            iterations += 1
            current_state_str = str(current_state)
            
            # Check for deadlock
            if self.evaluate(current_state) == float('inf'):
                print(f"Deadlock detected on iteration {iterations}. Restarting...")
                current_state, solution_moves = self.restart_from_current(current_state, solution_moves)
                visited_states.clear()
                cycle_count = 0
                continue
            
            # Check if we've been in this state before (cycle detection)
            if current_state_str in visited_states:
                cycle_count += 1
                
                if cycle_count > max_cycle_count:
                    print(f"Detected persistent cycle. Restarting with preserved heuristics...")
                    current_state, solution_moves = self.restart_from_current(current_state, solution_moves)
                    visited_states.clear()
                    cycle_count = 0
                    continue
                
                # Try a random move to break out of cycle
                valid_moves = self.get_valid_moves(current_state)
                if valid_moves:
                    random_move = random.choice(valid_moves)
                    current_state.apply_move(random_move)
                    solution_moves.append(random_move)
                else:
                    print("No valid moves available to break cycle.")
                    break
                continue
            
            visited_states.add(current_state_str)
            
            # Get valid moves
            valid_moves = self.get_valid_moves(current_state)
            if not valid_moves:
                print("No valid moves available.")
                break
            
            # Choose the best move based on strategic scoring
            best_move = self.find_best_move(current_state, valid_moves)
            
            if best_move is None:
                print("No best move found.")
                break
            
            # Apply the move
            next_state = current_state.copy()
            next_state.apply_move(best_move)
            next_state_str = str(next_state)
            
            # Update heuristic value
            if current_state_str in self.h_table:
                # Update heuristic value based on chosen move and next state
                h_next = self.evaluate(next_state)
                if h_next != float('inf'):
                    # Use a weighted learning rate to adjust heuristic faster
                    learning_rate = 1.2  # Slightly increase the learning rate
                    new_h = min(
                        learning_rate * (1 + h_next),  # cost to move + heuristic of successor
                        self.h_table[current_state_str] + 2  # Limit how much it can increase
                    )
                    self.h_table[current_state_str] = max(self.h_table[current_state_str], new_h)
            
            # Apply the best move
            current_state = next_state
            solution_moves.append(best_move)
            
            # Print progress
            if iterations % 10 == 0:
                print(f"Iteration {iterations}, moves so far: {len(solution_moves)}")
        
        if current_state.is_solved():
            print(f"Solution found in {iterations} iterations, {len(solution_moves)} moves.")
            return solution_moves
        else:
            print(f"No solution found after {iterations} iterations.")
            return None

# Create a simple function to run the solver
def solve_sokoban(map_path, heuristic_func=None, max_iterations=1000, time_limit=30):
    map_state = Map.from_yaml(map_path)
    solver = LRTAStar(map_state, heuristic_func, max_iterations, time_limit)
    solution = solver.solve()
    
    if solution:
        # Apply the solution moves to visualize
        result_state = map_state.copy()
        solution_states = [result_state.copy()]
        
        for move in solution:
            result_state.apply_move(move)
            solution_states.append(result_state.copy())
        
        # Print the solution steps
        print("Solution steps:")
        for i, move in enumerate(solution):
            print(f"Step {i+1}: {moves_meaning[move]}")
        
        return solution_states
    else:
        return None
