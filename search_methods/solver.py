from sokoban.map import Map
from search_methods.lrta_star import lrta_star
from search_methods.simulated_annealing import simulated_annealing
from search_methods.simulated_annealing import simulated_annealing_v2
from search_methods import heuristics

class Solver:

    def __init__(self, map: Map) -> None:
        self.map = map

    def solve(self, algorithm=1, time_limit=300):
        if algorithm == 1:
            return lrta_star(self.map, heuristics.H1, time_limit)
        elif algorithm == 2:
            return simulated_annealing(self.map, heuristics.H1, time_limit)
        else:
            return simulated_annealing_v2(self.map, heuristics.H2, time_limit)
