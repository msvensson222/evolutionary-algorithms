import numpy as np
from Objective import Objective

class Individual:
    def __init__(self, genes: np.ndarray, objective: Objective):
        self.genes: np.ndarray = genes
        self.objective = objective
        self.fitness: float = self.calculate_fitness()

    def calculate_fitness(self) -> float:
        return self.objective.objective_function(self.genes)

    def recalculate_fitness(self):
        self.fitness = self.calculate_fitness()