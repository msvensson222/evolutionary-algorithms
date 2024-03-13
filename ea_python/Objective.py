import numpy as np

class Objective:
    def __init__(self, target_genes: np.ndarray):
        self.target_genes = target_genes

    def objective_function(self, genes: np.ndarray) -> float:
        diff = genes - self.target_genes
        #return diff @ diff.T
        return np.sum(diff ** 2)