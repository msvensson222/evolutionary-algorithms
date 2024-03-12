import numpy as np
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from Objective import Objective

if __name__ == "__main__":
    np.random.seed(1)
    INDIVIDUAL_SIZE = 5
    POPULATION_SIZE = 30
    CXPB = 0.25
    MUTPB = 0.25
    TOURNAMENT_SIZE = 3

    target_genes = np.array(range(INDIVIDUAL_SIZE))

    assert len(target_genes) == INDIVIDUAL_SIZE, "Length of target genes must match individual size"

    evolutionary_algorithm = EvolutionaryAlgorithm(
        population_size=POPULATION_SIZE,
        individual_size=INDIVIDUAL_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        cxpb=CXPB,
        mutpb=MUTPB,
        target_genes=target_genes,
        objective=Objective(target_genes)
    )
    evolutionary_algorithm.evolve(time_limit_seconds=60, epsilon=1e-3)
