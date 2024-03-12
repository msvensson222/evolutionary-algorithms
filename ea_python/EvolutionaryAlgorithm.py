import time
import numpy as np
from Individual import Individual
from Objective import Objective

class EvolutionaryAlgorithm:
    def __init__(
            self,
            population_size: int,
            individual_size: int,
            tournament_size: int,
            cxpb: float,
            mutpb: float,
            target_genes: np.ndarray,
            objective: Objective
        ):
        self.population_size = population_size
        self.individual_size = individual_size
        self.tournament_size = tournament_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.target_genes = target_genes
        self.objective = objective
        self.population = self.generate_population()

    def generate_population(self) -> list[Individual]:
        return [Individual(np.random.uniform(low=0, high=self.individual_size, size=self.individual_size), self.objective) for _ in range(self.population_size)]

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        crossover_point = np.random.randint(1, len(parent1.genes))
        child1_genes = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child2_genes = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])
        return Individual(genes=child1_genes, objective=self.objective), Individual(genes=child2_genes, objective=self.objective)

    def mutate(self, individual: Individual):
        mutation_point = np.random.randint(len(individual.genes))
        individual.genes[mutation_point] += np.random.uniform(-1.0, 1.0)
        individual.recalculate_fitness()

    def generate_offspring(self) -> list[Individual]:
        offspring = []
        for i in range(0, len(self.population), 2):
            if i + 1 < len(self.population) and np.random.rand() < self.cxpb:
                child1, child2 = self.crossover(self.population[i], self.population[i+1])
                offspring.extend([child1, child2])
            else:
                offspring.append(self.population[i])
                if i + 1 < len(self.population):
                    offspring.append(self.population[i+1])

        for individual in offspring:
            if np.random.rand() < self.mutpb:
                self.mutate(individual)

        return offspring

    def select_survivors(self, offspring: list[Individual], tournsize: int) -> list[Individual]:
        combined = self.population + offspring
        selected: list[Individual] = []
        while len(selected) < self.population_size:
            tournament = np.random.choice(combined, size=tournsize, replace=False)
            best = min(tournament, key=lambda ind: ind.fitness)
            selected.append(best)
        return selected[:self.population_size]

    def evolve(self, time_limit_seconds: float, epsilon: float):
        start_time = time.time()
        generation_count = 0

        best_individual = self.population[0] # Take the first individual as the best for now
        best_individual_fitness = best_individual.fitness

        while True:
            offspring = self.generate_offspring()
            self.population = self.select_survivors(offspring, self.tournament_size)

            for individual in self.population:
                if individual.fitness < best_individual_fitness:
                    best_individual = individual
                    best_individual_fitness = individual.fitness

            generation_count += 1

            if best_individual_fitness < epsilon or (time.time() - start_time) > time_limit_seconds:
                break

            if generation_count % 100 == 0:
                print(f"Generation {generation_count}: Best Individual Fitness: {best_individual_fitness}")

        end_time = time.time()
        print(f"Total Time: {end_time - start_time:.4f} seconds")
        print(f"Avg. time (ms) per generation: {(end_time - start_time) / generation_count * 1000 :.3f} miliseconds")
        print(f"Generations: {generation_count}")
        print(f"Best Individual Fitness: {best_individual}")
        print(f"Target Individual: {self.target_genes}")
        print(f"Best Individual: {best_individual.genes}")
