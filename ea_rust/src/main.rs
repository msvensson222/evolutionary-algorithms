use ea_rust::EvolutionaryAlgorithm;
use ea_rust::Objective;
use ea_rust::Population;

fn main() {
    /// Improvments possible:
    /// The evolutionary algorithm is currently single-threaded. And the code non-vectorized.

    const INDIVIDUAL_SIZE: usize = 50;
    const POPULATION_SIZE: usize = 500;
    const CXPB: f64 = 0.25;
    const MUTPB: f64 = 0.25;
    const TOURNAMENT_SIZE: usize = 3;

    let target_genes: Vec<f64> = (0..INDIVIDUAL_SIZE).map(|i| i as f64).collect();
    let objective = Objective {
        target_genes: target_genes,
    };

    let population = Population::new(POPULATION_SIZE, INDIVIDUAL_SIZE, objective.into());

    let mut evolutionary_algorithm = EvolutionaryAlgorithm::new(
        TOURNAMENT_SIZE, // TODO: Make named args explicit
        CXPB,
        MUTPB,
    );

    evolutionary_algorithm.evolve(population, 30.0, 1e-3);
}
