use ea_rust::EvolutionaryAlgorithm;
use ea_rust::Objective;

fn main() {
    /// Improvments possible:
    /// Over-usage of .clone() and .to_vec() in the code. Could be optimized with lifetimes and references.
    /// The objective function is currently hard-coded into the Objective struct. It would be better to make it a trait that can be implemented for different problems.
    /// The evolutionary algorithm is currently single-threaded. And the code non-vectorized.

    const INDIVIDUAL_SIZE: usize = 5;
    const POPULATION_SIZE: usize = 3;
    const CXPB: f64 = 0.25;
    const MUTPB: f64 = 0.25;
    const TOURNAMENT_SIZE: usize = 3;

    let target_genes: Vec<f64> = (0..INDIVIDUAL_SIZE).map(|i| i as f64).collect();
    let objective = Objective {
        target_genes: target_genes.clone(), // Could skip .clone() here, but then I need lifetime annotations
    };

    let mut evolutionary_algorithm = EvolutionaryAlgorithm::new(
        POPULATION_SIZE,
        INDIVIDUAL_SIZE,
        TOURNAMENT_SIZE,
        CXPB,
        MUTPB,
        objective
    );

    evolutionary_algorithm.evolve(60.0, 1e-3);
}
