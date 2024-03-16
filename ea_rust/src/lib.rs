use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

#[derive(Debug)]
pub struct Objective {
    pub target_genes: Vec<f64>,
}

impl Objective {
    pub fn objective_function(&self, genes: &[f64]) -> f64 {
        self.target_genes
            .iter()
            .zip(genes.iter())
            .map(|(target_gene, gene)| (gene - target_gene).powi(2))
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct Individual {
    pub genes: Vec<f64>,
    pub fitness: Option<f64>,
    pub objective: Arc<Objective>,
}

impl Individual {
    pub fn calculate_fitness(&self) -> f64 {
        self.objective.objective_function(&self.genes)
    }

    pub fn recalculate_fitness(&mut self) {
        self.fitness = Some(self.calculate_fitness());
    }
}

pub struct Population {
    pub individuals: Vec<Individual>,
    population_size: usize,
}

impl Population {
    pub fn new(population_size: usize, individual_size: usize, objective: Arc<Objective>) -> Self {
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);
        let individuals: Vec<Individual> = (0..population_size)
            .map(|_| {
                // Generate random genes for each individual
                let genes: Vec<f64> = (0..individual_size)
                    .map(|_| rng.gen_range(0.0..=1.0))
                    .collect();
                let fitness = objective.objective_function(&genes);
                Individual {
                    genes: genes,
                    fitness: Some(fitness),
                    objective: objective.clone(),
                }
            })
            .collect();

        Population {
            individuals,
            population_size,
        }
    }
    pub fn find_best_individual(&self) -> Option<Individual> {
        self.individuals
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .cloned() // Clone the individual to return an owned copy
    }
}

pub struct EvolutionaryAlgorithm {
    pub tournament_size: usize,
    pub cxpb: f64,
    pub mutpb: f64,
    rng: StdRng,
}

impl EvolutionaryAlgorithm {
    pub fn new(tournament_size: usize, cxpb: f64, mutpb: f64) -> Self {
        let seed = [42; 32];
        let rng = StdRng::from_seed(seed);

        EvolutionaryAlgorithm {
            tournament_size,
            cxpb,
            mutpb,
            rng,
        }
    }

    pub fn evolve(&mut self, mut population: Population, time_limit_seconds: f64, epsilon: f64) {
        let start_time = std::time::Instant::now();
        let mut generation_count = 0;

        let mut best_individual = population
            .find_best_individual()
            .expect("Population cannot be empty");

        while best_individual
            .fitness
            .expect("Fitness should be calculated at this point")
            > epsilon
            && start_time.elapsed().as_secs_f64() < time_limit_seconds
        {
            let mut offspring = self.generate_offspring(&population);
            offspring.iter_mut().for_each(|ind| {
                if ind.fitness.is_none() {
                    ind.recalculate_fitness();
                }
            });

            population = self.select_survivors(population, offspring, self.tournament_size);
            best_individual = population
                .find_best_individual()
                .expect("Population cannot be empty");

            generation_count += 1;
            if generation_count % 100 == 0 {
                println!(
                    "Generation {}: Best Individual Fitness: {}",
                    generation_count,
                    best_individual.fitness.unwrap()
                );
            }
        }

        let end_time = start_time.elapsed().as_secs_f64();
        println!("Total Time: {:.4} seconds", end_time);
        println!(
            "Avg. time (ms) per generation: {:.3} miliseconds",
            (end_time / (generation_count as f64)) * 1000.0
        );
        println!("Generations: {}", generation_count);
        println!(
            "Best Individual Fitness: {}",
            best_individual.fitness.unwrap()
        );
        println!("Best Individual: {:?}", best_individual.genes);
    }

    fn generate_offspring(&mut self, population: &Population) -> Vec<Individual> {
        let mut offspring: Vec<Individual> = Vec::with_capacity(population.population_size);
        let mut indices: Vec<usize> = (0..population.population_size).collect();

        // Randomize indices to ensure diversity in crossover pairing
        indices.shuffle(&mut self.rng);

        // Process the population in pairs
        for i in (0..indices.len()).step_by(2) {
            // Check if the next index exists and if crossover should occur.
            let should_crossover = i + 1 < indices.len() && self.rng.gen::<f64>() < self.cxpb;

            if should_crossover {
                let parent1 = &population.individuals[indices[i]];
                let parent2 = &population.individuals[indices[i + 1]];
                let (mut child1, mut child2) = self.crossover(parent1, parent2);

                if self.rng.gen::<f64>() < self.mutpb {
                    self.mutate(&mut child1);
                }
                if self.rng.gen::<f64>() < self.mutpb {
                    self.mutate(&mut child2);
                }

                offspring.push(child1);
                offspring.push(child2);
            } else {
                // For the last individual in an odd-sized population or if crossover didn't happen,
                // clone and add the current individual. If it's not the last one, also clone and add the next.
                offspring.push(population.individuals[indices[i]].clone());
                if i + 1 < indices.len() {
                    offspring.push(population.individuals[indices[i + 1]].clone());
                }
            }
        }
        offspring
    }

    fn crossover(
        &mut self,
        parent1: &Individual,
        parent2: &Individual,
    ) -> (Individual, Individual) {
        // Implements single-point crossover. It is assumed both parents' genes are of equal length
        let crossover_point = self.rng.gen_range(1..parent1.genes.len());
        (
            Individual {
                genes: parent1.genes[..crossover_point]
                    .iter()
                    .chain(&parent2.genes[crossover_point..])
                    .cloned()
                    .collect(),
                fitness: None,
                objective: Arc::clone(&parent1.objective),
            },
            Individual {
                genes: parent2.genes[..crossover_point]
                    .iter()
                    .chain(&parent1.genes[crossover_point..])
                    .cloned()
                    .collect(),
                fitness: None,
                objective: Arc::clone(&parent2.objective),
            },
        )
    }

    fn mutate(&mut self, individual: &mut Individual) {
        // Uniform, random, mutation
        let mutation_point = self.rng.gen_range(0..individual.genes.len());
        individual.genes[mutation_point] += self.rng.gen_range(-1.0..1.0);
    }

    fn select_survivors(
        &mut self,
        mut population: Population,
        offspring: Vec<Individual>,
        tournsize: usize,
    ) -> Population {
        let mut combined = population.individuals;
        combined.extend(offspring);
        let mut survivors: Vec<Individual> = Vec::new();

        while survivors.len() < population.population_size {
            // Directly choose tournsize random references from the combined vector
            let tournament: Vec<&Individual> = combined
                .as_slice()
                .choose_multiple(&mut self.rng, tournsize)
                .collect(); // Collects references without needing to clone

            if let Some(best) = tournament
                .iter()
                .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            {
                // Dereference (*best) and then clone the Individual to push it into survivors
                survivors.push((*best).clone());
            }
        }

        population.individuals = survivors;
        population
    }
}
