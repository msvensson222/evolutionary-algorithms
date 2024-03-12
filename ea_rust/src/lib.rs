use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)] // Deriving Debug for printing and Clone for cloning
pub struct Objective {
    pub target_genes: Vec<f64>
}

impl Objective {
    pub fn objective_function(&self, genes: &Vec<f64>) -> f64 {
        // TODO: Vectorize this
        let mut sum: f64 = 0.0;
        for i in 0..self.target_genes.len() {
            sum += (genes[i] - self.target_genes[i]).powi(2);
        }
        sum
    }
}

#[derive(Debug, Clone)] // Deriving Debug for printing and Clone for cloning
pub struct Individual {
    pub genes: Vec<f64>,
    pub fitness: f64,
    pub objective: Objective
}

impl Individual {
    pub fn new(genes: Vec<f64>, objective: Objective) -> Self {
        let fitness = objective.objective_function(&genes);
        Individual { genes, fitness, objective }
    }

    pub fn calculate_fitness(&self) -> f64 {
        self.objective.objective_function(&self.genes)
    }

    pub fn recalculate_fitness(&mut self) {
        self.fitness = self.calculate_fitness();
    }
    
}

pub struct EvolutionaryAlgorithm {
    pub population_size: usize,
    pub individual_size: usize,
    pub tournament_size: usize,
    pub cxpb: f64,
    pub mutpb: f64,
    pub objective: Objective,
    pub population: Vec<Individual>,
    rng: StdRng
}

impl EvolutionaryAlgorithm {
    pub fn new(
        population_size: usize,
        individual_size: usize,
        tournament_size: usize,
        cxpb: f64,
        mutpb: f64,
        objective: Objective
    ) -> Self {
        //let mut rng = rand::thread_rng(); // Create a random number generator
        let seed = [42; 32]; // A seed for demonstration purposes. Use a more random seed in practice.
        let mut rng = StdRng::from_seed(seed);
        let population = (0..population_size).map(|_| Individual::new(
            (0..individual_size).map(|_| rng.gen_range(0.0..(individual_size as f64))).collect(),
            objective.clone()
        )).collect();

        EvolutionaryAlgorithm {
            population_size,
            individual_size,
            tournament_size,
            cxpb,
            mutpb,
            objective,
            population,
            rng
        }
    }

    pub fn evolve(&mut self, time_limit_seconds: f64, epsilon: f64) {
        let start_time = std::time::Instant::now();
        let mut generation_count = 0;

        let mut best_individual: Individual = self.population[0].clone();
        let mut best_individual_fitness = best_individual.fitness;

        while best_individual_fitness > epsilon && start_time.elapsed().as_secs_f64() < time_limit_seconds {
            let offspring = self.generate_offspring();
            self.population = self.select_survivors(offspring, self.tournament_size);

            for individual in &self.population {
                if individual.fitness < best_individual_fitness {
                    best_individual = individual.clone();
                    best_individual_fitness = individual.fitness;
                }
            }

            generation_count += 1;

            if generation_count % 100 == 0 {
                println!("Generation {}: Best Individual Fitness: {}", generation_count, best_individual_fitness);
            }
        }

        let end_time = start_time.elapsed().as_secs_f64();
        println!("Total Time: {:.4} seconds", end_time);
        println!("Avg. time (ms) per generation: {:.3} miliseconds", (end_time / (generation_count as f64)) * 1000.0);
        println!("Generations: {}", generation_count);
        println!("Best Individual Fitness: {}", best_individual_fitness);
        println!("Target Individual: {:?}", self.objective.target_genes);
        println!("Best Individual: {:?}", best_individual.genes);
    }

    fn generate_offspring(&mut self) -> Vec<Individual> {
        let mut offspring: Vec<Individual> = Vec::new();
        //let mut rng = rand::thread_rng();
        for i in (0..self.population_size).step_by(2) {
            if i + 1 < self.population_size && self.rng.gen::<f64>() < self.cxpb {
                let (child1, child2) = self.crossover(self.population[i].clone(), self.population[i+1].clone());
                offspring.push(child1);
                offspring.push(child2);
            } else {
                offspring.push(self.population[i].clone());
                if i + 1 < self.population_size {
                    offspring.push(self.population[i+1].clone());
                }
            }
        }

        for individual in &mut offspring {
            if self.rng.gen::<f64>() < self.mutpb {
                self.mutate(individual);
            }
        }

        offspring
    }

    fn crossover(&mut self, parent1: Individual, parent2: Individual) -> (Individual, Individual) {
        // Single point crossover
        //let mut rng = rand::thread_rng();
        let crossover_point = self.rng.gen_range(1..self.individual_size);
        let child1_genes: Vec<f64> = parent1.genes[..crossover_point].iter().cloned()
            .chain(parent2.genes[crossover_point..].iter().cloned())
            .collect();
        let child2_genes: Vec<f64> = parent2.genes[..crossover_point].iter().cloned()
            .chain(parent1.genes[crossover_point..].iter().cloned())
            .collect();
        (Individual::new(child1_genes, self.objective.clone()), Individual::new(child2_genes, self.objective.clone()))
    }

    fn mutate(&mut self, individual: &mut Individual) {
        // Uniform, random, mutation
        // let mut rng = rand::thread_rng();
        let mutation_point = self.rng.gen_range(0..self.individual_size);
        individual.genes[mutation_point] += self.rng.gen_range(-1.0..1.0);
        individual.recalculate_fitness();
    }

    fn select_survivors(&mut self, offspring: Vec<Individual>, tournsize: usize) -> Vec<Individual> {
        let mut combined: Vec<Individual> = self.population.clone();
        combined.extend(offspring);
        let mut selected: Vec<Individual> = Vec::new();
        // let mut rng = rand::thread_rng();
        while selected.len() < self.population_size {
            let mut tournament: Vec<Individual> = Vec::new();
            for _ in 0..tournsize {
                let index = self.rng.gen_range(0..combined.len());
                tournament.push(combined[index].clone());
            }
            let best = tournament.iter().min_by(|a: &&Individual, b: &&Individual| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap();
            selected.push(best.clone());
        }
        selected
    }

}