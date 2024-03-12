# Evolutionary Algorithms
This repository consists of two implementations of an evolutionary algorithm, one in Python - and one in Rust. Both have the
same implemented logic, and serves as an exercise to improve my Rust abilities.

## Logic
This script demonstrates the usage of a evolutionary algorithm to solve an optimization problem.
It evolves a population of individuals over a number of generations to find the best solution.

The objective function is to minimize the sum of squared differences between the genes of an individual and a target individual.
The target individual is a sequence of numbers from 0 to INDIVIDUAL_SIZE - 1 in this example, but could be any sequence of numbers.

The evolution process consists of the following steps:
1. Generate an initial population of individuals with random genes.
2. Evaluate the fitness of each individual by calculating the objective function.
3. Apply crossover and mutation to create offspring.
4. Select parents for the next generation using tournament selection from the combined population of parents and offspring.
5. Replace the old population with the new population of offspring.
6. Repeat steps 2-5 for a fixed period of time or until an individual is found with a fitness below a certain threshold.

## Install & Run
Python, Pip and Cargo are required dependencies.

To run the Python script:
```
cd ea_python \
&& pip install numpy \
&& python main.py
```

To run the Rust script:
```
cd ea_rust \
&& cargo run
```
