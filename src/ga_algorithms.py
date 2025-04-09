environment.pyimport numpy as np
import copy

class EvolutionaryAlgorithm:
    """
    A generic evolutionary algorithm class containing common functionality.
    """
    def __init__(self, generations=100, population_size=100, max_value=5, min_value=0, mutation_mean=0, mutation_std=0.1):
        self.generations = generations         # Number of generations to run
        self.population_size = population_size  # Size of the population
        self.max_value = max_value              # Maximum value for genotype elements
        self.min_value = min_value              # Minimum value for genotype elements
        self.mutation_mean = mutation_mean      # Mean of mutation noise
        self.mutation_std = mutation_std        # Standard deviation for mutation noise

    def generate_genotype(self):
        """
        Generates a random genotype within the allowed range.
        """
        return np.random.rand(6) * self.max_value

    def initialize_population(self):
        """
        Initializes the population with random genotypes.
        """
        return [self.generate_genotype() for _ in range(self.population_size)]

    def mutate(self, genotype):
        """
        Applies Gaussian mutation to the given genotype.
        """
        mutated_genotype = copy.deepcopy(genotype)
        noise = np.random.normal(self.mutation_mean, self.mutation_std, size=mutated_genotype.shape)
        mutated_genotype += noise

        # Ensure that mutated genotype values remain within bounds
        mutated_genotype[mutated_genotype > self.max_value] = self.max_value
        mutated_genotype[mutated_genotype < self.min_value] = self.min_value

        return mutated_genotype

    def fitness(self, positions, intensities, lightsource=(0, 0)):
        """
        Calculates fitness based on the reduction in distance to the light source.
        """
        start_dist = np.linalg.norm(positions[:, 0] - np.array(lightsource))
        end_dist = np.linalg.norm(positions[:, -1] - np.array(lightsource))
        distance_gained = start_dist - end_dist
        return distance_gained / start_dist

    def evaluate_fitness(self, population, env, starting_position, starting_bearing, runtime):
        """
        Evaluates the fitness of each individual in the population.
        """
        fitness_scores = []
        for genotype in population:
            # Create an agent with the given genotype (agent creation will be handled externally)
            from braitenberg import Braitenberg  # Dynamic import to decouple dependency
            agent = Braitenberg(starting_position, starting_bearing, genotype)
            trajectory, intensities = env.run(runtime, agent, show=False)
            fitness_scores.append(self.fitness(trajectory, intensities))
        return np.array(fitness_scores)

    def run(self, starting_position, starting_bearing, env, runtime=5, track_fitness=False):
        """
        Runs the evolutionary algorithm for a set number of generations.
        Returns the best genotype and optionally the fitness trend.
        """
        population = self.initialize_population()
        fitness_trend = []

        for generation in range(self.generations):
            fitness_scores = self.evaluate_fitness(population, env, starting_position, starting_bearing, runtime)
            if track_fitness:
                fitness_trend.append(np.max(fitness_scores))
            new_population = self.evolve_population(population, fitness_scores)
            population = new_population

        best_index = np.argmax(fitness_scores)
        if track_fitness:
            return population[best_index], fitness_scores[best_index], fitness_trend
        return population[best_index], fitness_scores[best_index]

    def evolve_population(self, population, fitness_scores):
        """
        Placeholder method to evolve population.
        Must be implemented in child classes.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class FullMicrobialGA(EvolutionaryAlgorithm):
    """
    Full Microbial Genetic Algorithm implementation.
    """
    def __init__(self, generations=100, population_size=100, max_value=5, min_value=0, mutation_mean=0, mutation_std=0.1, elitism_count=10):
        super().__init__(generations, population_size, max_value, min_value, mutation_mean, mutation_std)
        self.elitism_count = elitism_count  # Number of top individuals to preserve

    def evolve_population(self, population, fitness_scores):
        """
        Evolves the population using microbial GA strategy with elitism.
        """
        # Sort indices based on fitness (descending order)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_population = [population[i] for i in sorted_indices[:self.elitism_count]]

        # Fill the rest of the population via tournament selection and mutation
        while len(new_population) < self.population_size:
            a = np.random.randint(0, self.population_size)
            b = np.random.randint(0, self.population_size)
            if fitness_scores[a] > fitness_scores[b]:
                new_individual = self.mutate(population[a])
            else:
                new_individual = self.mutate(population[b])
            new_population.append(new_individual)
        return new_population

class SpatialGA(EvolutionaryAlgorithm):
    """
    Spatial Genetic Algorithm implementation.
    Individuals are arranged in a grid and interact with their neighbors.
    """
    def __init__(self, population_size=10, generations=100, runtime=5, grid_size=None, max_value=5, min_value=0, mutation_mean=0, mutation_std=0.1):
        super().__init__(generations, population_size, max_value, min_value, mutation_mean, mutation_std)
        self.runtime = runtime
        if grid_size is None:
            side_length = int(np.sqrt(population_size))
            if side_length * side_length < population_size:
                side_length += 1
            self.grid_size = (side_length, side_length)
        else:
            self.grid_size = grid_size
        assert self.grid_size[0] * self.grid_size[1] >= population_size, "Grid size must be at least as large as the population size"

    def get_neighbors(self, index):
        """
        Returns the indices of the neighboring individuals in the grid.
        """
        x = index % self.grid_size[0]
        y = index // self.grid_size[0]
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.grid_size[0]
                ny = (y + dy) % self.grid_size[1]
                neighbor_index = ny * self.grid_size[0] + nx
                if neighbor_index < self.population_size:
                    neighbors.append(neighbor_index)
        return neighbors

    def evolve_population(self, population, fitness_scores):
        """
        Evolve the population by comparing each individual with its neighbors.
        """
        new_population = population[:]
        for i in range(self.population_size):
            neighbors = self.get_neighbors(i)
            neighbor_fitness = [fitness_scores[n] for n in neighbors]
            best_neighbor = neighbors[np.argmax(neighbor_fitness)]
            if fitness_scores[i] < fitness_scores[best_neighbor]:
                new_population[i] = self.mutate(population[best_neighbor])
        return new_population
