import numpy as np
from typing import Tuple, List, Callable

class GeneticAlgorithm(object):
    def __init__(self, fit_func: Callable, predict: Callable):
        pass
    def fit_spikes_GA(self, t: np.ndarray,
                    obj_spikes: np.ndarray,
                    I_input: np.ndarray,
                    n_per_bin: int,
                    N_iter: int=1000,
                    popu_size: int=100,
                    mut_rate: float=0.01) -> None:
        """
        Tweaks the object's parameters to fit a voltage curve using genetic algorithms
        :param t: time array
        :param obj_spikes: neuron spike readings
        :param n_per_bin: bins for the firing rate calculation
        :param I_input: input current
        :param popu_size: population size for GA
        :param N_iter: maximum number of algorithm iterations
        :param mut_rate: float in [0, 1) for mutation rate in GA
        :return: None, but the internal parameters are tweaked to the best fitting

        """
        obj_rates = firing_rate(t, obj_spikes, n_per_bin)
        tweak_keys = ['tau_m', 'g_L']
        tweak_units = [ms, nS]
        def_pars = self.get_init_pars_2_fit(tweak_keys, tweak_units)

        def fitness_function(pars):
            self.update_params(tweak_keys, pars, tweak_units)
            _, sim_spikes = self.simulate_trajectory(t, I_input)
            sim_rates = firing_rate(t, sim_spikes, n_per_bin)
            rate_error = np.sum((obj_rates - sim_rates)**2)
            # timing_error = sum([abs(t1 - t2) for t1, t2 in zip(sim_spikes, obj_spikes)])
            return 1 / (1 + rate_error)

        def initialize_population(pop_size):
            return np.array([def_pars + np.random.normal() for _ in range(pop_size)])

        def select_parents(population, fitnesses, num_parents):
            return population[np.argsort(fitnesses)[-num_parents:]]

        def crossover(parents, offspring_size: Tuple[int, int]):
            offspring = np.empty(offspring_size)
            crossover_point = offspring_size[1] // 2
            for k in range(offspring_size[0]):
                parent1_idx = k % parents.shape[0]
                parent2_idx = (k + 1) % parents.shape[0]
                offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
                offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            return offspring

        def mutate(offspring, mutation_rate):
            for idx in range(offspring.shape[0]):
                if np.random.rand() < mutation_rate:
                    mutation_idx = np.random.randint(0, offspring.shape[1])
                    offspring[idx, mutation_idx] += np.random.normal()
            return offspring

        def genetic_algorithm(pop_size, init_pars, num_generations, mutation_rate):
            param_size = init_pars.shape[0]
            population = initialize_population(pop_size)
            best_solution = None
            best_fitness = -np.inf

            for generation in range(num_generations):
                fitnesses = np.array([fitness_function(params) for params in population])

                if np.max(fitnesses) > best_fitness:
                    best_fitness = np.max(fitnesses)
                    best_solution = population[np.argmax(fitnesses)].copy()

                parents = select_parents(population, fitnesses, pop_size // 2)
                offspring_crossover = crossover(parents, (pop_size - parents.shape[0], param_size))
                offspring_mutation = mutate(offspring_crossover, mutation_rate)

                population[:parents.shape[0]] = parents
                population[parents.shape[0]:] = offspring_mutation

                print(f"Generation {generation}: Best Fitness = {best_fitness}")

            self.update_params(tweak_keys, best_solution, tweak_units)

        genetic_algorithm(popu_size, def_pars, N_iter, mut_rate)
