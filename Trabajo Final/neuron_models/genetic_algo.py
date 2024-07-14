import numpy as np
from typing import Tuple, Callable


class GeneticAlgorithm(object):
    def __init__(self, pop_size: int,
                 N_iter: int,
                 max_rep: int,
                 mut_rate: float,
                 fitness_function: Callable,
                 init_pars: np.ndarray):
        """
        Definimos el objeto de algoritmo genético
        :param pop_size: cantidad de individuos en la población
        :param N_iter: cantidad máxima de generaciones
        :param max_rep: cantidad máxima permitida de genraciones sucesivas sin mejora
        :param mut_rate: pernece a [0, 1], indica la probabilidad de mutar de un individuo
        :param fitness_function: función que define qué tan buena es una población
        :param init_pars: valores iniciales para los parámetros del algoritmo
        """

        self.pop_size = pop_size
        self.N_iter = N_iter
        self.max_rep = max_rep
        self.mut_rate = mut_rate
        self.fitness_function = fitness_function
        self.population = self.initialize_population(init_pars)


    def initialize_population(self, init_pars: np.ndarray) -> np.ndarray:
        return np.array([init_pars + np.random.normal() for _ in range(self.pop_size)])


    def select_parents(self, fitnesses, num_parents):
        return self.population[np.argsort(fitnesses)[-num_parents:]]


    def crossover(self, parents, offspring_size: Tuple[int, int]):
        offspring = np.empty(offspring_size)
        crossover_point = offspring_size[1] // 2
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring


    def mutate(self, offspring):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.mut_rate:
                mutation_idx = np.random.randint(0, offspring.shape[1])
                offspring[idx, mutation_idx] += np.random.normal()
        return offspring


    def genetic_algorithm(self, *args, **kwargs) -> np.ndarray:
        """
        Generar poblaciones iterativas, seleccionar a los mas aptos, reproducirlos, mutar algunos, repetir
        :param args: argumentos por posicion para la funcion de fitness
        :param kwargs: argumentos por keyword para la funcion de fitness
        :return: la mejor combinacion de parametros
        """
        param_size = self.population.shape[1]
        best_solution = self.population[0]
        best_fitness = -np.inf
        rep_count = 0

        for generation in range(self.N_iter):
            # Hallamos la fitness de cada miembro de la poblacion
            fitnesses = np.array([self.fitness_function(params, *args, **kwargs) for params in self.population])
            # Identificamos al mejor, y actualizamos de ser el necesario
            if np.max(fitnesses) > best_fitness:
                rep_count = 0
                best_fitness = np.max(fitnesses)
                best_solution = self.population[np.argmax(fitnesses)].copy()

            # Seleccion de padres, crossover, y mutacion
            parents = self.select_parents(fitnesses, self.pop_size // 2)
            offspring_crossover = self.crossover(parents, (self.pop_size - parents.shape[0], param_size))
            offspring_mutation = self.mutate(offspring_crossover)

            # Actualizacion de la poblacion
            self.population[:parents.shape[0]] = parents
            self.population[parents.shape[0]:] = offspring_mutation

            # Strings para mostrar al usuario
            par_str = ', '.join([f"{par:3f}" for par in best_solution])
            print(f"Generation {generation}: Best Fitness = {best_fitness}", end=' ')
            print(f"Params: [{par_str}]")

            rep_count += 1
            # Si llevamos más de max_rep generaciones sin mejorar, nos quedamos con lo que tenemos
            if rep_count > self.max_rep:
                print('No more evolution')
                break

        return best_solution