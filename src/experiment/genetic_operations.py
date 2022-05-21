from typing import Callable, Iterable, List, Tuple

import numpy as np
from numba import njit


def __swap(sw):
    if sw == 0:
        return 1
    else:
        return 0


def __crossover(indiv_flat_size: int, offspring: np.ndarray, parent_index: int, parents: np.ndarray, crossover_sections: List):
    for s, e in crossover_sections:
        if s != e:
            offspring[0].reshape(indiv_flat_size)[
                s:e] = parents[parent_index, __swap(1)].reshape(indiv_flat_size)[s:e]
            offspring[1].reshape(indiv_flat_size)[
                s:e] = parents[parent_index, __swap(0)].reshape(indiv_flat_size)[s:e]


def __mutate_allel(sw: np.ndarray) -> np.ndarray:
    indices_one = sw == 1
    indices_zero = sw == 0
    sw[indices_one] = 0  # replacing 1s with 0s
    sw[indices_zero] = 1  # replacing 0s with 1s
    return sw


def __random_selection(size, pop_size) -> Tuple[np.ndarray]:
    return tuple(np.random.randint(low=0, high=pop_size, size=size))


def __spin_wheel(fitnesses: np.ndarray, sum_of_fitnesses: int) -> int:
    pin_point = np.random.randint(low=0, high=sum_of_fitnesses)
    for index, fitness in enumerate(fitnesses):
        pin_point = pin_point + fitness
        if pin_point > sum_of_fitnesses:
            return index
    return index


def __tournament_match(population_with_fitness: List[Tuple]) -> np.ndarray:
    rivals = __random_selection(size=2, pop_size=len(population_with_fitness))
    return rivals[np.argmax([population_with_fitness[i][0] for i in rivals])]


def select(selection_method: Callable, size: int, population_with_fitness: List[Tuple[int, np.ndarray]], count: int) -> Iterable[np.ndarray]:
    for _ in range(count):
        yield np.array([population_with_fitness[i][1] for i in selection_method(population_with_fitness, size)])


def crossover(parents: np.ndarray, indiv_flat_size: int, number_of_crossover_section: int) -> np.ndarray:
    crossover_points = [p for p in np.sort(np.random.randint(
        low=1, high=indiv_flat_size, size=number_of_crossover_section))]
    crossover_sections = list(
        zip([0]+crossover_points, crossover_points+[indiv_flat_size]))
    offsprings = np.zeros(parents.shape, dtype='int')
    for index, offspring in enumerate(offsprings):
        __crossover(indiv_flat_size, offspring, index,
                    parents, crossover_sections)
    return offsprings


def mutation(generation, indiv_size, indiv_flat_size, pop_size, mutation_prob, mutation_amount):
    mutation_indicies = np.random.random_sample(
        size=(pop_size-2)) < mutation_prob
    mutants = generation[mutation_indicies]
    mutants = mutants.reshape((mutants.shape[0], indiv_flat_size))
    mutation_positions = np.random.randint(low=0, high=mutants.shape[1],
                                           size=(mutants.shape[0], int(mutation_amount)))
    for mutant_index, mutation_position in enumerate(mutation_positions):
        mutants[mutant_index, mutation_position] = __mutate_allel(
            mutants[mutant_index, mutation_position])

    generation[mutation_indicies] = mutants.reshape(
        ((mutants.shape[0],) + indiv_size))

    return generation


def roulettewheel_selection(population_with_fitness: List[Tuple], size: int) -> Tuple[np.ndarray]:
    fitnesses = np.array([f for (f, i) in population_with_fitness])
    sum_of_fitnesses = fitnesses.sum()
    if sum_of_fitnesses == 0:
        return __random_selection(size=size, pop_size=len(population_with_fitness))
    else:
        return tuple(np.array([__spin_wheel(fitnesses, sum_of_fitnesses) for i in range(size)]))


def tournament_selection(population_with_fitness: List[Tuple], size):
    return tuple([__tournament_match(population_with_fitness) for _ in range(size)])


def genetic_iteration(fn_fitness: Callable, indiv_size: Tuple, indiv_flat_size: int, population: List, selection_method: Callable, number_of_crossover_section: int, mutation_prob: float, mutation_amount: int, fitness_history: List) -> List:
    pop_size = len(population)

    fitness_results = fn_fitness(population)

    population_with_fitness: List[Tuple[int, np.ndarray]] = sorted(
        fitness_results, key=lambda tup: tup[0], reverse=True)

    fitness_history.append(population_with_fitness[0][0])
    elit1 = population_with_fitness[0][1]
    elit2 = population_with_fitness[1][1]

    parents = np.stack([a for a in select(
        selection_method, 2, population_with_fitness, (pop_size-2)//2)])

    next_generation = crossover(parents, indiv_flat_size, number_of_crossover_section).reshape(
        ((pop_size-2,) + indiv_size))

    next_generation = mutation(next_generation, indiv_size,
                               indiv_flat_size, pop_size, mutation_prob, mutation_amount)

    next_generation = np.append(next_generation, elit1)
    next_generation = np.append(next_generation, elit2)
    return next_generation.reshape(((pop_size,) + indiv_size))
