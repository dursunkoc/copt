from time import time
import sys
from typing import Callable, List, Tuple
import numpy as np
#from numba import njit, prange, jit
from numba.typed import Dict
from numba.core import types
from tqdm import trange, tqdm
from base_network_opt import solve_network_model_definite
import math

import logging
import net_opt_case as noc

from genetic_operations import genetic_iteration, tournament_selection, roulettewheel_selection

logger = logging.getLogger("GA-NET-OP-LOGGER")
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S'))
logger.addHandler(log_handler)

POPULATION_SIZE = "P_S"
INDIV_SIZE = "I_S"
GENERATION_SIZE = "G_S"
CHECK_BACK = "CB_C"
NUMBER_OF_CROSS_OVER_SECTION = "N_COS"
MUTATION_PROP = "MUT_PB"
MUTATION_AMOUNT = "MUT_AM"


def random_initial_population(indiv_size: Tuple[int], population_size: int) -> np.ndarray:
    return np.random.choice(2, ((population_size,) + indiv_size))


def hard_initial_population(indiv_size: int, population_size: int, e_u:np.ndarray) -> np.ndarray:
     h1 = np.ones((population_size//2,) + indiv_size, dtype='int')
     h0 = np.zeros(((population_size//2)-1,) + indiv_size, dtype='int')
#     sol=np.array([[0,1,0,0,1,0,1,0,0,0]])
     return np.concatenate((h1,h0,e_u.reshape((1,*e_u.shape))))


def fn_fitness(a_uv: np.ndarray, e_u: np.ndarray) -> Callable:
    def fn_fitness0(population: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        return [(fn_fitness_indiv(indiv, a_uv, e_u), indiv) for indiv in population]
#    return njit(fn_fitness0)
    return fn_fitness0

def fn_fitness_fixing(a_uv: np.ndarray, e_u: np.ndarray) -> Callable:
    def fn_fitness0_fixing(population: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        return [(fn_fitness_indiv_fixing(indiv, a_uv, e_u), indiv) for indiv in population]
#    return njit(fn_fitness0_fixing)
    return fn_fitness0_fixing



def fn_fitness_fixing_ls(a_uv: np.ndarray, e_u: np.ndarray, search_size: int, search_ext: int) -> Callable:
    def fn_fitness0_fixing_ls(population: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        return [(fn_fitness_indiv_fixing_with_ls(indiv, a_uv, e_u, search_size, search_ext), indiv) for indiv in population]
#    return njit(fn_fitness0_fixing)
    return fn_fitness0_fixing_ls



#@njit()
def max_fit(shape):
    acc = 1
    for s in shape:
        acc = acc*s
    return acc

#@njit
def fix_to_fit(indiv, a_uv, e_u):
    if np.random.randint(10) <= 10:
        diff = np.asarray(( indiv * a_uv ) - e_u)[0]
        fails = diff<0
        eligibles = e_u!=0
        canfix = indiv==0
        indiv[fails & eligibles & canfix] = 1

#@njit
def increase_fitness(indiv, a_uv, e_u):
    if np.random.randint(100) <= 10:
        for i in range(len(indiv)):
            if indiv[i]==1:
                indiv[i]=0
                if not check_constraint(indiv, a_uv, e_u):
                    indiv[i]=1

#@njit()
def fn_fitness_indiv_fixing(indiv: np.ndarray, a_uv: np.ndarray, e_u: np.ndarray) -> int:
    mf = max_fit(indiv.shape)
    if not check_constraint(indiv, a_uv, e_u):
        fix_to_fit(indiv, a_uv, e_u)
        if not check_constraint(indiv, a_uv, e_u):
            return 0
    increase_fitness(indiv, a_uv, e_u)
    return mf-np.sum(indiv)

def fn_fitness_indiv_fixing_with_ls(indiv: np.ndarray, a_uv: np.ndarray, e_u: np.ndarray, search_size: int, search_ext: int) -> int:
    mf = max_fit(indiv.shape)
    r_indiv = indiv
    r_val = fn_fitness_indiv_fixing(indiv, a_uv, e_u)
    for _ in range(search_size):
        if np.where([indiv==0])[1].size > 0:
            ext_loc = np.unique(np.random.choice(np.where([indiv==1])[1], size=search_ext))
            if ext_loc.size > 0:
                l_indiv = indiv.copy()
                l_indiv.put(ext_loc, 0)
                l_val = fn_fitness_indiv_fixing(l_indiv, a_uv, e_u)
                if l_val > r_val:
                    r_val = l_val
                    r_indiv = l_indiv
    indiv[True] = r_indiv
    return r_val

#@njit()
def fn_fitness_indiv(indiv: np.ndarray, a_uv: np.ndarray, e_u: np.ndarray) -> int:
    mf = max_fit(indiv.shape)
    if not check_constraint(indiv, a_uv, e_u):
        return 0
    return mf-np.sum(indiv)

#@njit()
def check_constraint(indiv: np.ndarray, a_uv: np.ndarray, e_u: np.ndarray) -> bool:
#    result = solve_network_model_definite(a_uv, len(indiv), e_u, indiv)
#    return result is not None
    return (indiv + (indiv*a_uv)>=e_u).all()


def main(fn_fitness0: Callable, next_generation:np.ndarray, indiv_size: Tuple[int],
         flat_indiv_size: int, generation_size: int, population_size: int, check_back: int,
         number_of_crossover_section: int, mutation_prob: float, mutation_amount: int):
    start = time()
    fitness_history = []
    pbar = trange(generation_size)
    for generation_index in pbar:
        next_generation = genetic_iteration(
            fn_fitness=fn_fitness0,
            indiv_size=indiv_size,
            indiv_flat_size=flat_indiv_size,
            population=next_generation,
            selection_method=tournament_selection,
            number_of_crossover_section=number_of_crossover_section,
            mutation_prob=mutation_prob,
            mutation_amount=mutation_amount,
            fitness_history=fitness_history)
        pbar.set_description('max fitness: '+str([(flat_indiv_size - f)for f in sorted(set(fitness_history))]))
        if generation_index > check_back and fitness_history[generation_index] <= fitness_history[generation_index-check_back]:
            break
        

    fitness_results = fn_fitness0(next_generation)
    population_with_fitness: List[Tuple[int, np.ndarray]] = sorted(
        fitness_results, key=lambda tup: tup[0], reverse=True)
    fitness_value = population_with_fitness[0][0]
    value = int(flat_indiv_size - fitness_value)
    
    end = time()

    print(f"------------------------------Staring GA Solution------------------------------")
    print(f"|Parametes: ")
    print(f"|fitness_function            : {fn_fitness0}")
    print(f"|indiv_size                  : {indiv_size}")
    print(f"|flat_indiv_size             : {flat_indiv_size}")
    print(f"|generation_size             : {generation_size}")
    print(f"|population_size             : {population_size}")
    print(f"|check_back                  : {check_back}")
    print(f"|number_of_crossover_section : {number_of_crossover_section}")
    print(f"|mutation_prob               : {mutation_prob}")
    print(f"|mutation_amount             : {mutation_amount}")
    print(f"|iterations completed        : {generation_index}")
    print(f"|duration                    : {end-start} seconds")
    print(f"|Case: customer_size={flat_indiv_size}, Objective is: {value}")
    print(f"---------------------------------------------------------------------------------")
    
#    print(value)
#    print(population_with_fitness[0][1])
#    print(fitness_history)

def get_population_size(indiv_size):
    q=indiv_size
    if q%2 != 0:
        q=q+1
    return min(max(30,q),50)

if __name__ == '__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    for case in noc.cases:
        population_size = get_population_size(case.indiv_size)

        fn_fitness0 = fn_fitness(case.a_uv, case.e_u)
        fn_fitness1 = fn_fitness_fixing(case.a_uv, case.e_u)
        fn_fitness2 = fn_fitness_fixing_ls(case.a_uv, case.e_u, search_size=math.ceil(population_size/10), search_ext=math.ceil(case.indiv_size//20))

#        next_generation = random_initial_population(indiv_size, population_size)   
        next_generation = hard_initial_population((case.indiv_size,), population_size, case.e_u)

        main(fn_fitness0=fn_fitness0, flat_indiv_size=case.indiv_size, next_generation=next_generation,
             indiv_size=(case.indiv_size,), generation_size=2000, population_size=population_size, check_back=40,
             number_of_crossover_section=max(5, case.indiv_size//5), mutation_prob=1, mutation_amount=min(10, max(5,case.indiv_size/10)))

        main(fn_fitness0=fn_fitness1, flat_indiv_size=case.indiv_size, next_generation=next_generation,
             indiv_size=(case.indiv_size,), generation_size=2000, population_size=population_size, check_back=40,
             number_of_crossover_section=max(5, case.indiv_size//5), mutation_prob=1, mutation_amount=min(10, max(5,case.indiv_size/10)))

        main(fn_fitness0=fn_fitness2, flat_indiv_size=case.indiv_size, next_generation=next_generation,
             indiv_size=(case.indiv_size,), generation_size=2000, population_size=population_size, check_back=40,
             number_of_crossover_section=max(5, case.indiv_size//5), mutation_prob=1, mutation_amount=min(10, max(5,case.indiv_size/10)))

#❯ workon copt
#❯ python src/experiment/mip_net_opt.py
#Case: customer_size=10, Objective is: 3.0
#Case: customer_size=20, Objective is: 7.0
#Case: customer_size=50, Objective is: 16.0
#Case: customer_size=100, Objective is: 37.0
#Case: customer_size=200, Objective is: 67.0
#Case: customer_size=500, Objective is: 160.0
#Case: customer_size=1000, Objective is: 352.0
#Case: customer_size=5000, Objective is: 1694.0
#Case: customer_size=10000, Objective is: 3402.0
#❯
#❯ python src/experiment/ga_net_opt.py
#  6%|███████▌                                                                                                                                  | 110/2000 [00:00<00:07, 242.48it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (10,)
#|flat_indiv_size             : 10
#|generation_size             : 2000
#|population_size             : 30
#|check_back                  : 80
#|number_of_crossover_section : 5
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 110
#|duration                    : 0.460129976272583 seconds
#|Case: customer_size=10, Objective is: 3
#---------------------------------------------------------------------------------
#  4%|█████▉                                                                                                                                     | 86/2000 [00:00<00:07, 241.79it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (20,)
#|flat_indiv_size             : 20
#|generation_size             : 2000
#|population_size             : 30
#|check_back                  : 80
#|number_of_crossover_section : 5
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 86
#|duration                    : 0.35794878005981445 seconds
#|Case: customer_size=20, Objective is: 8
#---------------------------------------------------------------------------------
#  7%|█████████▌                                                                                                                                | 139/2000 [00:01<00:15, 120.24it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (50,)
#|flat_indiv_size             : 50
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 80
#|number_of_crossover_section : 10
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 139
#|duration                    : 1.1599302291870117 seconds
#|Case: customer_size=50, Objective is: 20
#---------------------------------------------------------------------------------
#  6%|████████                                                                                                                                   | 116/2000 [00:02<00:42, 44.06it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (100,)
#|flat_indiv_size             : 100
#|generation_size             : 2000
#|population_size             : 100
#|check_back                  : 80
#|number_of_crossover_section : 20
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 116
#|duration                    : 2.6407480239868164 seconds
#|Case: customer_size=100, Objective is: 51
#---------------------------------------------------------------------------------
#  4%|█████▋                                                                                                                                      | 81/2000 [00:05<02:20, 13.64it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (200,)
#|flat_indiv_size             : 200
#|generation_size             : 2000
#|population_size             : 200
#|check_back                  : 80
#|number_of_crossover_section : 40
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 81
#|duration                    : 5.954800844192505 seconds
#|Case: customer_size=200, Objective is: 94
#---------------------------------------------------------------------------------
#  7%|█████████▏                                                                                                                                 | 133/2000 [01:08<16:01,  1.94it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (500,)
#|flat_indiv_size             : 500
#|generation_size             : 2000
#|population_size             : 500
#|check_back                  : 80
#|number_of_crossover_section : 100
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 133
#|duration                    : 68.69252824783325 seconds
#|Case: customer_size=500, Objective is: 246
#---------------------------------------------------------------------------------
#  4%|█████▌                                                                                                                                    | 81/2000 [12:33<4:57:34,  9.30s/it]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (1000,)
#|flat_indiv_size             : 1000
#|generation_size             : 2000
#|population_size             : 1000
#|check_back                  : 80
#|number_of_crossover_section : 200
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 81
#|duration                    : 761.3674039840698 seconds
#|Case: customer_size=1000, Objective is: 508
#---------------------------------------------------------------------------------
#############################################################################################
#❯ python src/experiment/ga_net_opt.py
#  2%|████▋                                                                                                                                                                                                                                 | 41/2000 [00:00<00:11, 172.45it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (10,)
#|flat_indiv_size             : 10
#|generation_size             : 2000
#|population_size             : 30
#|check_back                  : 40
#|number_of_crossover_section : 5
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 41
#|duration                    : 0.2470548152923584 seconds
#|Case: customer_size=10, Objective is: 3
#---------------------------------------------------------------------------------
#  2%|████▋                                                                                                                                                                                                                                 | 41/2000 [00:00<00:11, 171.01it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (10,)
#|flat_indiv_size             : 10
#|generation_size             : 2000
#|population_size             : 30
#|check_back                  : 40
#|number_of_crossover_section : 5
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 41
#|duration                    : 0.24664092063903809 seconds
#|Case: customer_size=10, Objective is: 3
#---------------------------------------------------------------------------------
#  3%|█████▊                                                                                                                                                                                                                                | 51/2000 [00:00<00:12, 150.28it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (20,)
#|flat_indiv_size             : 20
#|generation_size             : 2000
#|population_size             : 30
#|check_back                  : 40
#|number_of_crossover_section : 5
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 51
#|duration                    : 0.343886137008667 seconds
#|Case: customer_size=20, Objective is: 7
#---------------------------------------------------------------------------------
#  2%|████▊                                                                                                                                                                                                                                  | 42/2000 [00:00<00:24, 78.80it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (50,)
#|flat_indiv_size             : 50
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 40
#|number_of_crossover_section : 10
#|mutation_prob               : 1
#|mutation_amount             : 5
#|iterations completed        : 42
#|duration                    : 0.5401191711425781 seconds
#|Case: customer_size=50, Objective is: 16
#---------------------------------------------------------------------------------
#  2%|████▊                                                                                                                                                                                                                                  | 42/2000 [00:00<00:31, 61.29it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (100,)
#|flat_indiv_size             : 100
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 40
#|number_of_crossover_section : 20
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 42
#|duration                    : 0.6913762092590332 seconds
#|Case: customer_size=100, Objective is: 39
#---------------------------------------------------------------------------------
#  6%|█████████████▍                                                                                                                                                                                                                        | 117/2000 [00:03<00:56, 33.44it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (200,)
#|flat_indiv_size             : 200
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 40
#|number_of_crossover_section : 40
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 117
#|duration                    : 3.5084328651428223 seconds
#|Case: customer_size=200, Objective is: 67
#---------------------------------------------------------------------------------
# 10%|███████████████████████                                                                                                                                                                                                               | 200/2000 [00:31<04:39,  6.45it/s]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (500,)
#|flat_indiv_size             : 500
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 40
#|number_of_crossover_section : 100
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 200
#|duration                    : 31.06557011604309 seconds
#|Case: customer_size=500, Objective is: 170
#---------------------------------------------------------------------------------
#  5%|███████████▏                                                                                                                                                                                                                         | 98/2000 [05:16<1:42:16,  3.23s/it]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (1000,)
#|flat_indiv_size             : 1000
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 40
#|number_of_crossover_section : 200
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 98
#|duration                    : 317.028844833374 seconds
#|Case: customer_size=1000, Objective is: 373
#---------------------------------------------------------------------------------
#  3%|███████▉                                                                                                                                                                                                                             | 69/2000 [04:19<2:01:15,  3.77s/it]
#------------------------------Staring GA Solution------------------------------
#|Parametes:
#|indiv_size                  : (1000,)
#|flat_indiv_size             : 1000
#|generation_size             : 2000
#|population_size             : 50
#|check_back                  : 40
#|number_of_crossover_section : 200
#|mutation_prob               : 1
#|mutation_amount             : 10
#|iterations completed        : 69
#|duration                    : 263.8862693309784 seconds
#|Case: customer_size=1000, Objective is: 392
#---------------------------------------------------------------------------------