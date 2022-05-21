from datetime import datetime
from typing import Dict, List, Tuple, Iterable
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from co_constraints import do_check_all
from tqdm import trange
import numpy as np
from numba import njit, jit
from multiprocessing.pool import ThreadPool
from genetic_operations import genetic_iteration, tournament_selection


THREADS_ = 2
#POOL_ = ThreadPool(THREADS_)

import logging

logger = logging.getLogger("GA-LOGGER")
logger.setLevel(logging.INFO)

import sys
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S'))
logger.addHandler(log_handler)

def indiv_binary_rep(indiv):
    return "".join(map(lambda b: str(int(b)),indiv.reshape(np.prod(indiv.shape)).tolist()))

@njit
def objective_fn_no_net(rp_c, X):
    return (rp_c * X.sum(1).sum(1).sum(1)).sum()

@njit
def __fn_fitness_for_indiv(indiv:np.ndarray, rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd,):
    if not do_check_all(indiv,b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd):
        return 0
    return objective_fn_no_net(rp_c, indiv)


@njit(parallel=True)
def fn_fitness(population, rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd):
#    pop_size = len(population)
#    for index in prange(pop_size):
#        __fn_fitness_for_indiv(population[index], rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd)
    return [(__fn_fitness_for_indiv(indiv, rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd), indiv) for indiv in population]


class GASolution(Solution):
    def __init__(self):
        super().__init__("Genetic Algorihm")

    def run(self, case:Case, ph=False)->SolutionResult:
        from time import time
        import operator
        import functools
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        PMS:Parameters = super().generate_parameters(case)
        
        #variables
        indiv_size = (C,U,H,D)
        indiv_flat_size = functools.reduce(operator.mul, indiv_size, 1)
        pop_size = 3*THREADS_#functools.reduce(operator.mul, indiv_size, 1)*8
        generation_size = 100
        trace_back_counter = 40
        mutation_prob = 1
        mutation_rate = 0.4
        mutation_amount = int(mutation_rate*indiv_flat_size)
        number_of_crossover_section = U
        
        X_X=[z for z in self.gen_greedy(indiv_size, PMS, C, D, H, U)]
#        M_X = self.gen_greedy0(indiv_size, PMS, C, D, H, U)
#        X_X.append(M_X)
        
        next_generation = self.fn_initialize_population(pop_size=pop_size,X_X=X_X)
        fn_fitness = self.fn_fitness(PMS)
        
        fitness_history = []
        for generation_index in trange(generation_size):
            next_generation = genetic_iteration(
                fn_fitness=fn_fitness,
                indiv_size=indiv_size,
                indiv_flat_size=indiv_flat_size,
                population=next_generation, 
                selection_method=tournament_selection,
                number_of_crossover_section=number_of_crossover_section,
                mutation_prob=mutation_prob,
                mutation_amount=mutation_amount,
                fitness_history=fitness_history)
            if generation_index > trace_back_counter and fitness_history[generation_index]<=fitness_history[generation_index-trace_back_counter]:
                break
#            if generation_index%100 == 0 and mutation_prob > 0.05:
#                mutation_prob = mutation_prob *.8
#            if generation_index%100 == 0 and mutation_amount > mutation_rate*indiv_flat_size*.05:
#                mutation_amount = mutation_amount - mutation_rate*indiv_flat_size*.05

        fitness_results = fn_fitness(next_generation)
        population_with_fitness:List[Tuple[int,np.ndarray]] = sorted(fitness_results, key = lambda tup: tup[0], reverse=True)
        value = population_with_fitness[0][0]
        end_time = time()
        duration = end_time - start_time
        return SolutionResult(case, value, round(duration,4))

    
    def fn_fitness(self, PMS)->np.ndarray:
#        asyn_fitness_results=[POOL_.apply_async(self.fn_fitness, args=(PMS, popl)) for popl in np.split(next_generation, THREADS_)]
#        fitness_results = sum([asyn_result.get() for asyn_result in asyn_fitness_results], [])
        def fn_fitness0(population:np.ndarray)->np.ndarray:
            return fn_fitness(population, PMS.rp_c, PMS.b, PMS.cuhd, PMS.e_cu, PMS.k, PMS.l_c, PMS.m_i, PMS.n_i, PMS.q_ic, PMS.s_cuhd, PMS.t_hd)
        return fn_fitness0

    def gen_greedy(self, indiv_size:Tuple, PMS:Parameters, C:int, D:int, H:int, U:int )->Iterable[np.ndarray]:
        for c in range(C):#tqdm(np.argsort(-MP.rp_c), desc="Campaigns Loop"):
            X = np.zeros(indiv_size, dtype=np.int0)
            for d in range(D):#trange(MP.D, desc=f"Days Loop"):            
                for h in range(H):#trange(H, desc=f"Channels Loop at Day-{d}, Campapaign-{c}"):
                    for u in range(U):#trange(U, desc=f"Users Loop On Campaign-{c}"):
                        X[c,u,h,d]=1
                        if not self.check(X, PMS, (c, u, h, d)):
                            X[c,u,h,d]=0
            yield X

    def gen_greedy0(self, indiv_size:Tuple, PMS:Parameters, C:int, D:int, H:int, U:int )->Iterable[np.ndarray]:
        X = np.zeros(indiv_size, dtype=np.int0)
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        X[c,u,h,d]=1
                        if not self.check(X, PMS, (c, u, h, d)):
                            X[c,u,h,d]=0
        return X

    def fn_initialize_population(self, pop_size:int, X_X:List)->List[np.ndarray]:
        population = [X_X[i%len(X_X)] for i in range(pop_size)]
        return np.stack(population)


if __name__ == '__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GASolution(), False)
    for solution in solutions:
        print(solution)


#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 8580, duration: 135.0099>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 120160, duration: 1096.2517>
#<case: {'C': 15, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1800000, duration: 27886.2746>

#{'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 13083, 'duration': 6.346, 'info': None}
#
#{'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 12180, 'duration': 11.6786, 'info': None}
#
#{'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 63096, 'duration': 24.0241, 'info': None}
#
#{'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 74664, 'duration': 102.3899, 'info': None}
#
#{'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 340699, 'duration': 145.1705, 'info': None}
#
#{'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 658443, 'duration': 343.6769, 'info': None}
#
#{'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 1672590, 'duration': 401.4614, 'info': None}
#
#{'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 1158598, 'duration': 647.0601, 'info': None}
#
#{'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 1979939, 'duration': 697.714, 'info': None}
#
#{'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 5287034, 'duration': 3306.7774, 'info': None}
#
#{'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 4941547, 'duration': 5342.363, 'info': None}
#
#{'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'value': 10256858, 'duration': 10116.4649, 'info': None}
