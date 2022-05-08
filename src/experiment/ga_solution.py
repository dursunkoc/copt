from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Callable
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from tqdm import trange
from tqdm import tqdm
import numpy as np
from multiprocessing.pool import ThreadPool
import time 
THREADS_ = 2
POOL_ = ThreadPool(THREADS_)

def log(message):
    pass
    #print(datetime.now(), " > ", message)

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
        
        log("Starting")
        #variables
        indiv_size = (C,U,H,D)
        indiv_flat_size = functools.reduce(operator.mul, indiv_size, 1)
        pop_size = 8*THREADS_#functools.reduce(operator.mul, indiv_size, 1)*8
        generation_size = 1000
        trace_back_counter = 100
        mutation_prob = 1
        mutation_rate = 0.4
        mutation_amount = int(mutation_rate*indiv_flat_size)
        number_of_crossover_section = U
        log("Parameters Set!")
        
        log("Initializing Population")
        X_X=[z for z in self.gen_greedy(indiv_size, PMS, C, D, H, U)]
        M_X = self.gen_greedy0(indiv_size, PMS, C, D, H, U)
        X_X.append(M_X)
        log("Initializied Population")
        
        log("Preparing next-gen")
        next_generation = self.fn_initialize_population(pop_size=pop_size,X_X=X_X)
        log("Preparied next-gen")
        
        fitness_history = []
        for generation_index in trange(generation_size):
            next_generation = self.genetic_iteration(
                                                PMS=PMS,
                                                indiv_size=indiv_size,
                                                indiv_flat_size=indiv_flat_size,
                                                population=next_generation, 
                                                selection_method=self.tournament_selection,
                                                number_of_crossover_section=number_of_crossover_section,
                                                mutation_prob=mutation_prob,
                                                mutation_amount=mutation_amount,
                                                fitness_history=fitness_history)
            if generation_index > trace_back_counter and fitness_history[generation_index]<=fitness_history[generation_index-trace_back_counter]:
                print("!!CONVERGED EXITING!!")
                break
#            if generation_index%100 == 0 and mutation_prob > 0.05:
#                mutation_prob = mutation_prob *.8
#            if generation_index%100 == 0 and mutation_amount > mutation_rate*indiv_flat_size*.05:
#                mutation_amount = mutation_amount - mutation_rate*indiv_flat_size*.05

        asyn_fitness_results=[POOL_.apply_async(self.fn_fitness, args=(PMS, popl)) for popl in np.split(next_generation, THREADS_)]
        fitness_results = sum([asyn_result.get() for asyn_result in asyn_fitness_results], [])
        population_with_fitness:List[Tuple[int,np.ndarray]] = sorted(fitness_results, key = lambda tup: tup[0], reverse=True)
        value = population_with_fitness[0][0]

        end_time = time()
        duration = end_time - start_time
        print(fitness_history)
        return SolutionResult(case, value, round(duration,4))

    def gen_greedy(self, indiv_size:Tuple, PMS:Parameters, C:int, D:int, H:int, U:int )->Iterable[np.ndarray]:
        for c in range(C):#tqdm(np.argsort(-MP.rp_c), desc="Campaigns Loop"):
            X = np.zeros(indiv_size, dtype=np.bool)
            for d in range(D):#trange(MP.D, desc=f"Days Loop"):            
                for h in range(H):#trange(H, desc=f"Channels Loop at Day-{d}, Campapaign-{c}"):
                    for u in range(U):#trange(U, desc=f"Users Loop On Campaign-{c}"):
                        X[c,u,h,d]=1
                        if not self.check(X, PMS, (c, u, h, d)):
                            X[c,u,h,d]=0
            log("Yielding For C: "+str(c))
            yield X

    def gen_greedy0(self, indiv_size:Tuple, PMS:Parameters, C:int, D:int, H:int, U:int )->Iterable[np.ndarray]:
        X = np.zeros(indiv_size, dtype=np.bool)
        for c in range(C):#tqdm(np.argsort(-MP.rp_c), desc="Campaigns Loop"):
            for d in range(D):#trange(MP.D, desc=f"Days Loop"):            
                for h in range(H):#trange(H, desc=f"Channels Loop at Day-{d}, Campapaign-{c}"):
                    for u in range(U):#trange(U, desc=f"Users Loop On Campaign-{c}"):
                        X[c,u,h,d]=1
                        if not self.check(X, PMS, (c, u, h, d)):
                            X[c,u,h,d]=0
        return X

    def fn_initialize_population(self, pop_size:int, X_X:List)->List[np.ndarray]:
        population = [X_X[i%len(X_X)] for i in range(pop_size)]
        return np.stack(population)

    def indiv_binary_rep(self, indiv):
        return "".join(map(lambda b: str(int(b)),indiv.tolist()))

    def genetic_iteration(self, PMS:Parameters, indiv_size:Tuple, indiv_flat_size:int, population:List, selection_method:Callable, number_of_crossover_section:int, mutation_prob:float, mutation_amount:int, fitness_history:List):
        pop_size = len(population)
        
        #uniq_pop = {self.indiv_binary_rep(i):i for i in population.reshape((pop_size, indiv_flat_size))}
        #fitness_dict = self.fn_fitness_dict(PMS, uniq_pop, indiv_size)
        #fitness_results = [(fitness_dict.get(self.indiv_binary_rep(i)),i) for i in population.reshape((pop_size, indiv_flat_size))]
        
        asyn_fitness_results=[POOL_.apply_async(self.fn_fitness, args=(PMS, popl)) for popl in np.split(population, THREADS_)]
        fitness_results = sum([asyn_result.get() for asyn_result in asyn_fitness_results], [])

        population_with_fitness:List[Tuple[int,np.ndarray]] = sorted(fitness_results, key = lambda tup: tup[0], reverse=True)
        #if(len(fitness_history)%10 == 0):
        #    print(population_with_fitness[0][0])
        fitness_history.append(population_with_fitness[0][0])
        elit1 = population_with_fitness[0][1]
        elit2 = population_with_fitness[1][1]

        #for i in range(pop_size):
        #    print(population_with_fitness[i][1].reshape(indiv_flat_size),"==>",population_with_fitness[i][0])
        #print("-------------###################-------------")

        parents = np.stack([a for a in self.select(selection_method, 2, population_with_fitness, (pop_size-2)//2)])
        next_generation = self.crossover(parents, indiv_flat_size, number_of_crossover_section).reshape(((pop_size-2,) + indiv_size))
        next_generation = self.mutation(next_generation, indiv_size, indiv_flat_size, pop_size, mutation_prob, mutation_amount)
        next_generation = np.append(next_generation, elit1)
        next_generation = np.append(next_generation, elit2)
        return next_generation.reshape(((pop_size,) + indiv_size))

    def fn_fitness(self, PMS:Parameters, population:np.ndarray)->List[Tuple]:
        return [(self.__fn_fitness_for_indiv(PMS, indiv), indiv) for indiv in population]

    def fn_fitness_dict(self, PMS:Parameters, population:Dict[str,np.ndarray], indiv_size)->Dict[str, int]:
        return {tag: self.__fn_fitness_for_indiv(PMS, indiv.reshape(indiv_size)) for tag,indiv in population.items()}

    def __fn_fitness_for_indiv(self, PMS:Parameters, indiv:np.ndarray)->int:
        if not self.X_check(PMS, indiv):
            return 0
        return self.objective_fn_no_net(PMS.rp_c, indiv)

    def __random_selection(self, size, pop_size)->Tuple[np.ndarray]:
        return tuple(np.random.randint(low=0, high=pop_size, size=size))

    def __spin_wheel(self, fitnesses:np.ndarray, sum_of_fitnesses:int)->int:
        pin_point = np.random.randint(low=0, high=sum_of_fitnesses)
        for index, fitness in enumerate(fitnesses):
            pin_point = pin_point + fitness
            if pin_point > sum_of_fitnesses:
                return index
        return index

    def roulettewheel_selection(self, population_with_fitness:List[Tuple], size:int)->Tuple[np.ndarray]:
        fitnesses = np.array([f for (f,i) in population_with_fitness])
        sum_of_fitnesses = fitnesses.sum()
        if sum_of_fitnesses == 0:
            return self.__random_selection(size=size, pop_size=len(population_with_fitness))
        else:
            return tuple(np.array([self.__spin_wheel(fitnesses, sum_of_fitnesses) for i in range(size)]))

    def __tournament_match(self, population_with_fitness:List[Tuple])->np.ndarray:
        rivals = self.__random_selection(size=2, pop_size=len(population_with_fitness))
        return rivals[np.argmax([population_with_fitness[i][0] for i in rivals])]

    def tournament_selection(self, population_with_fitness:List[Tuple], size):
        return tuple([self.__tournament_match(population_with_fitness) for i in range(size)])

    def select(self, selection_method, size:int, population_with_fitness:List[Tuple[int, np.ndarray]], count:int)->Iterable[np.ndarray]:
        for _ in range(count):
            yield np.array([population_with_fitness[i][1] for i in selection_method(population_with_fitness, size)])

    def __swap(self, sw):
        if sw ==0:
            return 1
        else:
            return 0

    def __crossover(self, indiv_flat_size:int, offspring:np.ndarray, parent_index:int, parents:np.ndarray, crossover_sections:List):
        for s,e in crossover_sections:
            if s!=e:
                offspring[0].reshape(indiv_flat_size)[s:e] = parents[parent_index,self.__swap(1)].reshape(indiv_flat_size)[s:e]
                offspring[1].reshape(indiv_flat_size)[s:e] = parents[parent_index,self.__swap(0)].reshape(indiv_flat_size)[s:e]
        log("parent-s : "+str(parents[parent_index,0].reshape(indiv_flat_size))+" "+str(parents[parent_index,1].reshape(indiv_flat_size)))
        log("child-s  : "+str(offspring[0].reshape(indiv_flat_size))+" "+str(offspring[1].reshape(indiv_flat_size)))
        log(">----------------------------------<")

    def crossover(self, parents:np.ndarray, indiv_flat_size:int, number_of_crossover_section:int):
        crossover_points = [p for p in np.sort(np.random.randint(low=1, high=indiv_flat_size, size=number_of_crossover_section))]
        crossover_sections = list(zip([0]+crossover_points,crossover_points+[indiv_flat_size]))
        offsprings = np.zeros(parents.shape, dtype='int')
        for index, offspring in enumerate(offsprings):
           self.__crossover(indiv_flat_size, offspring, index ,parents, crossover_sections)
        return offsprings

    def __mutate_allel(self, sw:np.ndarray):
        indices_one = sw == 1
        indices_zero = sw == 0
        sw[indices_one] = 0 # replacing 1s with 0s
        sw[indices_zero] = 1 # replacing 0s with 1s
        return sw

    def mutation(self, generation, indiv_size, indiv_flat_size, pop_size, mutation_prob, mutation_amount):
        mutation_indicies = np.random.random_sample(size=(pop_size-2))<mutation_prob
        mutants = generation[mutation_indicies]
        mutants = mutants.reshape((mutants.shape[0], indiv_flat_size))
        mutation_positions = np.random.randint(low=0, high=mutants.shape[1], 
                                            size=(mutants.shape[0], int(mutation_amount)))
        for mutant_index,mutation_position in enumerate(mutation_positions):
            mutants[mutant_index,mutation_position]=self.__mutate_allel(mutants[mutant_index,mutation_position])
        
        log("original : "+str(generation[mutation_indicies].reshape(mutants.shape[0], indiv_flat_size)))
        generation[mutation_indicies] = mutants.reshape(((mutants.shape[0],) + indiv_size))
        log("mutant   : "+str(generation[mutation_indicies].reshape(mutants.shape[0], indiv_flat_size)))
        return generation

if __name__ == '__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    cases = [
            Case({"C":2,"U":10,"H":1, "D":1, "I":1, "P":1}),#0
#            Case({"C":2,"U":100,"H":3, "D":7, "I":3, "P":3}),#1
#            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),#2
#            Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3}),#3
#            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),#4
#            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),#5
#            Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3}),#6
#            Case({"C":10,"U":3000,"H":3, "D":7, "I":3, "P":3}),#7
#            Case({"C":10,"U":4000,"H":3, "D":7, "I":3, "P":3}),#8
#            Case({"C":10,"U":5000,"H":3, "D":7, "I":3, "P":3}),#9
#            Case({"C":20,"U":10000,"H":3, "D":7, "I":3, "P":3}),#10
#            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),#11
#            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),#12
#            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3})
            ]
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
