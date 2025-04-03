from numpy import random
from experiment import Solution, SolutionResult, Case, Experiment, Parameters
from camps_order_model import start_model as camps_order_model
from tqdm import trange
from tqdm import tqdm
from time import time
import math
import numpy as np
from network_generator import gen_network
import networkx.algorithms.centrality as nxac
import base_network_opt as bno
from datetime import datetime
import co_constraints as cstr
from numba.typed import List
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import copy 


class GreedySolutionWithNet(Solution):
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("Greedy With Network")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type

    def max_degree(self, grph):
        nn = max(grph.degree, key=lambda x: x[1])
        grph.remove_node(nn[0])
        return nn[0]

    def max_degree_b(self, grph, seen, X_u):
        dd = {gd[0]: gd[1] * X_u[gd[0]] for gd in grph.degree}
        max_val = max(dd.items(), key=lambda x: x[1])[1]
        nns = [n for n in dd.items() if n[1] == max_val]
        nns_ = [n for n in nns if n[0] not in seen]
        if len(nns_) > 0:
            nn = nns_[0]
        else:
            nn = nns[0]
        neighbors = grph.neighbors(nn[0])
        grph.remove_node(nn[0])
        seen |= set(list(neighbors))
        return (nn[0], seen)

    def sort_nodes_to_inc_span(self, grph, X_u):
        result = list()
        seen = set()
        for _ in range(len(grph.degree)):
            elem, seen = self.max_degree_b(grph, seen, X_u)
            result.append(elem)
        return result

    def solve_and_sort(self, U, PMS, c):
        a_uv, grph = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        print(f"Random Net Generated & Solving for campaign {c}")
        X_u = bno.solve_network_model(a_uv=a_uv, U=U, e_u=PMS.e_cu[c])
        return self.sort_nodes_to_inc_span(grph, X_u)

    def improve_solution(self, X_cuhd, PMS):
        if self.X_check(PMS, X_cuhd):
            for h in range(PMS.H):
                for d in range(PMS.D):
                    for c in range(PMS.C):
                        u_s = list(range(PMS.U))
                        random.shuffle(u_s)
                        for u in u_s:
                            if X_cuhd[c, u, h, d] == 0:
                                X_cuhd[c, u, h, d] = 1
                                if not self.check(X_cuhd, PMS, (c, u, h, d)):
                                    X_cuhd[c, u, h, d] = 0
        for h in range(PMS.H):
            for d in range(PMS.D):
                for c in range(PMS.C):
                    u_s = list(range(PMS.U))
                    random.shuffle(u_s)
                    for u in u_s:
                        consistent = self.check(X_cuhd, PMS, (c, u, h, d))
                        if X_cuhd[c, u, h, d] == 1 and not consistent:
                            X_cuhd[c, u, h, d] = 0
        return X_cuhd

    def local_search_iteration(self, X_cuhd, PMS, a_uv, prev_value):
        current_solution = X_cuhd.copy()
        iterations = 0
        while iterations < 1000:
            with ThreadPoolExecutor(max_workers=32) as executor:
                future = executor.submit(self.iteration_body, current_solution, PMS, a_uv, prev_value)
                current_solution, prev_value = future.result()
            iterations += 1
        return current_solution

    def iteration_body(self, current_solution, PMS, a_uv, prev_value):
        for neighborhood_solution in self.generate_neighborhood(current_solution, PMS):
            neighborhood_solution = self.improve_solution(neighborhood_solution, PMS)
            value = self.objective_fn(PMS.rp_c, neighborhood_solution, a_uv)
            if value > prev_value:
                current_solution = neighborhood_solution
                prev_value = value
        return current_solution, prev_value

    def generate_neighborhood(self, X_cuhd, PMS, max_neighbors=10):
        neighbor_solutions = []
        num_neighbors = min(max_neighbors, X_cuhd.size // 2)
        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4), thread_name_prefix='TPE-NEIGHBOR-GEN-') as executor:
            futures = []
            while len(neighbor_solutions) < num_neighbors:
                future = executor.submit(self.generate_a_neighbor, X_cuhd, PMS)
                futures.append(future)

            for future in futures:
                neighbor = future.result()
                neighbor_solutions.append(neighbor)
        return neighbor_solutions

    def generate_a_neighbor(self, X, PMS, mod_rate=0.1):
        neighbor = X.copy()
        neighbor = neighbor.flatten()
        size = neighbor.size
        swap_points = np.random.choice(np.arange(size), size=int(size * mod_rate), replace=False)
        neighbor[swap_points] = 1 - neighbor[swap_points]
        neighbor = neighbor.reshape(X.shape)
        neighbor = self.improve_solution(neighbor, PMS)
        return neighbor

    def validate(self, X_cuhd, PMS, C, D, H, U):
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        if X_cuhd[c, u, h, d] == 1:
                            print(f"c:{c}_u:{u}_h:{h}_d:{d}={X_cuhd[c, u, h, d]}")
                        if X_cuhd[c, u, h, d] == 1 and not self.check(X_cuhd, PMS, (c, u, h, d)):
                            raise RuntimeError(f'{(c, u, h, d)} does not consistent with previous values!')
        print("Solution is consistent with greedy from mip respect")

    def simulated_annealing(self, X_cuhd, PMS, a_uv, prev_value, initial_temp=100, final_temperature=1e-5, max_iterations=1000):
        """HANSA adapted for this class."""
        S = X_cuhd.copy()
        S_best = S.copy()
        T = initial_temp
        NS_perf = {1: 1.0, 2: 1.0, 3: 1.0}
        TL = []
        SC = 0
        eta_stag = 50
        eta_severe = 100
        eta_recent = 10
        base_tenure = 7
        alpha_perf = 0.7
        beta_perf = 0.3
        r_heat = 1.5
        epsilon = 0.001
        eta_early = 5
        recent_improvements = 0
        temperature_levels = 0

        while T > final_temperature and temperature_levels < max_iterations:
            for i in range(max_iterations):
                neighborhood_probs = np.array(list(NS_perf.values())) / sum(NS_perf.values())
                neighborhood_choice = np.random.choice(list(NS_perf.keys()), p=neighborhood_probs)
                if neighborhood_choice == 1:
                    S_prime = self.generate_neighbor_n1(S.copy(), PMS)
                elif neighborhood_choice == 2:
                    S_prime = self.generate_neighbor_n2(S.copy(), PMS)
                else:
                    S_prime = self.generate_neighbor_n3(S.copy(), PMS)

                if tuple(map(tuple, S_prime.flatten())) in TL:
                    continue

                S_prime, f_prime = self.solve_network_influence_subproblem(S_prime, PMS, a_uv)
                f_S, _ = self.solve_network_influence_subproblem(S, PMS, a_uv)
                delta_f = f_prime - f_S

                alpha_i = 1.0
                p = np.exp(-delta_f / (T * alpha_i))
                r = random.random()

                if delta_f >= 0 or r < p:
                    S = S_prime.copy()
                    TL.append(tuple(map(tuple, S.flatten())))
                    tenure = base_tenure + SC // 10
                    if len(TL) > tenure:
                        TL.pop(0)

                    if f_prime > self.objective_fn(PMS.rp_c, S_best, a_uv):
                        S_best = S.copy()
                        SC = 0
                        recent_improvements += 1
                    else:
                        SC += 1
                        recent_improvements = 0

                    NS_perf[neighborhood_choice] = (alpha_perf * (1 if delta_f >= 0 else 0) + beta_perf * abs(delta_f)) / 1
                else:
                    SC += 1
                    recent_improvements = 0

                if SC > eta_stag:
                    for _ in range(5):
                        S = self.generate_neighbor_n2(S.copy(), PMS)
                    if SC > eta_severe:
                        T *= r_heat
                        SC = 0
            temperature_levels += 1
            if recent_improvements > eta_recent:
                T *= 0.98
            elif SC < eta_stag / 2:
                T *= 0.95
            else:
                T *= 0.90

            if abs(self.objective_fn(PMS.rp_c, S_best, a_uv) - self.objective_fn(PMS.rp_c, S, a_uv)) < epsilon and temperature_levels > eta_early:
                break
        return S_best

    def solve_network_influence_subproblem(self, solution, PMS, a_uv):
        """Placeholder for CPLEX integration."""
        objective_value = self.objective_fn(PMS.rp_c, solution, a_uv)
        return solution, objective_value

    def generate_neighbor_n1(self, solution, PMS):
        C, U, H, D = solution.shape
        new_solution = copy.deepcopy(solution)
        c1, c2 = random.sample(range(C), 2)
        u, h, d = random.choice([(u, h, d) for u in range(U) for h in range(H) for d in range(D)])
        new_solution[c1, u, h, d], new_solution[c2, u, h, d] = new_solution[c2, u, h, d], new_solution[c1, u, h, d]
        return new_solution

    def generate_neighbor_n2(self, solution, PMS):
        C, U, H, D = solution.shape
        new_solution = copy.deepcopy(solution)
        c, u = random.choice([(c, u) for c in range(C) for u in range(U)])
        h1, d1 = random.choice([(h, d) for h in range(H) for d in range(D) if solution[c, u, h, d] == 1])
        h2, d2 = random.choice([(h, d) for h in range(H) for d in range(D) if (h, d) != (h1, d1)])
        new_solution[c, u, h1, d1] = 0
        new_solution[c, u, h2, d2] = 1
        return new_solution

    def generate_neighbor_n3(self, solution, PMS):
        C, U, H, D = solution.shape
        new_solution = copy.deepcopy(solution)

        # 1 boyutlu liste oluşturma (doğru şekilde)
        indices = []
        for c in range(C):
            for u in range(U):
                for h in range(H):
                    for d in range(D):
                        indices.append((c, u, h, d)) # tuple'ları 1 boyutlu listeye ekliyoruz.

        if indices: # Check if indices is not empty.
            c, u, h, d = random.choice(indices)
            new_solution[c, u, h, d] = 1 - new_solution[c, u, h, d]
        return new_solution

    def runPh(self, case: Case, Xp_cuhd):
        import random
        start_time = time()
        C = case.arguments["C"]
        U = case.arguments["U"]
        H = case.arguments["H"]
        D = case.arguments["D"]

        nw_start_time = time()
        a_uv, _ = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        PMS: Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)
        X_cuhd = np.zeros((C, U, H, D), dtype='int')
        PMS.U = U
        PMS.H = H
        PMS.C = C
        PMS.D = D
        self.improve_solution(X_cuhd, PMS)
        value = self.objective_fn(PMS.rp_c, X_cuhd, a_uv)

        X_cuhd = self.simulated_annealing(X_cuhd, PMS, a_uv, value)
        value = self.objective_fn(PMS.rp_c, X_cuhd, a_uv)

        end_time = time()
        duration = end_time - start_time

        direct_msg = X_cuhd.sum()
        total_edges = a_uv.sum()

        result = (X_cuhd, SolutionResult(case, value, round(duration, 4), {'direct_msg': direct_msg, 'total_edges': total_edges}))
        with open(f'result_gsn_HANSA.txt', 'a') as f:
            f.write(repr(result[1]))
        return result


if __name__ == '__main__':
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GreedySolutionWithNet(seed=142, net_type='erdos', m=None, p=.003, drop_prob=.995), False)
    print(solutions)
    print("values:")
    print(" ".join([str(v.value) for v in [solution for solution in solutions]]))
    print("durations:")
    print(" ".join([str(v.duration) for v in [solution for solution in solutions]]))