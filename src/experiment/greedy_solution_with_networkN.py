from numpy import random
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
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
        dd = {gd[0]:gd[1]*X_u[gd[0]] for gd in grph.degree}
        max_val = max(dd.items(), key=lambda x: x[1])[1]
        nns = [n for n in dd.items() if n[1] == max_val]
        nns_ = [n for n in nns if n[0] not in seen]
        if len(nns_)>0:
            nn = nns_[0]
        else:
            nn = nns[0]
        neighbors = grph.neighbors(nn[0])
        grph.remove_node(nn[0])
        seen |= set(list(neighbors))
        return (nn[0], seen)

    def sort_nodes_to_inc_span(self, grph, X_u):
        result=list()
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
                            if X_cuhd[c,u,h,d] == 0:
                                X_cuhd[c,u,h,d]=1
                                if not self.check(X_cuhd, PMS, (c, u, h, d)):
                                    X_cuhd[c,u,h,d]=0
        for h in range(PMS.H):
            for d in range(PMS.D):
                for c in range(PMS.C):
                    u_s = list(range(PMS.U))
                    random.shuffle(u_s)
                    for u in u_s:
                        consistent = self.check(X_cuhd, PMS, (c, u, h, d))
                        if X_cuhd[c,u,h,d] == 1 and not consistent:
                            X_cuhd[c,u,h,d]=0
        return X_cuhd

#    def local_search_iteration(self, X_cuhd, PMS, a_uv, prev_value):
#        current_solution = X_cuhd.copy()
#
#        iterations = 0
#        while iterations < PMS.U*PMS.C*PMS.H*PMS.D:
#            with ThreadPoolExecutor(max_workers=32) as executor:
#                # Generate neighborhood solutions concurrently
#                neighborhood_solutions = list(executor.map(self.improve_solution, self.generate_neighborhood(current_solution), [PMS] * len(current_solution)))
#            for neighborhood_solution, value in zip(neighborhood_solutions, [self.objective_fn(PMS.rp_c, solution, a_uv) for solution in neighborhood_solutions]):
#                if value > prev_value:
#                    current_solution = neighborhood_solution
#                    prev_value = value
##            for neighborhood_solution in self.generate_neighborhood(current_solution):
##                neighborhood_solution = self.improve_solution(neighborhood_solution, PMS)
##                value= self.objective_fn(PMS.rp_c, neighborhood_solution, a_uv)
##                if value > prev_value:
##                    current_solution = neighborhood_solution
##                    prev_value = value
#            iterations += 1
#
#        return current_solution
    
    def local_search_iteration(self, X_cuhd, PMS, a_uv, prev_value):
        current_solution = X_cuhd.copy()
        iterations = 0
        while iterations < 1000:
            # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=32) as executor:
                # Submit the entire loop body as a single task
                future = executor.submit(self.iteration_body, current_solution, PMS, a_uv, prev_value)
                # Get the result of the task (blocking until it's finished)
                current_solution, prev_value = future.result()
            iterations += 1
        return current_solution
    
    def iteration_body(self, current_solution, PMS, a_uv, prev_value):
        for neighborhood_solution in self.generate_neighborhood(current_solution):
            neighborhood_solution = self.improve_solution(neighborhood_solution, PMS)
            value= self.objective_fn(PMS.rp_c, neighborhood_solution, a_uv)
            if value > prev_value:
                current_solution = neighborhood_solution
                prev_value = value
        return current_solution, prev_value

    def generate_neighborhood(self, X_cuhd):
        neighborhood_solutions = []

        C, U, _, _ = X_cuhd.shape
        # Swap the assignments of the selected customers between the campaigns
        for _ in range(C):
            # Randomly select two different campaigns
            c1, c2 = np.random.choice(C, size=2, replace=False)
            # Randomly select two different customers within each campaign
            u1 = np.random.choice(U)
            u2 = np.random.choice(U)
            while u2 == u1:
                u2 = np.random.choice(U)
            neighbor_solution = np.copy(X_cuhd)
            neighbor_solution[c1, u1], neighbor_solution[c2, u2] = neighbor_solution[c2, u2], neighbor_solution[c1, u1]
            neighborhood_solutions.append(neighbor_solution)
        return neighborhood_solutions

    def validate(self, X_cuhd, PMS, C, D, H, U):
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        if X_cuhd[c,u,h,d]==1:
                            print(f"c:{c}_u:{u}_h:{h}_d:{d}={X_cuhd[c,u,h,d]}")
                        if X_cuhd[c,u,h,d]==1 and not self.check(X_cuhd, PMS, (c, u, h, d)):
                            raise RuntimeError(f'{(c, u, h, d)} does not consistent with previous values!')
        print("Solution is consistent with greedy from mip respect")

    def runPh(self, case:Case, Xp_cuhd):
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.

        nw_start_time = time()
        a_uv, _ = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        PMS.U = U
        PMS.H = H
        PMS.C = C
        PMS.D = D
        self.improve_solution(X_cuhd, PMS)
        value=self.objective_fn(PMS.rp_c, X_cuhd, a_uv)

        X_cuhd = self.local_search_iteration(X_cuhd, PMS, a_uv, value)
        value=self.objective_fn(PMS.rp_c, X_cuhd, a_uv)

        end_time = time()
        duration = end_time - start_time

        direct_msg = X_cuhd.sum()
        total_edges = a_uv.sum()

        result = (X_cuhd, SolutionResult(case, value, round(duration,4), {'direct_msg': direct_msg, 'total_edges':total_edges}))
        with open(f'result_gsn_N.txt','a') as f:
            f.write(repr(result[1]))
        return result
        
#####################


if __name__ == '__main__':
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GreedySolutionWithNet(seed=142, net_type='erdos', m=None, p=.003, drop_prob=.995), False)
    #solutions = expr.run_cases_with(GreedySolutionWithNet(seed=142, net_type='barabasi', m=3, p=None, drop_prob=.8), False)
    print(solutions)
    print("values:")
    print(" ".join([str(v.value) for v in [solution for solution in solutions]]))
    print("durations:")
    print(" ".join([str(v.duration) for v in [solution for solution in solutions]]))
