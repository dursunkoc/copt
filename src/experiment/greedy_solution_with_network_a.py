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

    def runPh(self, case:Case, Xp_cuhd):
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of planning days.
        a_uv, grph = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
#        U_range = list(map(lambda x: x[0], sorted(grph.degree, key=lambda x: x[1], reverse=True)))
        U_range = [self.max_degree(grph) for i in range(len(grph.nodes()))]
#        U_range = self.sort_nodes_to_inc_span(grph, X_u)
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)
#        print("Solving U ranges")
        typed_U_ranges = List()
#        U_ranges = [ self.solve_and_sort(U, PMS, c) for c in range(C)]
        [typed_U_ranges.append(U_range) for c in range(C)]
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
#        mdl, Y = camps_order_model(C, D, I, PMS)
#        camps_order_result = mdl.solve()

#        camp_order = np.array(
#            [
#                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
#            for d in range(D)]
#            , dtype='int')
#        camp_prio = (camp_order) * PMS.rp_c
        threshold = 0.10
        ee_cu = (PMS.e_cu.T * PMS.rp_c).T
#        for c in tqdm(np.lexsort((-camp_prio.sum(axis=0),-PMS.l_c,-PMS.rp_c))):#tqdm(np.argsort(-(PMS.rp_c)), desc="Campaigns Loop"):
        print("LOOPING:...")
        cstr.do_greedy_loop_for_net(X_cuhd, ee_cu, D, H, PMS.b, PMS.cuhd, PMS.e_cu, PMS.k, PMS.l_c, PMS.m_i, PMS.n_i, PMS.q_ic, PMS.s_cuhd, PMS.t_hd, typed_U_ranges)
#        for c in np.argsort(-ee_cu.sum(1)): #tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
#            for u in U_ranges[c]:
##        for ee in np.argsort(-ee_cu, axis=None):
##            c = math.floor(ee / U)
##            u = ee % U
#                for d in range(D):#trange(D, desc=f"Days Loop for campaign-{c}"):
#                    for h in range(H):
#                        if(ee_cu[c][u]>0):
#                            X_cuhd[c,u,h,d]=1
#                            if not self.check(X_cuhd, PMS, (c, u, h, d)):
#                                X_cuhd[c,u,h,d]=0
        end_time = time()
        duration = end_time - start_time
        value=self.objective_fn(PMS.rp_c, X_cuhd, a_uv)
#        Y_cu = self.interaction_matrix(X_cuhd, a_uv)
        direct_msg = X_cuhd.sum()
        total_edges = a_uv.sum()
        result = (X_cuhd, SolutionResult(case, value, round(duration,4), {'direct_msg': direct_msg, 'total_edges':total_edges}))
        with open(f'result_gsn_A_less_995.txt','a') as f:
            f.write(repr(result[1]))
        return result

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
