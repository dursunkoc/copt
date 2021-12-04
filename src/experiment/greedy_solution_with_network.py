from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from camps_order_model import start_model as camps_order_model
from tqdm import trange
from tqdm import tqdm
from time import time
import numpy as np
from network_generator import gen_network
import networkx.algorithms.centrality as nxac
import base_network_opt as bno


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
        X_u = bno.solve_network_model(a_uv=a_uv, U=U, e_u=PMS.e_cu[0])
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
#        U_range = [self.max_degree(grph) for i in range(len(grph.nodes()))]
#        U_range = self.sort_nodes_to_inc_span(grph, X_u)
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)
        U_ranges = [ self.solve_and_sort(U, PMS, c) for c in range(C)]
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        mdl, Y = camps_order_model(C, D, I, PMS)
        camps_order_result = mdl.solve()

        camp_order = np.array(
            [
                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int')
        camp_prio = (camp_order) * PMS.rp_c
        for c in tqdm(np.lexsort((-camp_prio.sum(axis=0),-PMS.l_c,-PMS.rp_c))):#tqdm(np.argsort(-(PMS.rp_c)), desc="Campaigns Loop"):
#        for c in np.argsort(-PMS.rp_c): #tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
            for d in range(D):#trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in U_ranges[c]:
                        X_cuhd[c,u,h,d]=1
                        if not self.check(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        end_time = time()
        duration = end_time - start_time
        value=self.objective_fn(PMS.rp_c, X_cuhd, a_uv)
#        Y_cu = self.interaction_matrix(X_cuhd, a_uv)
        direct_msg = X_cuhd.sum()
        total_edges = a_uv.sum()
        return (X_cuhd, SolutionResult(case, value, round(duration,4), {'direct_msg': direct_msg, 'total_edges':total_edges}))

if __name__ == '__main__':
    cases = [
            Case({"C":2,"U":100,"H":3, "D":7, "I":3, "P":3}),#1
            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),#2
            Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3}),#3
            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),#4
            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),#5
            Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3}),#6
            Case({"C":10,"U":3000,"H":3, "D":7, "I":3, "P":3}),#7
            Case({"C":10,"U":4000,"H":3, "D":7, "I":3, "P":3}),#8
            Case({"C":10,"U":5000,"H":3, "D":7, "I":3, "P":3}),#9
#            Case({"C":20,"U":10000,"H":3, "D":7, "I":3, "P":3}),#10
#            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),#11
#            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),#12
#            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),#13
#            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3}),#14
#            Case({"C":30,"U":50000,"H":3, "D":7, "I":3, "P":3}),#16
#            Case({"C":30,"U":60000,"H":3, "D":7, "I":3, "P":3}),#17
            ]
    expr = Experiment(cases)
    #solutions = expr.run_cases_with(GreedySolutionWithNet(seed=142, net_type='erdos', m=None, p=.03, drop_prob=None), False)
    solutions = expr.run_cases_with(GreedySolutionWithNet(seed=142, net_type='barabasi', m=3, p=None, drop_prob=.8), False)
    print(solutions)
    print("values:")
    print(" ".join([str(v.value) for v in [solution for solution in solutions]]))
    print("durations:")
    print(" ".join([str(v.duration) for v in [solution for solution in solutions]]))


#values:
#[[8615]] [[13428]] [[57537]] [[67586]] [[502510]] [[1104453]] [[1721100]] [[1875604]] [[2659300]]
#durations:
#0.9889 3.2732 5.3542 34.7738 47.8813 112.5247 106.018 252.638 230.1216

#values:
#9546.0 18164.0 57717.0 69000.0 508000.0 1230000.0 1728000.0 1888000.0 2660000.0
#durations:
#4.3644 10.7578 32.8997 699.2396 1361.5039 5495.8364 11958.204 21312.3695 33117.8134

#########barabasi#########
#[{'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[6705]], 'duration': 1.0522, 'info': {'direct_msg': 228, 'total_edges': 122}}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[10034]], 'duration': 3.3425, 'info': {'direct_msg': 283, 'total_edges': 122}}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[42114]], 'duration': 5.0045, 'info': {'direct_msg': 1068, 'total_edges': 254}}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[35951]], 'duration': 37.377, 'info': {'direct_msg': 3468, 'total_edges': 1184}}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[266038]], 'duration': 62.678, 'info': {'direct_msg': 6374, 'total_edges': 1184}}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[389094]], 'duration': 182.6615, 'info': {'direct_msg': 9057, 'total_edges': 2360}}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[840730]], 'duration': 244.2337, 'info': {'direct_msg': 19811, 'total_edges': 3630}}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[841445]], 'duration': 488.8882, 'info': {'direct_msg': 18342, 'total_edges': 4878}}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': [[1444498]], 'duration': 605.6763, 'info': {'direct_msg': 31765, 'total_edges': 6056}}
#]
#values:
#[[6705]] [[10034]] [[42114]] [[35951]] [[266038]] [[389094]] [[840730]] [[841445]] [[1444498]]
#durations:
#1.0522 3.3425 5.0045 37.377 62.678 182.6615 244.2337 488.8882 605.6763