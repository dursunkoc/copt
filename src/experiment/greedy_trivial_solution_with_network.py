from experiment import Solution,  SolutionResult, Case, Experiment, TrivialParameters
import numpy as np
from time import time
from network_generator import gen_network

class GreedyTrivialSolution(Solution):
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("Greedy Trivial")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type

    def max_degree(self, grph):
        nn = max(grph.degree, key=lambda x: x[1])
        grph.remove_node(nn[0])
        return nn[0]

    def max_degree_b(self, grph, seen):
        max_val = max(grph.degree, key=lambda x: x[1])[1]
        nns = [n for n in grph.degree if n[1] == max_val]
        nns_ = [n for n in nns if n[0] not in seen]
        if len(nns_)>0:
            nn = nns_[0]
        else:
            nn = nns[0]
        neighbors = grph.neighbors(nn[0])
        grph.remove_node(nn[0])
        seen |= set(list(neighbors))
        return (nn[0], seen)

    def sort_nodes_to_inc_span(self, grph):
        result=list()
        seen = set()
        for _ in range(len(grph.degree)):
            elem, seen = self.max_degree_b(grph, seen)
            result.append(elem)
        return result

    def runPh(self, case:Case, _)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        nw_start_time = time()
        print("Building Network", nw_start_time)
        a_uv, grph = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        PMS:TrivialParameters = super().prepare_trivial(case, a_uv=a_uv, seed=self.seed)


        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        #U_range = list(map(lambda x: x[0], sorted(grph.degree, key=lambda x: x[1], reverse=True)))
#        U_range = [self.max_degree(grph) for i in range(len(grph.nodes()))]
        U_range = self.sort_nodes_to_inc_span(grph)
        print(U_range)

        for c in np.argsort(-PMS.rp_c): #tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
            for d in range(D):#trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in U_range:
                        X_cuhd[c,u,h,d]=1
                        if not self.check_trivial(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        end_time = time()

        for c in range(C): 
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        if X_cuhd[c,u,h,d] == 1:
                            print(f"X_cuhd[{c},{u},{h},{d}]==>", X_cuhd[c,u,h,d])
        
        Y_cu = self.interaction_matrix(X_cuhd, a_uv)
        for c in range(C): 
            for u in range(U):
                if Y_cu[c, u] == 1:
                    print(f"Y_cu[{c},{u}]==>", Y_cu[c, u])

        value=self.objective_fn(PMS.rp_c, X_cuhd, a_uv)
        duration = end_time - start_time
        return (X_cuhd, SolutionResult(case, value, round(duration,4)))

if __name__ == '__main__':
    cases = [
             Case({"C":1,"U":50,"H":1, "D":1, "I":3, "P":3}),#1
            ]
    expr = Experiment(cases)

    solutions = expr.run_cases_with(GreedyTrivialSolution(seed=1, net_type='barabasi', m=4, p=None, drop_prob=.9), False)
    print("values:")
    print(" ".join([str(v.value) for v in [solution for solution in solutions]]))
    print("durations:")
    print(" ".join([str(v.duration) for v in [solution for solution in solutions]]))


