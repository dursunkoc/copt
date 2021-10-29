from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from camps_order_model import start_model as camps_order_model
from tqdm import trange
from tqdm import tqdm
from time import time
import numpy as np
from network_generator import gen_network

class GreedySolutionWithNet(Solution):
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("Greedy With Network")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type

    def runPh(self, case:Case, Xp_cuhd):
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of planning days.
        nw_start_time = time()
        print("Building Network", nw_start_time)
        a_uv, grph = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)

        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        U_range = list(map(lambda x: x[0], sorted(grph.degree, key=lambda x: x[1], reverse=True)))


        for c in np.argsort(-PMS.rp_c): #tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
            for d in range(D):#trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in U_range:
                        X_cuhd[c,u,h,d]=1
                        if not self.check(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        end_time = time()
        value=self.objective_fn(PMS.rp_c, X_cuhd, a_uv)
        duration = end_time - start_time
        return (X_cuhd, SolutionResult(case, value, round(duration,4)))

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
    solutions = expr.run_cases_with(GreedySolutionWithNet(seed=142, net_type='erdos', m=None, p=.03, drop_prob=None), False)
    print("values:")
    print(" ".join([str(v.value) for v in [solution[0] for solution in solutions]]))
    print("durations:")
    print(" ".join([str(v.duration) for v in [solution[0] for solution in solutions]]))


#values:
#[[8615]] [[13428]] [[57537]] [[67586]] [[502510]] [[1104453]] [[1721100]] [[1875604]] [[2659300]]
#durations:
#0.9889 3.2732 5.3542 34.7738 47.8813 112.5247 106.018 252.638 230.1216

#values:
#9546.0 18164.0 57717.0 69000.0 508000.0 1230000.0 1728000.0 1888000.0 2660000.0
#durations:
#4.3644 10.7578 32.8997 699.2396 1361.5039 5495.8364 11958.204 21312.3695 33117.8134