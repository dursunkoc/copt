from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np
from time import time
from network_generator import gen_network
from datetime import datetime

class MipSolutionWithNetwork(Solution, MipCore):
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("MIP With Network")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type


    def runPh(self, case:Case, Xp_cuhd=None)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        nw_start_time = time()
        a_uv, _ = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)
        mdl, _ = super().start_model(True, PMS, C, U, H, D, I)
        mdl.set_time_limit(180)

        result = mdl.solve(log_output=False)

        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time - nw_duration
        X = self.create_var_for_greedy(result, C, D, H, U)
        direct_msg = X.sum()
        total_edges = a_uv.sum()
        resp = (X, SolutionResult(case, value, round(duration,4), {'direct_msg': direct_msg, 'total_edges':total_edges}))
        del mdl
        del result
        with open(f'result_msn_{datetime.now().strftime("%d-%m-%Y %H_%M_%S")}.txt','w') as f:
            f.write(repr(resp[1]))
        return resp

    def create_var_for_greedy(self, solution, C, D, H, U):
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        if solution is not None and solution.as_name_dict() is not None:
            for ky,_ in solution.as_name_dict().items():
                if ky.startswith('X'):
                    exec(f'X_cuhd2{[int(i.split(":")[1]) for i in ky.split("_")[1:]]} = 1', {}, {'X_cuhd2':X_cuhd2})
        return X_cuhd2

    def print_solution(self, solution, PMS, C, D, H, U):
        X_cuhd2 = self.create_var_for_greedy(solution, C, D, H, U)
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        print(f"X_c:{c}_u:{u}_h:{h}_d:{d}={X_cuhd2[c,u,h,d]}")
                        print(f"s_c:{c}_u:{u}_h:{h}_d:{d}={PMS.s_cuhd[c,u,h,d]}")

    def validate(self, solution, PMS, C, D, H, U):
        X_cuhd2 = self.create_var_for_greedy(solution, C, D, H, U)
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        if X_cuhd2[c,u,h,d]==1 and not self.check(X_cuhd2, PMS, (c, u, h, d)):
                            raise RuntimeError(f'{(c, u, h, d)} does not consistent with previous values!')
        print("Solution is consistent with greedy from mip respect")

    def anti_validate(self, solution, PMS, C, D, H, U):
        X_cuhd2 = self.create_var_for_greedy(solution, C, D, H, U)
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for u in range(U):
                        if X_cuhd2[c,u,h,d]==0:
                            X_cuhd2[c,u,h,d]=1
                            if self.check(X_cuhd2, PMS, (c, u, h, d)):
                                raise RuntimeError(f'X_c:{c}_u:{u}_h:{h}_d:{d} should failed')
                            else:
                                X_cuhd2[c,u,h,d]=0
        print("Solution is consistent with greedy from greedy respect")


if __name__ == '__main__':
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(MipSolutionWithNetwork(seed=142, net_type='erdos', m=None, p=.003, drop_prob=.95), False)
#    solutions = expr.run_cases_with(MipSolutionWithNetwork(seed=142, net_type='barabasi', m=3, p=None, drop_prob=.8), False)
    print(solutions)
    print("values:")
#    print(" ".join([str(v.value) for v in [c for solution in solutions for c in solution]]))
    print(" ".join([str(v.value) for v in [solution for solution in solutions]]))
    print("durations:")
#    print(" ".join([str(v.duration) for v in [c for solution in solutions for c in solution]]))
    print(" ".join([str(v.duration) for v in [solution  for solution in solutions]]))
