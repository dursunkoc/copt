from datetime import datetime
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np
from time import time

class MipSolutionDNC(Solution, MipCore):
    def __init__(self):
        super().__init__("MIP")

    def runPh(self, case:Case, Xp_cuhd=None)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
        models = super().start_partial_models(True, PMS, C, U, H, D, I)
        total_resp = []
        for (k,v) in models.items():
            end_time = time()
            duration = end_time - start_time
            resp = SolutionResult(case, v, round(duration,4))
            total_resp.append(resp)
        total_value = sum([sr.value for sr in total_resp])
        total_duration = sum([sr.duration for sr in total_resp])
        total_resp = SolutionResult(case, total_value, total_duration)
        with open(f'result_mip_up_{datetime.now().strftime("%d-%m-%Y %H_%M_%S")}.txt','w') as f:
                f.write(repr(total_resp))
        return (0, total_resp)

    def create_var_for_greedy(self, solution, C, D, H, U):
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        if solution is not None and solution.as_name_dict() is not None:
            for ky,_ in solution.as_name_dict().items():
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
    solutions = expr.run_cases_with(MipSolutionDNC(), False)
    for solution in solutions:
        print(solution)
#    print("values:")
##    print(" ".join([str(v.value) for v in [c for solution in solutions for c in solution]]))
#    print(" ".join([str(v.value) for v in [solution[0] for solution in solutions]]))
#    print("durations:")
##    print(" ".join([str(v.duration) for v in [c for solution in solutions for c in solution]]))
#    print(" ".join([str(v.duration) for v in [solution[0] for solution in solutions]]))
#
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180.0, duration: 1.5269>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 341667.0, duration: 33.3006>
#<case: {'C': 15, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4836874.0, duration: 741.4099>