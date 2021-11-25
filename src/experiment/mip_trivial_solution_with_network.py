from experiment import Solution,  SolutionResult, Case, Experiment, TrivialParameters
from mip_core import MipCore
import numpy as np
from time import time
from network_generator import gen_network

class MipTrivialSolution(Solution, MipCore):
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("MIP")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type

    def runPh(self, case:Case, _)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        a_uv, _ = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        PMS:TrivialParameters = super().prepare_trivial(case, a_uv=a_uv, seed=self.seed)
        mdl, _ = super().start_trivial_model(True, PMS, C, U, H, D, I)

        result = mdl.solve(log_output=False)
        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time
        resp = (self.sol2NDArr(result, C, D, H, U), SolutionResult(case, value, round(duration,4)))
        
        print("========")
        print("Model:")
        print("========")
        print("Max:")
        print(f"\t{mdl._objective_expr}")
        print("subject to:")
        for c in mdl.iter_constraints():
            print(f"\t{c}")
        print("========")
        print("obj:", value)
        self.print_solution(result)
        print("========")
        del mdl
        del result
        del a_uv
        return resp

    def sol2NDArr(self, solution, C, D, H, U):
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        Y_cu2=np.zeros((C,U), dtype='int')
        if solution is not None and solution.as_name_dict() is not None:
            for ky,v in solution.as_name_dict().items():
                if(ky.startswith("X")):
                    exec(f'v{ky.lstrip("X")} = 1', {}, {'v':X_cuhd2})
                if(ky.startswith("Y")):
                    exec(f'v{ky.lstrip("Y")} = 1', {}, {'v':Y_cu2})
        return X_cuhd2, Y_cu2

    def print_solution(self, solution):
        print("Solution: ")
        if solution is not None and solution.as_name_dict() is not None:
            for ky,v in solution.as_name_dict().items():
                print(ky, "==>" ,v)


if __name__ == '__main__':
    cases = [
             Case({"C":1,"U":50,"H":1, "D":1, "I":3, "P":3}),#1
            ]
    expr = Experiment(cases)

    expr.run_cases_with(MipTrivialSolution(seed=1, net_type='barabasi', m=4, p=None, drop_prob=.9), False)