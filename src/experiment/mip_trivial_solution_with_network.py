from experiment import Solution,  SolutionResult, Case, Experiment, TrivialParameters
from mip_core import MipCore
import numpy as np
from time import time

class MipTrivialSolution(Solution, MipCore):
    def __init__(self):
        super().__init__("MIP")

    def runPh(self, case:Case, _)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        PMS:TrivialParameters = super().prepare_trivial(case, network=True)
        mdl, _ = super().start_trivial_model(True, PMS, C, U, H, D, I)

        result = mdl.solve(log_output=False)
        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time
        resp = (self.sol2NDArr(result, C, D, H, U), SolutionResult(case, value, round(duration,4)))
        print(resp)
        self.print_solution(result, C, D, H, U)
        print("========")
        print("Model:")
        print("========")
        print("Max:")
        print(f"\t{mdl._objective_expr}")
        print("subject to:")
        for c in mdl.iter_constraints():
            print(f"\t{c}")
        del mdl
        del result
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

    def print_solution(self, solution, C, D, H, U):
        print("Solution: ")
        if solution is not None and solution.as_name_dict() is not None:
            for ky,v in solution.as_name_dict().items():
                print(ky, "==>" ,v)


if __name__ == '__main__':
    cases = [
             Case({"C":2,"U":10,"H":1, "D":1, "I":3, "P":3}),#1
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(MipTrivialSolution(), False)
    for solution in solutions:
        print(solution)
