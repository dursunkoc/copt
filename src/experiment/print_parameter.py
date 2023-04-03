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
        print("""
\\begin{lstlisting}[language=Python]
Instance - """ +str(case.id())+ """
=================================
""")
        print(PMS)
        print("\end{lstlisting}")
        total_resp = SolutionResult(case, 0, 0)
        return (0, total_resp)


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