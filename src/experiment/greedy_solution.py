from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from tqdm import trange
from tqdm import tqdm


class GreedySolution(Solution):
    def __init__(self):
        super().__init__("Greedy")

    def run(self, case:Case)->SolutionResult:
        from time import time
        import numpy as np
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        PMS:Parameters = super().generate_parameters(case)
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        for c in tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
            for d in trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in range(U):
                        X_cuhd[c,u,h,d]=1
                        if not self.check(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        end_time = time()
        value=self.objective_fn(PMS.rp_c, X_cuhd)
        duration = end_time - start_time
        return SolutionResult(case, value, round(duration,4))

if __name__ == '__main__':
    cases = [
            Case({"C":2,"U":10,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":2,"U":100,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":1000,"H":3, "D":7, "I":3, "P":3})            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GreedySolution())
    for solution in solutions:
        print(solution)