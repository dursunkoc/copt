from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np
from tqdm import trange
from tqdm import tqdm
from time import time

class MipSolutionValidation(Solution, MipCore):
    def __init__(self):
        super().__init__("MIP")


    def run(self, case:Case)->SolutionResult:

        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        PMS:Parameters = self.generate_parameters(case)
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')

        for c in tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
            for d in trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in range(U):
                        X_cuhd[c,u,h,d]=1
                        if not self.check(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        value=self.objective_fn(PMS.rp_c, X_cuhd)
        print("======================")
        print(f"Starting validation: value:{value}")

        mdl, _ = self.start_model(PMS, C, U, H, D, I, X_cuhd)
        
        start_time = time()
        result = mdl.solve(log_output=False)
#        for ky,_ in result.as_name_dict().items():
#            print(f'{tuple([int(i.split(":")[1]) for i in ky.split("_")[1:]])}')
        if result is not None and value != 0:
            v_value = result.objective_value
            if int(v_value) < int(value):
                raise ValueError(f"Value does not match {v_value}!= {value} !")    
        else:
            raise ValueError("No Valid Solution!")

        end_time = time()
        duration = end_time - start_time

        return SolutionResult(case, value, round(duration,4))


if __name__ == '__main__':
    cases = [
            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":2000,"H":3, "D":7, "I":3, "P":3}),
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(MipSolutionValidation())
    for solution in solutions:
        print(solution)
