from datetime import datetime
from pickle import FALSE
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from camps_order_model import start_model as camps_order_model
from tqdm import trange
from tqdm import tqdm
from time import time
import numpy as np
import co_constraints as cstr

class GreedySolution(Solution):
    def __init__(self, use_campaign_expectation=False):
        super().__init__("Greedy")
        self.use_campaign_expectation=use_campaign_expectation

    def runPh(self, case:Case, Xp_cuhd):
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of planning days.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        mdl, Y = camps_order_model(C, D, I, PMS)
        camps_order_result = mdl.solve()
        print(camps_order_result.objective_value)
        value = camps_order_result.objective_value
        result = (X_cuhd, SolutionResult(case, value, 0))
        with open(f'result{datetime.now().strftime("%d-%m-%Y %H_%M_%S")}.txt','w') as f:
            f.write(repr(result[1]))
        return result


if __name__ == '__main__':
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GreedySolution(True), False)
    for solution in solutions:
        print(solution)