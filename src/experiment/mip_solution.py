from experiment import Solution,  SolutionResult, Case, Experiment, Parameters

class MipSolution(Solution):
    def __init__(self):
        super().__init__("MIP")

    def run(self, case:Case)->SolutionResult:
        from time import time
        import numpy as np
        from docplex.mp.model import Model
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        P = case.arguments["P"]  # number of priority categories.
        PMS:Parameters = super().generate_parameters(case)
        mdl = Model(name='Campaign Optimization')
        #variables
        X_cuhd = {(c,u,h,d): mdl.binary_var(f"X_c:{c}_u:{u}_h:{h}_d:{d}")
            for c in range(0,C)
            for u in range(0,U) 
            for h in range(0,H)
            for d in range(0,D)}
        #objectivefunction
        maximize = mdl.maximize(mdl.sum([X_cuhd[(c,u,h,d)] * PMS.rp_c[c]
                  for c in range(0,C)
                  for u in range(0,U) 
                  for h in range(0,H) 
                  for d in range(0,D)]))
        #constraints
        eligibilitiy = mdl.add_constraints(
            (X_cuhd[(c,u,h,d)] <= PMS.e_cu[c,u]
            for c in range(0,C)
            for u in range(0,U) 
            for h in range(0,H) 
            for d in range(0,D)))
        
        weekly_communication = mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)] 
               for d in range(0,D) 
               for c in range(0,C) 
               for h in range(0,H)) <= PMS.b)
            for u in range(0,U)))

        daily_communication = mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]  
                    for c in range(0,C) 
                    for h in range(0,H)) <= PMS.k)
                for d in range(0,D)
                for u in range(0,U)))
        
        campaign_communication = mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]  
                for h in range(0,H) 
                for d in range(0,D)) <= PMS.l_c[c] )
            for c in range(0,C)
            for u in range(0,U)))

        weekly_quota = mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]*PMS.q_ic[i,c]
                for c in range(0,C)
                for h in range(0,H) 
                for d in range(0,D)) <= PMS.m_i[i])
            for u in range(0,U)
            for i in range(0,I)))
        
        result = mdl.solve(log_output=False)
        value = result.objective_value
        end_time = time()
        duration = end_time - start_time
        return SolutionResult(case, value, round(duration,4))

if __name__ == '__main__':
    cases = [
            Case({"C":2,"U":10,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":2,"U":11,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":2,"U":12,"H":3, "D":7, "I":3, "P":3})
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(MipSolution())
    for solution in solutions:
        print(solution)