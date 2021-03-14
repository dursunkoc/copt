from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
import numpy as np

class LpRelSolution(Solution):
    def __init__(self):
        super().__init__("LP-Relaxation")

    def round_with_greedy(self, X_non_integral, X, PMS, D, H, U):
        from tqdm import tqdm
        from tqdm import trange
        for c in np.argsort(PMS.rp_c):#tqdm(np.argsort(PMS.rp_c), desc="Campaigns Loop"):
            for d in range(D):#trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in range(U):
                        if (c,u,h,d) in X_non_integral and X[c,u,h,d]==1 and not self.check(X, PMS, (c, u, h, d)):
                            X[c,u,h,d]=0

    def run(self, case:Case)->SolutionResult:
        from time import time
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
        X_cuhd = {(c,u,h,d): mdl.continuous_var(lb=0, ub=1, name=f"X_c:{c}_u:{u}_h:{h}_d:{d}")
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
        v_non_integral = result.objective_value
        X_non_integral = [tuple(int(i.split(":")[1]) for i in ky.split("_")[1:]) for ky,val in result.as_name_dict().items() if val!=1]
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        for ky,_ in result.as_name_dict().items():
            exec(f'X_cuhd2{[int(i.split(":")[1]) for i in ky.split("_")[1:]]} = 1', {}, {'X_cuhd2':X_cuhd2})
        v_non_opt = self.objective_fn(PMS.rp_c, X_cuhd2)
        self.round_with_greedy(X_non_integral, X_cuhd2, PMS, D, H, U)
        if(v_non_opt!=v_non_integral):
            print(f"Non-optimistic & integral solution: {v_non_opt}")
            print(f"Non-integral solution: {v_non_integral}")
            print(f"Optimistic solution: {self.objective_fn(PMS.rp_c, X_cuhd2)}")
        value = self.objective_fn(PMS.rp_c, X_cuhd2)
        end_time = time()
        duration = end_time - start_time
        return SolutionResult(case, value, round(duration,4))

if __name__ == '__main__':
    cases = [
            Case({"C":2,"U":100,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":10,"U":3000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":10,"U":4000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":10,"U":5000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":10000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),
            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3})
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(LpRelSolution())
    for solution in solutions:
        print(solution)

#
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180, duration: 1.5658>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 341667, duration: 29.2164>
#<case: {'C': 15, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4836874, duration: 445.1445>

#<case: {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 13083, duration: 0.8805>
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180, duration: 1.5107>
#<case: {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 63096, duration: 2.7442>
#<case: {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 74664, duration: 14.9263>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 341667, duration: 28.4282>
#<case: {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 697860, duration: 109.7368>
#<case: {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1672442, duration: 92.7799>
#<case: {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1161046, duration: 122.5116>
#<case: {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1982340, duration: 148.8915>
#<case: {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 5506153, duration: 632.4157>
#<case: {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4945718, duration: 1440.8951>
#<case: {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12309790, duration: 4512.4216>
#<case: {'C': 20, 'U': 40000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 26450884, duration: 4919.4901>
#<case: {'C': 20, 'U': 50000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 22563024, duration: 8466.245>