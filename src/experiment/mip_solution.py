from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
import numpy as np
class MipSolution(Solution):
    def __init__(self):
        super().__init__("MIP")

    def mip_eligibility(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints(
            (X_cuhd[(c,u,h,d)] <= PMS.e_cu[c,u]
            for c in range(0,C)
            for u in range(0,U) 
            for h in range(0,H) 
            for d in range(0,D)))

    def mip_daily_communication(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]  
                    for c in range(0,C) 
                    for h in range(0,H)) <= PMS.k)
                for d in range(0,D)
                for u in range(0,U)))

    def mip_weekly_communication_rh(self, mdl, X_cuhd, PMS, C, U, H, D, f_d):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)] if d<f_d else PMS.s_cuhd[(c,u,h,d)]
                for d in range(0,D) 
                for c in range(0,C) 
                for h in range(0,H)) <= PMS.b)
            for u in range(0,U)))

    def mip_campaign_communication_rh(self, mdl, X_cuhd, PMS, C, U, H, D, f_d):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)] if d< f_d else PMS.s_cuhd[(c,u,h,d)]
                for d in range(0,D)
                for h in range(0,H)) <= PMS.l_c[c] )
            for c in range(0,C)
            for u in range(0,U)))

    def mip_weekly_quota_rh(self, mdl, X_cuhd, PMS, C, U, H, D, I, f_d):
        return mdl.add_constraints((
            (mdl.sum( (X_cuhd[(c,u,h,d)] if d < f_d else PMS.s_cuhd[(c,u,h,d)]) * PMS.q_ic[i,c]
                for d in range(0,D)
                for c in range(0,C)
                for h in range(0,H)) <= PMS.m_i[i])
            for u in range(0,U)
            for i in range(0,I)))

    def mip_daily_quota(self, mdl, X_cuhd, PMS, C, U, H, D, I):
        return mdl.add_constraints((
                (mdl.sum(X_cuhd[(c,u,h,d)]*PMS.q_ic[i,c]
                    for c in range(0,C) 
                    for h in range(0,H)) <= PMS.n_i[i])
                for u in range(0,U)
                for d in range(0,D)
                for i in range(0,I)))

    def mip_channel_capacity(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]
                for u in range(0,U) 
                for c in range(0,C)) <= PMS.t_hd[h,d])
            for h in range(0,H)
            for d in range(0,D)))


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
        PMS:Parameters = super().generate_parameters(case)
        mdl = Model(name='Campaign Optimization')
        #variables
        X_cuhd = {(c,u,h,d): mdl.binary_var(f"X_c:{c}_u:{u}_h:{h}_d:{d}")
#        X_cuhd = {(c,u,h,d): mdl.binary_var(f"X[{c},{u},{h},{d}]")
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
        eligibilitiy = self.mip_eligibility(mdl, X_cuhd, PMS, C, U, H, D)
        
        weekly_communication = [self.mip_weekly_communication_rh(mdl, X_cuhd, PMS, C, U, H, D, f_d) for f_d in range(1, D+1)]

        daily_communication = self.mip_daily_communication(mdl, X_cuhd, PMS, C, U, H, D)
        
        campaign_communication = [self.mip_campaign_communication_rh(mdl, X_cuhd, PMS, C, U, H, D, f_d) for f_d in range(1, D+1)]
        
        weekly_quota = [self.mip_weekly_quota_rh(mdl, X_cuhd, PMS, C, U, H, D, I, f_d) for f_d in range(1, D+1)]

        daily_quota = self.mip_daily_quota(mdl, X_cuhd, PMS, C, U, H, D, I)

        channel_capacity = self.mip_channel_capacity(mdl, X_cuhd, PMS, C, U, H, D)
        
        result = mdl.solve(log_output=False)

        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time
#        self.print_solution(result, PMS, C, D, H, U)
        self.validate(result, PMS, C, D, H, U)
        self.anti_validate(result, PMS,  C, D, H, U)

        return SolutionResult(case, value, round(duration,4))

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
    cases = [
            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":10,"U":3000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":10,"U":4000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":10,"U":5000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":10000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3})
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(MipSolution())
    for solution in solutions:
        print(solution)
#
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180.0, duration: 1.5269>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 341667.0, duration: 33.3006>
#<case: {'C': 15, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4836874.0, duration: 741.4099>


#<case: {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 13083.0, duration: 1.2028>
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180.0, duration: 1.5958>
#<case: {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 63096.0, duration: 3.2227>
#<case: {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 74664.0, duration: 15.1346>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 341667.0, duration: 33.3833>
#<case: {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 705516.0, duration: 78.8779>
#<case: {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1672838.0, duration: 147.7841>
#<case: {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1161046.0, duration: 177.4551>
#<case: {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1982340.0, duration: 367.6678>
#<case: {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 5507447.0, duration: 1381.5458>
#<case: {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4945924.0, duration: 1676.1964>
#<case: {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12318459.0, duration: 8917.5635>
#<case: {'C': 20, 'U': 40000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 26451198.0, duration: 8204.8334>
#<case: {'C': 20, 'U': 50000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 22565321.0, duration: 43721.7264>