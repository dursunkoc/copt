from datetime import datetime
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np

class LpRelSolution(Solution, MipCore):
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

    def runPh(self, case:Case, Xp_cuhd=None)->SolutionResult:
        from time import time
        from docplex.mp.model import Model
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
        print("Staring Model")
        mdl, _ = super().start_model(False, PMS, C, U, H, D, I)
        print("Solving Model")
        result = mdl.solve(log_output=False)
        if hasattr(result, 'objective_value'):
            v_non_integral = result.objective_value
        else:
            v_non_integral = 0
        X_non_integral = [tuple(int(i.split(":")[1]) for i in ky.split("_")[1:]) for ky,val in result.as_name_dict().items() if val!=1]
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        print("Starting greedy")
        for ky,_ in result.as_name_dict().items():
            exec(f'X_cuhd2{[int(i.split(":")[1]) for i in ky.split("_")[1:]]} = 1', {}, {'X_cuhd2':X_cuhd2})
        v_non_opt = self.objective_fn_no_net(PMS.rp_c, X_cuhd2)
        print("Starting rounding with greedy")
        self.round_with_greedy(X_non_integral, X_cuhd2, PMS, D, H, U)
        if(v_non_opt!=v_non_integral):
            print(f"Non-optimistic & integral solution: {v_non_opt}")
            print(f"Non-integral solution: {v_non_integral}")
            print(f"Optimistic solution: {self.objective_fn_no_net(PMS.rp_c, X_cuhd2)}")
        print("Completed.")
        value = self.objective_fn_no_net(PMS.rp_c, X_cuhd2)
        end_time = time()
        duration = end_time - start_time
        result = (X_cuhd2, SolutionResult(case, value, round(duration,4)))
        with open(f'result_lpx_{datetime.now().strftime("%d-%m-%Y %H_%M_%S")}.txt','w') as f:
            f.write(repr(result[1]))
        return result

if __name__ == '__main__':
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(LpRelSolution(), False)
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

#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 15141, 'duration': 1.7109}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 15141, 'duration': 2.1263}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 18464, 'duration': 4.1774}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 18464, 'duration': 5.3171}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 63096, 'duration': 7.0218}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 63096, 'duration': 9.0927}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 87747, 'duration': 40.7323}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 87747, 'duration': 52.7989}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 329480, 'duration': 75.836}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 329480, 'duration': 97.647}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 776907, 'duration': 216.9998}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 765885, 'duration': 264.7093}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1673454, 'duration': 239.6793}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 1673454, 'duration': 306.4116}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1660425, 'duration': 355.51}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 1660200, 'duration': 440.5183}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 2050264, 'duration': 408.6972}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 2048470, 'duration': 524.7129}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 5800794, 'duration': 1723.3974}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 5797365, 'duration': 2735.4943}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 4800274, 'duration': 4375.9088}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 4799942, 'duration': 6280.2394}
#)


#THe Last.
#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 15141, 'duration': 1.7868}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 15141, 'duration': 2.6231}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 18464, 'duration': 4.9617}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 18464, 'duration': 6.2578}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 63096, 'duration': 7.6522}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 63096, 'duration': 10.8541}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 87747, 'duration': 45.3119}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 87747, 'duration': 62.1801}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 329480, 'duration': 76.9775}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 329480, 'duration': 96.3895}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 776907, 'duration': 224.1238}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 776841, 'duration': 272.2642}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1673454, 'duration': 241.0166}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1673454, 'duration': 309.8275}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1660425, 'duration': 354.5584}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1660599, 'duration': 439.5952}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 2050264, 'duration': 403.1607}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 2049298, 'duration': 524.9602}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5800794, 'duration': 1840.7676}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 5798946, 'duration': 2708.7463}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 4800274, 'duration': 4411.3576}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 4800004, 'duration': 6041.0063}
#)


#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 15141, 'duration': 1.625}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 15141, 'duration': 1.875}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 18464, 'duration': 3.5938}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 18464, 'duration': 4.4062}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 63096, 'duration': 5.7656}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 63096, 'duration': 7.5156}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 87747, 'duration': 35.3829}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 87747, 'duration': 44.4886}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 329480, 'duration': 64.0469}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 329480, 'duration': 80.75}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 776907, 'duration': 188.3906}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 776841, 'duration': 224.5469}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1673454, 'duration': 216.1094}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1673454, 'duration': 271.3281}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1660436, 'duration': 311.2656}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1660217, 'duration': 387.5}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 2050264, 'duration': 365.0469}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 2049919, 'duration': 440.0469}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5800794, 'duration': 1575.6562}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 5800290, 'duration': 2883.3281}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 4800274, 'duration': 2794.1891}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 4800004, 'duration': 3410.9779}
#)