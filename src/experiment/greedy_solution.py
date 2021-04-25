from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from tqdm import trange
from tqdm import tqdm
from time import time
import numpy as np
""
class GreedySolution(Solution):
    def __init__(self):
        super().__init__("Greedy")

    def runPh(self, case:Case, Xp_cuhd):
        start_time = time()
        #seed randomization
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        for c in tqdm(np.argsort(-PMS.rp_c), desc="Campaigns Loop"):
            for d in range(D):#trange(D, desc=f"Days Loop for campaign-{c}"):
                for h in range(H):
                    for u in range(U):
                        X_cuhd[c,u,h,d]=1
                        if not self.check(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        end_time = time()
        value=self.objective_fn(PMS.rp_c, X_cuhd)
        duration = end_time - start_time
        return (X_cuhd, SolutionResult(case, value, round(duration,4)))

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
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GreedySolution())
    for solution in solutions:
        print(solution)
#
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180, duration: 0.3595>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 340699, duration: 6.0798>
#<case: {'C': 15, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4827516, duration: 210.3571>

#<case: {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 13083, duration: 0.1519>
#<case: {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 12180, duration: 0.3341>
#<case: {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 63096, duration: 0.5835>
#<case: {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 74664, duration: 3.2638>
#<case: {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 340699, duration: 6.2448>
#<case: {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 658443, duration: 13.1715>
#<case: {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1672590, duration: 14.8908>
#<case: {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1158598, duration: 29.8888>
#<case: {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 1979939, duration: 34.4059>
#<case: {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 5287034, duration: 344.8326>
#<case: {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 4941547, duration: 911.2202>
#<case: {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 10256858, duration: 1925.4408>
#<case: {'C': 20, 'U': 40000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 26444568, duration: 4807.0272>
#<case: {'C': 20, 'U': 50000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, value: 22559756, duration: 5700.4705>

#    values = [solution.value for solution in solutions]
#    durations = [solution.duration for solution in solutions]
#    sizes = [solution.case.size() for solution in solutions]
#    import matplotlib.pyplot as plt
#    plt.subplot(1,2,1)
#    plt.plot(durations, values)
#    plt.xlabel("Durations")
#    plt.ylabel("Values")
#    plt.subplot(1,2,2)
#    plt.bar(durations, sizes)
#    plt.show()

#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 15141, 'duration': 0.413}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 4795, 'duration': 0.4176}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 18464, 'duration': 1.4004}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 1468, 'duration': 1.1328}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 62916, 'duration': 2.248}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 50334, 'duration': 1.2748}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 87747, 'duration': 12.548}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 5868, 'duration': 10.8955}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 329468, 'duration': 24.2685}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 270825, 'duration': 16.5951}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 732261, 'duration': 56.9867}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 56670, 'duration': 49.0317}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1667254, 'duration': 68.7656}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 463624, 'duration': 71.4095}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1655326, 'duration': 214.1147}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 88238, 'duration': 101.9019}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 2029337, 'duration': 146.6675}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 948038, 'duration': 100.6385}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 5775888, 'duration': 1119.5582}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 3477841, 'duration': 502.6625}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 4800274, 'duration': 3110.077}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 3703118, 'duration': 949.0124}
#)
#({'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 11128645, 'duration': 7005.3723}
#, {'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 2976370, 'duration': 1581.2668}
#)