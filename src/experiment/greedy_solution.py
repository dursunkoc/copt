from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from camps_order_model import start_model as camps_order_model
from tqdm import trange
from tqdm import tqdm
from time import time
import numpy as np

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
        I = case.arguments["I"]  # number of planning days.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
        #variables
        X_cuhd = np.zeros((C,U,H,D), dtype='int')
        mdl, Y = camps_order_model(C, D, I, PMS)
        camps_order_result = mdl.solve()

        camp_order = np.array(
            [
                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int')
        camp_prio = (camp_order) * PMS.rp_c
        for c in tqdm(np.lexsort((-PMS.l_c,-PMS.rp_c))):#tqdm(np.argsort(-(PMS.rp_c)), desc="Campaigns Loop"):
            for d in range(D):
                Ur = X_cuhd.sum(axis=(0,2,3)).argsort()
                for u in Ur[::-1]:
                    for h in range(H):
                        X_cuhd[c,u,h,d]=1
                        if not self.check(X_cuhd, PMS, (c, u, h, d)):
                            X_cuhd[c,u,h,d]=0
        end_time = time()
        value=self.objective_fn(PMS.rp_c, X_cuhd)
        duration = end_time - start_time
        return (X_cuhd, SolutionResult(case, value, round(duration,4)))

if __name__ == '__main__':
    cases = [
#            Case({"C":2,"U":100,"H":3, "D":7, "I":3, "P":3}),#1
#            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),#2
#            Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3}),#3
#            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),#4
#            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),#5
#            Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3}),#6
#            Case({"C":10,"U":3000,"H":3, "D":7, "I":3, "P":3}),#7
#            Case({"C":10,"U":4000,"H":3, "D":7, "I":3, "P":3}),#8
#            Case({"C":10,"U":5000,"H":3, "D":7, "I":3, "P":3}),#9
#            Case({"C":20,"U":10000,"H":3, "D":7, "I":3, "P":3}),#10
#            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),#11
#            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),#12
            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),#13
            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3})#14
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(GreedySolution(), False)
    print("values:")
#    print(" ".join([str(v.value) for v in [c for solution in solutions for c in solution]]))
    print(" ".join([str(v.value) for v in [solution[0] for solution in solutions]]))
    print("durations:")
#    print(" ".join([str(v.duration) for v in [c for solution in solutions for c in solution]]))
    print(" ".join([str(v.duration) for v in [solution[0] for solution in solutions]]))

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


#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 15141, 'duration': 0.9087}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 4795, 'duration': 0.886}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 18464, 'duration': 2.8772}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 1468, 'duration': 2.3368}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 62916, 'duration': 4.8879}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 50334, 'duration': 2.8236}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 87747, 'duration': 26.7036}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 5868, 'duration': 22.749}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 329468, 'duration': 50.0076}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 270825, 'duration': 35.0902}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 732261, 'duration': 114.2823}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 56670, 'duration': 96.7616}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1667254, 'duration': 125.6855}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 463624, 'duration': 135.7003}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1655326, 'duration': 280.082}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 88238, 'duration': 193.8125}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 2029337, 'duration': 257.3131}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 948038, 'duration': 179.1541}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 5775888, 'duration': 1526.9442}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 3477841, 'duration': 911.3727}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 4800274, 'duration': 3839.3523}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 3703118, 'duration': 1662.7208}
#)


# Final
#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 15141, 'duration': 0.671}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 15141, 'duration': 0.3998}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 18464, 'duration': 1.6584}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 18464, 'duration': 1.1163}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 62916, 'duration': 2.771}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 62916, 'duration': 1.3845}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 87747, 'duration': 15.6338}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 87747, 'duration': 8.707}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 329468, 'duration': 32.8776}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 329468, 'duration': 14.837}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 732261, 'duration': 64.1421}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 732261, 'duration': 35.284}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1667254, 'duration': 70.2392}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1667254, 'duration': 33.2197}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1655326, 'duration': 128.3947}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1655326, 'duration': 63.618}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 2029337, 'duration': 142.7931}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 2029337, 'duration': 61.2959}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5775888, 'duration': 903.1692}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 5775888, 'duration': 448.587}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 4800274, 'duration': 2534.8457}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 4800274, 'duration': 1474.5264}
#)

#
##After Fix
#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 13083, 'duration': 0.1159}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 13083, 'duration': 0.1159}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 11964, 'duration': 0.1693}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 11964, 'duration': 0.1693}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 62616, 'duration': 0.2474}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 62616, 'duration': 0.2474}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 47806, 'duration': 1.689}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 47806, 'duration': 1.689}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 313771, 'duration': 2.7625}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 313771, 'duration': 2.7625}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 687984, 'duration': 6.7322}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 687984, 'duration': 6.7322}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1555894, 'duration': 8.8569}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1555894, 'duration': 8.8569}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1102844, 'duration': 14.9555}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1102844, 'duration': 14.9555}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1716160, 'duration': 19.6974}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1716160, 'duration': 19.6974}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5291782, 'duration': 226.9922}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5291782, 'duration': 226.9922}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 3339926, 'duration': 1521.3785}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 3339926, 'duration': 1521.3785}
#)
#({'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 11467206, 'duration': 3485.0269}
#, {'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 11467206, 'duration': 3485.0269}
#)
#({'case': {'C': 20, 'U': 40000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 25556643, 'duration': 8933.781}
#, {'case': {'C': 20, 'U': 40000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 25556643, 'duration': 8933.781}
#)
#({'case': {'C': 20, 'U': 50000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 20915792, 'duration': 7938.3072}
#, {'case': {'C': 20, 'U': 50000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 20915792, 'duration': 7938.3072}
#)



#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 13083, 'duration': 0.1538}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 13083, 'duration': 0.1004}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 11964, 'duration': 0.2005}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 11964, 'duration': 0.2001}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 62616, 'duration': 0.3004}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 62616, 'duration': 0.2852}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 47806, 'duration': 1.6795}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 47806, 'duration': 1.6732}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 313771, 'duration': 2.7229}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 313771, 'duration': 2.6702}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 687984, 'duration': 6.4081}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 687984, 'duration': 6.4945}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1555894, 'duration': 8.6748}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1555894, 'duration': 8.7125}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1102844, 'duration': 14.9939}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1102844, 'duration': 14.8641}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1716160, 'duration': 19.4305}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1716160, 'duration': 19.2857}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5291782, 'duration': 236.2984}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 5291782, 'duration': 231.6517}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 3339926, 'duration': 1524.5995}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 3339926, 'duration': 1508.7853}