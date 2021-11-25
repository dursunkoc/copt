from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np
from time import time
from network_generator import gen_network

class MipSolutionWithNetwork(Solution, MipCore):
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("MIP With Network")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type


    def runPh(self, case:Case, Xp_cuhd=None)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        nw_start_time = time()
        print("Building Network", nw_start_time)
        a_uv, _ = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd, a_uv=a_uv)
        mdl, _ = super().start_model(True, PMS, C, U, H, D, I)

        result = mdl.solve(log_output=False)

        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time - nw_duration
        X = self.create_var_for_greedy(result, C, D, H, U)
        direct_msg = X.sum()
        total_edges = a_uv.sum()
        resp = (X, SolutionResult(case, value, round(duration,4), {'direct_msg': direct_msg, 'total_edges':total_edges}))
        del mdl
        del result
        print(resp)
        return resp

    def create_var_for_greedy(self, solution, C, D, H, U):
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        if solution is not None and solution.as_name_dict() is not None:
            for ky,_ in solution.as_name_dict().items():
                if ky.startswith('X'):
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
#            Case({"C":2,"U":100,"H":3, "D":7, "I":3, "P":3}),#1
#            Case({"C":5,"U":100,"H":3, "D":7, "I":3, "P":3}),#2
#            Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3}),#3
#            Case({"C":5,"U":1000,"H":3, "D":7, "I":3, "P":3}),#4
#            Case({"C":10,"U":1000,"H":3, "D":7, "I":3, "P":3}),#5
            Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3}),#6
            Case({"C":10,"U":3000,"H":3, "D":7, "I":3, "P":3}),#7
            Case({"C":10,"U":4000,"H":3, "D":7, "I":3, "P":3}),#8
            Case({"C":10,"U":5000,"H":3, "D":7, "I":3, "P":3}),#9
#            Case({"C":20,"U":10000,"H":3, "D":7, "I":3, "P":3}),#10
#            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),#11
#            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),#12
#            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3})
            ]
    expr = Experiment(cases)
    #solutions = expr.run_cases_with(MipSolutionWithNetwork(seed=142, net_type='erdos', m=None, p=.03, drop_prob=None), False)
    solutions = expr.run_cases_with(MipSolutionWithNetwork(seed=142, net_type='barabasi', m=3, p=None, drop_prob=.8), False)
    print(solutions)
    print("values:")
#    print(" ".join([str(v.value) for v in [c for solution in solutions for c in solution]]))
    print(" ".join([str(v.value) for v in [solution for solution in solutions]]))
    print("durations:")
#    print(" ".join([str(v.duration) for v in [c for solution in solutions for c in solution]]))
    print(" ".join([str(v.duration) for v in [solution  for solution in solutions]]))

#Case ->:   0%|                                                                                   | 0/9 [00:00<?, ?it/s]Building Network 1635411980.806674
#Built Network 1635411980.80905  duration: 0.002376079559326172
#Case ->:  11%|████████▎                                                                  | 1/9 [00:04<00:34,  4.37s/it]Building Network 1635411985.1802208
#Built Network 1635411985.181868  duration: 0.0016472339630126953
#Case ->:  22%|████████████████▋                                                          | 2/9 [00:15<00:56,  8.14s/it]Building Network 1635411995.9559731
#Built Network 1635411995.961826  duration: 0.005852937698364258
#Case ->:  33%|█████████████████████████                                                  | 3/9 [00:48<01:56, 19.47s/it]Building Network 1635412028.899165
#Built Network 1635412029.043518  duration: 0.14435315132141113
#Case ->:  44%|████████████████████████████████▉                                         | 4/9 [12:27<23:59, 287.95s/it]Building Network 1635412728.435783
#Built Network 1635412728.583394  duration: 0.14761114120483398
#Case ->:  56%|█████████████████████████████████████████                                 | 5/9 [35:09<45:00, 675.24s/it]Building Network 1635414090.380577
#Built Network 1635414090.9017391  duration: 0.5211620330810547
#Case ->:  67%|██████████████████████████████████████████████                       | 6/9 [2:06:46<1:55:43, 2314.65s/it]Building Network 1635419587.390611
#Built Network 1635419588.6521769  duration: 1.261565923690796
#Case ->:  78%|█████████████████████████████████████████████████████▋               | 7/9 [5:26:06<3:02:16, 5468.06s/it]Building Network 1635431547.755811
#Built Network 1635431550.094739  duration: 2.338927984237671
#Case ->:  89%|███████████████████████████████████████████████████████████▌       | 8/9 [11:21:22<2:55:13, 10513.29s/it]Building Network 1635452863.7491798
#Built Network 1635452867.267757  duration: 3.5185770988464355
#Case ->: 100%|██████████████████████████████████████████████████████████████████████| 9/9 [20:33:26<00:00, 8222.96s/it]
#values:
#9546.0 18164.0 57717.0 69000.0 508000.0 1230000.0 1728000.0 1888000.0 2660000.0
#durations:
#4.3644 10.7578 32.8997 699.2396 1361.5039 5495.8364 11958.204 21312.3695 33117.8134


#Case ->:   0%|                                                                               | 0/9 [00:00<?, ?it/s]Building Network 1637436576.479899
#Built Network 1637436576.484559  duration: 0.004660129547119141
#Case ->:  11%|███████▉                                                               | 1/9 [00:04<00:33,  4.16s/it]Building Network 1637436580.6371021
#Built Network 1637436580.641106  duration: 0.004003763198852539
#Case ->:  22%|███████████████▊                                                       | 2/9 [00:14<00:54,  7.81s/it]Building Network 1637436591.007412
#Built Network 1637436591.018024  duration: 0.010612010955810547
#Case ->:  33%|███████████████████████▋                                               | 3/9 [00:45<01:49, 18.19s/it]Building Network 1637436621.547246
#Built Network 1637436621.7158558  duration: 0.1686098575592041
#Case ->:  44%|███████████████████████████████                                       | 4/9 [11:06<21:21, 256.34s/it]Building Network 1637437242.983028
#Built Network 1637437243.039517  duration: 0.056488990783691406
#Case ->:  56%|██████████████████████████████████████▉                               | 5/9 [31:51<40:50, 612.70s/it]Building Network 1637438487.548598
#Built Network 1637438487.676822  duration: 0.12822389602661133
#Case ->:  67%|███████████████████████████████████████████▎                     | 6/9 [1:52:14<1:42:13, 2044.52s/it]Building Network 1637443311.4731789
#Built Network 1637443311.686791  duration: 0.21361207962036133
#Case ->:  78%|██████████████████████████████████████████████████▌              | 7/9 [4:53:51<2:44:37, 4938.57s/it]Building Network 1637454208.359286
#Built Network 1637454208.678161  duration: 0.3188748359680176
#Case ->:  89%|████████████████████████████████████████████████████████▉       | 8/9 [10:14:35<2:38:12, 9492.70s/it]Building Network 1637473452.233263
#Built Network 1637473452.662057  duration: 0.42879390716552734
#Case ->: 100%|██████████████████████████████████████████████████████████████████| 9/9 [18:38:41<00:00, 7457.94s/it]
#[{'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 7881.0, 'duration': 4.1451, 'info': {'direct_msg': 2982, 'total_edges': 122}}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 14538.0, 'duration': 10.3508, 'info': {'direct_msg': 7392, 'total_edges': 122}}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 43329.0, 'duration': 30.5001, 'info': {'direct_msg': 15141, 'total_edges': 254}}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 47046.0, 'duration': 621.121, 'info': {'direct_msg': 71778, 'total_edges': 1184}}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 344228.0, 'duration': 1244.2232, 'info': {'direct_msg': 141981, 'total_edges': 1184}}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 806541.0, 'duration': 4823.2282, 'info': {'direct_msg': 269241, 'total_edges': 2360}}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1215742.0, 'duration': 10895.7916, 'info': {'direct_msg': 440496, 'total_edges': 3630}}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1294222.0, 'duration': 19242.4799, 'info': {'direct_msg': 519897, 'total_edges': 4878}}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1847441.0, 'duration': 30243.7569, 'info': {'direct_msg': 724647, 'total_edges': 6056}}
#]
#values:
#7881.0 14538.0 43329.0 47046.0 344228.0 806541.0 1215742.0 1294222.0 1847441.0
#durations:
#4.1451 10.3508 30.5001 621.121 1244.2232 4823.2282 10895.7916 19242.4799 30243.7569



#[{'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 7881.0, 'duration': 5.3368, 'info': {'direct_msg': 84, 'total_edges': 122}}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 14538.0, 'duration': 11.5459, 'info': {'direct_msg': 188, 'total_edges': 122}}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 43329.0, 'duration': 31.9515, 'info': {'direct_msg': 401, 'total_edges': 254}}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 47046.0, 'duration': 704.2374, 'info': {'direct_msg': 1919, 'total_edges': 1184}}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 344228.0, 'duration': 1438.309, 'info': {'direct_msg': 3802, 'total_edges': 1184}}
#Case ->: 100%|█████████████████████████████████████████████████████████████████| 4/4 [19:13:13<00:00, 17298.34s/it]
#[{'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 806541.0, 'duration': 5210.3223, 'info': {'direct_msg': 7227, 'total_edges': 2360}}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1215742.0, 'duration': 11644.8602, 'info': {'direct_msg': 11622, 'total_edges': 3630}}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1294222.0, 'duration': 20592.2362, 'info': {'direct_msg': 13613, 'total_edges': 4878}}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1847441.0, 'duration': 31743.0434, 'info': {'direct_msg': 19333, 'total_edges': 6056}}
#]
#values:
#806541.0 1215742.0 1294222.0 1847441.0
#durations:
#5210.3223 11644.8602 20592.2362 31743.0434