from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np
from time import time

class MipSolution(Solution, MipCore):
    def __init__(self):
        super().__init__("MIP")

    def runPh(self, case:Case, Xp_cuhd=None)->SolutionResult:
        start_time = time()
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
        mdl, _ = super().start_model(True, PMS, C, U, H, D, I)

        result = mdl.solve(log_output=False)

        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time
#        self.validate(result, PMS, C, D, H, U)
#        self.anti_validate(result, PMS,  C, D, H, U)
        resp = (self.create_var_for_greedy(result, C, D, H, U), SolutionResult(case, value, round(duration,4)))
        del mdl
        del result
        return resp

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
            Case({"C":20,"U":20000,"H":3, "D":7, "I":3, "P":3}),#11
#            Case({"C":20,"U":30000,"H":3, "D":7, "I":3, "P":3}),#12
#            Case({"C":20,"U":40000,"H":3, "D":7, "I":3, "P":3}),
#            Case({"C":20,"U":50000,"H":3, "D":7, "I":3, "P":3})
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(MipSolution(), False)
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


#Case ->: 100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [11:55:33<00:00, 3577.83s/it]
#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 15141.0, 'duration': 1.7273}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 15141.0, 'duration': 2.1082}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 18464.0, 'duration': 4.1629}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 18464.0, 'duration': 5.2069}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 63096.0, 'duration': 6.8209}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 63096.0, 'duration': 8.9002}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 87747.0, 'duration': 41.4071}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 87747.0, 'duration': 53.0306}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 329480.0, 'duration': 78.7437}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 329480.0, 'duration': 100.505}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 779085.0, 'duration': 170.9284}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 779085.0, 'duration': 232.1129}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1673454.0, 'duration': 241.7832}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 1673412.0, 'duration': 315.7213}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 1660792.0, 'duration': 349.4963}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 1660780.0, 'duration': 562.9777}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 2050264.0, 'duration': 481.6846}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 2050264.0, 'duration': 574.7055}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 5800740.0, 'duration': 1737.4111}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 5800794.0, 'duration': 2568.6952}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 4800274.0, 'duration': 4137.185}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 4800274.0, 'duration': 5800.5665}
#)
#({'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1,'value': 11136000.0, 'duration': 8498.1905}
##, {'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2,'value': 11136000.0, 'duration': 16879.1063}
##)
#
#The Final
#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 15141.0, 'duration': 1.5461}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 15141.0, 'duration': 1.9312}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 18464.0, 'duration': 3.9401}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 18464.0, 'duration': 5.1289}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 63096.0, 'duration': 6.3447}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 63096.0, 'duration': 8.5749}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 87747.0, 'duration': 38.189}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 87747.0, 'duration': 48.5313}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 329480.0, 'duration': 72.6745}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 329480.0, 'duration': 93.75}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 779085.0, 'duration': 161.4825}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 779085.0, 'duration': 225.6714}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1673454.0, 'duration': 223.2495}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1673454.0, 'duration': 301.3561}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1660792.0, 'duration': 324.7104}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 1660768.0, 'duration': 478.4661}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 2050264.0, 'duration': 464.8144}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 2050264.0, 'duration': 580.4435}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5800740.0, 'duration': 1655.0168}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 5800794.0, 'duration': 2248.9456}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 0, 'duration': 2945.4788}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 0, 'duration': 4131.8188}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 4800274.0, 'duration': 3689.8222}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 2, 'value': 4800274.0, 'duration': 5671.4978}
#)



#({'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 13083.0, 'duration': 1.0156}
#, {'case': {'C': 2, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 13083.0, 'duration': 1.0156}
#)
#({'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 12180.0, 'duration': 1.7813}
#, {'case': {'C': 5, 'U': 100, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 12180.0, 'duration': 1.7813}
#)
#({'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 63096.0, 'duration': 3.1094}
#, {'case': {'C': 5, 'U': 200, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 63096.0, 'duration': 3.1094}
#)
#({'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 74664.0, 'duration': 18.1875}
#, {'case': {'C': 5, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 74664.0, 'duration': 18.1875}
#)
#({'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 341667.0, 'duration': 38.0938}
#, {'case': {'C': 10, 'U': 1000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 341667.0, 'duration': 38.0938}
#)
#({'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 705516.0, 'duration': 75.3906}
#, {'case': {'C': 10, 'U': 2000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 705516.0, 'duration': 75.3906}
#)
#({'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1672862.0, 'duration': 108.9263}
#, {'case': {'C': 10, 'U': 3000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1672862.0, 'duration': 108.9263}
#)
#({'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1161046.0, 'duration': 181.1272}
#, {'case': {'C': 10, 'U': 4000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1161046.0, 'duration': 181.1272}
#)
#({'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1982340.0, 'duration': 192.5312}
#, {'case': {'C': 10, 'U': 5000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 1982340.0, 'duration': 192.5312}
#)
#({'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5507447.0, 'duration': 922.7621}
#, {'case': {'C': 20, 'U': 10000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 5507447.0, 'duration': 922.7621}
#)
#({'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 0, 'duration': 1680.625}
#, {'case': {'C': 20, 'U': 20000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 0, 'duration': 1680.625}
#)
#({'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 0, 'duration': 2287.1875}
#, {'case': {'C': 20, 'U': 30000, 'H': 3, 'D': 7, 'I': 3, 'P': 3}, 'phase': 1, 'value': 0, 'duration': 2287.1875}
#)