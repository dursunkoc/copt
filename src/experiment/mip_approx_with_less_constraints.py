from datetime import datetime
from pickle import FALSE
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
from camps_order_model import start_model as camps_order_model
from tqdm import trange
from tqdm import tqdm
from time import time
import numpy as np
import co_constraints as cstr

class ApproxSolution(Solution, MipCore):
    def __init__(self):
        super().__init__("Approx")

    def runPh(self, case:Case, Xp_cuhd):
        start_time = time()
        #variables
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        PMS:Parameters = super().generate_parameters(case, Xp_cuhd)
#        mdl, _ = super().start_model(True, PMS, C, U, H, D, I)
        mdl, _ = super().start_trivial_model(True, PMS, C, U, H, D, I)
        mdl.set_time_limit(3600)

        result = mdl.solve(log_output=False)

        if result is not None:
            value = result.objective_value
        else:
            value = 0

        end_time = time()
        duration = end_time - start_time
        #self.validate(result, PMS, C, D, H, U)
        #self.anti_validate(result, PMS,  C, D, H, U)
        print("Solved, creating var for greedy")
        resp = (None, SolutionResult(case, value, round(duration,4)))
        del mdl
        del result
        with open(f'result_est_mip_{datetime.now().strftime("%d-%m-%Y %H_%M_%S")}.txt','w') as f:
            f.write(repr(resp[1]))
        return resp

    def create_var_for_greedy(self, solution, C, D, H, U):
        X_cuhd2 = np.zeros((C,U,H,D), dtype='int')
        if solution is not None and solution.as_name_dict() is not None:
            for ky,_ in solution.as_name_dict().items():
                exec(f'X_cuhd2{[int(i.split(":")[1]) for i in ky.split("_")[1:]]} = 1', {}, {'X_cuhd2':X_cuhd2})
        return X_cuhd2


if __name__ == '__main__':
    from cases import cases
    expr = Experiment(cases)
    solutions = expr.run_cases_with(ApproxSolution(), False)
    for solution in solutions:
        print(solution)

#
#raceback (most recent call last):
#  File "D:\dev\workspaces\python\copt\src\experiment\mip_approx_with_less_constraints.py", line 59, in <module>
#    solutions = expr.run_cases_with(ApproxSolution(), False)
#  File "D:\dev\workspaces\python\copt\src\experiment\experiment.py", line 74, in run_cases_with
#    return [solution.run(case, ph) for case in tqdm(self.cases, f"Case ->")]
#  File "D:\dev\workspaces\python\copt\src\experiment\experiment.py", line 74, in <listcomp>
#    return [solution.run(case, ph) for case in tqdm(self.cases, f"Case ->")]
#  File "D:\dev\workspaces\python\copt\src\experiment\experiment.py", line 86, in run
#    (Xp_cuhd1, sr1)=self.runPh(case,None)
#  File "D:\dev\workspaces\python\copt\src\experiment\mip_approx_with_less_constraints.py", line 26, in runPh
#    mdl, _ = super().start_trivial_model(True, PMS, C, U, H, D, I)
#  File "D:\dev\workspaces\python\copt\src\experiment\mip_core.py", line 101, in start_trivial_model
#    X = {(c,u,h,d): mdl.binary_var(f"X[{c},{u},{h},{d}]")
#  File "D:\dev\workspaces\python\copt\src\experiment\mip_core.py", line 101, in <dictcomp>
#    X = {(c,u,h,d): mdl.binary_var(f"X[{c},{u},{h},{d}]")
#  File "D:\dev\platforms\python\envs\copt\lib\site-packages\docplex\mp\model.py", line 2275, in binary_var
#    return self._var(self.binary_vartype, name=name)
#  File "D:\dev\platforms\python\envs\copt\lib\site-packages\docplex\mp\model.py", line 2238, in _var
#    return self._lfactory.new_var(vartype, lb, ub, name)
#  File "D:\dev\platforms\python\envs\copt\lib\site-packages\docplex\mp\mfactory.py", line 232, in new_var
#    return self._make_new_var(vartype, rlb, rub, used_varname, origin=None)
#  File "D:\dev\platforms\python\envs\copt\lib\site-packages\docplex\mp\mfactory.py", line 217, in _make_new_var
#    self_model._register_one_var(var, idx, varname)
#  File "D:\dev\platforms\python\envs\copt\lib\site-packages\docplex\mp\model.py", line 1344, in _register_one_var
#    self.__notify_new_model_object("variable", var, var_index, var_name, self._vars_by_name, self._var_scope)
#  File "D:\dev\platforms\python\envs\copt\lib\site-packages\docplex\mp\model.py", line 1337, in __notify_new_model_object
#    name_dir[mobj_name] = mobj
#MemoryError