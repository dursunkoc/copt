from docplex.mp.model import Model
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
import numpy as np

def start_model(C, D, I, l_c, q_ic, m_i, n_i):
    mdl = Model(name='Campaigns Order Optimization')
    Y = {(c,d): mdl.continuous_var(lb=0, name=f"Y_c:{c}_d:{d}")
            for c in range(0,C)
            for d in range(0,D)}
    mdl.maximize(mdl.sum([Y[(c,d)]
                  for c in range(0,C)
                  for d in range(0,D)]))
    mdl.add_constraints(
            (mdl.sum(Y[(c,d)] for d in range(0,D)) <= l_c[c] )
            for c in range(0,C))

    mdl.add_constraints((
            (mdl.sum(Y[(c,d)]* q_ic[i,c]
                for d in range(0,D) 
                for c in range(0,C)) <=  m_i[i])
            for i in range(0,I)))

    mdl.add_constraints((
                (mdl.sum(Y[(c,d)]*q_ic[i,c]
                    for c in range(0,C)) <= n_i[i])
                for d in range(0,D)
                for i in range(0,I)))

    return mdl, Y

if __name__ == '__main__':
    case = Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3})#3
    solution = Solution("Many")
    PMS:Parameters = solution.generate_parameters(case, None)
    C = case.arguments["C"] # number of campaigns
    U = case.arguments["U"]  # number of customers.
    H = case.arguments["H"]  # number of channels.
    D = case.arguments["D"]  # number of planning days.
    I = case.arguments["I"]  # number of planning days.

    mdl, Y = start_model(C, D, I, PMS.l_c, PMS.q_ic, PMS.m_i, PMS.n_i)
    #print(mdl.objective_expr)
    #for consts in mdl.iter_constraints():
    #    print(consts)
    print("q_ic:",PMS.q_ic)
    print("m_i:",PMS.m_i)
    print("n_i:",PMS.n_i)
    print("l_c:",PMS.l_c)
    result = mdl.solve()
    print(f"OV:{result.objective_value}")
    print(
        np.array(
            [
                [vv[0] if vv[0]>0 else vv[1] for vv in [(result.get_var_value(Y[(c,d)])*10000, PMS.l_c[c]) for c in range(C)] ]
            for d in range(D)]
            , dtype='int').min(axis=0) * PMS.rp_c
        )

    print(
        np.array(
            [
                [(result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int')
        )