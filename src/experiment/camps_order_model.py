from docplex.mp.model import Model
from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
import numpy as np

def start_model(C, D, I, PMS):
    mdl = Model(name='Campaigns Order Optimization')
    E_c = PMS.e_cu.sum(axis=1)
    L_c = (PMS.l_c*E_c)
    E_d_i = (PMS.q_ic @ E_c.reshape(C,1)).reshape(I,)
    M_i = E_d_i * PMS.m_i
    N_i = E_d_i * PMS.n_i

    Y = {(c,d): mdl.continuous_var(lb=0, name=f"Y_c:{c}_d:{d}")
            for c in range(0,C)
            for d in range(0,D)}
    mdl.maximize(mdl.sum([Y[(c,d)]
                  for c in range(0,C)
                  for d in range(0,D)])) #29
    mdl.add_constraints(
            (mdl.sum(Y[(c,d)] for d in range(0,D)) <= L_c[c] )
            for c in range(0,C))#30

    mdl.add_constraints((
            (mdl.sum(Y[(c,d)]* PMS.q_ic[i,c]
                for d in range(0,D)
                for c in range(0,C)
                ) <=  M_i[i])
            for i in range(0,I)))#31

    mdl.add_constraints((
                (mdl.sum(Y[(c,d)]*PMS.q_ic[i,c]
                    for c in range(0,C)) <=  N_i[i])
                for d in range(0,D)
                for i in range(0,I)))#32

    mdl.add_constraint(
            (mdl.sum(Y[(c,d)]
                for c in range(0,C)
                for d in range(0,D)) <= E_c.sum() * PMS.b)
            )#33

    mdl.add_constraints(
            (mdl.sum(Y[(c,d)] for c in range(0,C)) <= E_c.sum() * PMS.k )
            for d in range(0,D))#34


    return mdl, Y

if __name__ == '__main__':
    #case = Case({"C":5,"U":200,"H":3, "D":7, "I":3, "P":3})#3
    case = Case({"C":10,"U":2000,"H":3, "D":7, "I":3, "P":3})#6
    solution = Solution("Many")
    PMS:Parameters = solution.generate_parameters(case, None)
    C = case.arguments["C"] # number of campaigns
    U = case.arguments["U"]  # number of customers.
    H = case.arguments["H"]  # number of channels.
    D = case.arguments["D"]  # number of planning days.
    I = case.arguments["I"]  # number of planning days.

    mdl, Y = start_model(C, D, I, PMS)
#    print(mdl.objective_expr)
#    for consts in mdl.iter_constraints():
#        print(consts)
    E_c = PMS.e_cu.sum(axis=1)
    L_c = (PMS.l_c*E_c)
    E_d_i = (PMS.q_ic @ E_c.reshape(C,1)).reshape(I,)
    M_i = E_d_i * PMS.m_i
    N_i = E_d_i * PMS.n_i

    print("e_uc:\n",E_c)
    print("q_ic:\n",PMS.q_ic)
    print("m_i:\n",PMS.m_i)
    print("n_i:\n",PMS.n_i)
    print("l_c:\n",PMS.l_c)
    print("=======================")
    print("L_c:\n",L_c)
    print("E_d_i:\n",E_d_i)
    print("M_i:\n",M_i)
    print("N_i:\n",N_i)
    print("E_c.sum()*b",E_c.sum() * PMS.b)
    camps_order_result = mdl.solve()
    print(f"OV:{camps_order_result.objective_value}")
    print(
        np.array(
            [
                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int').sum(axis=0)
        )

    print(
        (np.array(
            [
                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int') * PMS.rp_c
            ).sum(axis=0)
        )

    print(
        (np.array(
            [
                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int') * PMS.rp_c
            ).sum(axis=1)
        )

    camp_order = np.array(
            [
                [(camps_order_result.get_var_value(Y[(c,d)])) for c in range(C)]
            for d in range(D)]
            , dtype='int')
    camp_prio = (camp_order) * PMS.rp_c
    print(camp_prio.sum(axis=0))