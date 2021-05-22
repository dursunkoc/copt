from docplex.mp.model import Model
import numpy as np

def start_model(C, D, I, l_c, q_ic, m_i, n_i):
    mdl = Model(name='Campaigns Order Optimization')
    Y = mdl.continuous_var_dict((c,d)
                for c in range(0,C)
                for d in range(0,D))
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
    C=5
    D=7
    I=3
    P=3
    r_p = np.random.choice(100, P) #r_p = np.ones(P, dtype='int8')
    rp_c = np.array([r_p[r] for r in np.random.choice(P, C)])
    q_ic = np.random.choice(2, (I,C)) #q_ic = np.zeros((I,C), dtype='int8')
    l_c = np.random.choice([2,3,4],C)
    ##quota limitations daily/weekly
    m_i = np.random.choice([4,3,5],I)#m_i = np.ones((I), dtype='int8')*10
    n_i = np.random.choice([1,3,2],I)#n_i = np.ones((I), dtype='int8')*10

    mdl, Y = start_model(C, D, I, l_c, q_ic, m_i, n_i)
    #print(mdl.objective_expr)
    #for consts in mdl.iter_constraints():
    #    print(consts)
    result = mdl.solve()
    print(f"OV:{result.objective_value}")
    for d in range(D):
        y_vl = [(int(result.get_var_value(Y[(c,d)])*1000)+l_c[c]) for c in range(C)]
        print(f"{d}->{y_vl}")
        print(f"\t==> rp_c* -->{rp_c*y_vl}")