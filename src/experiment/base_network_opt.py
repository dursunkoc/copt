from docplex.mp.model import Model
from time import time
from network_generator import gen_network
import numpy as np

def solve_network_model_definite(a_uv, U, e_u, X_val):
    mdl = Model(name='Network Optimization')
    X = {u: mdl.binary_var(f"X_u:{u}") for u in range(0,U)}
    mdl.minimize(mdl.sum([ X[u] for u in range(0,U) ]))
    mdl.add_constraints(
            e_u[u] <= mdl.sum((X[u]) + (mdl.sum(X[v] * a_uv[u,v] for v in range(0,U) if a_uv[u,v]==1))) for u in range(0,U)
    )
    mdl.add_constraints(
        X_val[u] == X[u] for u in range(0,U)
    )

    result = mdl.solve(log_output=False)
    return result

def solve_network_model(a_uv, U, e_u, with_obj=False):
    mdl = Model(name='Network Optimization')
    X = {u: mdl.binary_var(f"X_u:{u}")
                for u in range(0,U)}
    
    mdl.minimize(mdl.sum([ X[u] for u in range(0,U) ]))

    mdl.add_constraints(
            e_u[u] <= mdl.sum((X[u]) + (mdl.sum(X[v] * a_uv[u,v] for v in range(0,U) if a_uv[u,v]==1)))
            for u in range(0,U)
    )
#    mdl.add_constraints(
#            (X[u] <= 0 for u in range(0,U) if e_u[u]==0)
#    )
    result = mdl.solve(log_output=False)
#    print("Solved Model: "+str(end_time - start_time))
#    print("========")
#    print("Model:")
#    print("========")
#    print("Min:")
#    print(f"\t{mdl._objective_expr}")
#    print("subject to:")
#    for c in mdl.iter_constraints():
#        print(f"\t{c}")
#    print("========")
#    print("obj:", result.objective_value)
#print_solution(result)
#    print("========")

    X_u = np.zeros(U, dtype='int')
    if result is not None and result.as_name_dict() is not None:
        for ky,_ in result.as_name_dict().items():
            exec(f'X_u{[int(i.split(":")[1]) for i in ky.split("_")[1:]]} = 1', {}, {'X_u':X_u})
    if with_obj:
        return (X_u, result.objective_value)
    return X_u

def print_solution(solution):
    print("Solution: ")
    if solution is not None and solution.as_name_dict() is not None:
        for ky,v in solution.as_name_dict().items():
            print(ky, "==>" ,v)

if __name__ == '__main__':
    U=50
    a_uv, grph = gen_network(seed=1, p=None, n=U, m=4, drop_prob=.9, net_type='barabasi')
    X_u = solve_network_model(a_uv=a_uv, U=U)
    print(X_u)
    print(max([0,1,2,3,4,5], key=lambda x: X_u[x]==1))