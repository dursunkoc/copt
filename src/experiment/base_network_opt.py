from docplex.mp.model import Model
from time import time
from network_generator import gen_network
import numpy as np

def solve_network_model_definite(a_uv, U, e_u, X_val):
    mdl = Model(name='Network Optimization')
    X = {u: mdl.binary_var(f"X_u:{u}") for u in range(0,U)}
    mdl.minimize(mdl.sum([ X[u] for u in range(0,U) ]))
    links = np.where(a_uv==1)
    eligibles = np.where(e_u==1)

    mdl.add_constraints(
#            e_u[u] <= mdl.sum((X[u]) + (mdl.sum(X[v] * a_uv[u,v] for v in range(0,U) if a_uv[u,v]==1))) for u in range(0,U)
            e_u[u] <= mdl.sum((X[u]) + (mdl.sum(X[links[1][_u]] * a_uv[u,links[1][_u]] for _u in range(links[0].shape[0]) if links[0][_u]==u))) for u in range(0,U)
    )
    mdl.add_constraints(
        X_val[u] == X[u] for u in range(0,U)
    )

    result = mdl.solve(log_output=False)
    return result

def solve_network_model(a_uv, U, e_u, with_obj=False):
    mdl = Model(name='Network Optimization')
    print("Model Created")
    X = {u: mdl.binary_var(f"X_u:{u}")
                for u in range(0,U)}
    print("Variables OK")
    mdl.minimize(mdl.sum([ X[u] for u in range(0,U) ]))
    print("Obj OK")
    links = np.where(a_uv==1)
    eligibles = np.where(e_u==1)
    print("AUX OK")
    mdl.add_constraints(
#            e_u[u] <= mdl.sum((X[u]) + (mdl.sum(X[v] * a_uv[u,v] for v in range(0,U) if a_uv[u,v]==1)))
#            e_u[u] <= mdl.sum((X[u]) + (mdl.sum(X[links[1][_u]] * a_uv[u,links[1][_u]] for _u in range(links[0].shape[0]) if links[0][_u]==u)))
#            for u in range(0,U)
        1 <= mdl.sum((X[eligibles[0][_e]]) + (mdl.sum(X[links[1][_u]] * a_uv[eligibles[0][_e],links[1][_u]] for _u in range(links[0].shape[0]) if links[0][_u]==eligibles[0][_e])))
        for _e in range(eligibles[0].shape[0])
    )
    print("CONS OK")
#    mdl.add_constraints(
#            (X[u] <= 0 for u in range(0,U) if e_u[u]==0)
#    )
    mdl.set_time_limit(60)
    result = mdl.solve(log_output=False)
    print("Solved Model")
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
    U=80000
    e_cu = np.random.choice(2,(1, U))
    e_u = e_cu[0]
    a_uv, grph = gen_network(seed=1, p=0.03, n=U, m=4, drop_prob=.95, net_type='erdos')
    X_u = solve_network_model(a_uv=a_uv, U=U, e_u=e_u)
    print(X_u)