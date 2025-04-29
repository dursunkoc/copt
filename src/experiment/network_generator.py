import numpy as np
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph, barabasi_albert_graph
import matplotlib.pyplot as plt
from pylab import *

def gen_network(p, n, m, net_type, drop_prob=None, seed=34) :
    if net_type == 'erdos':
        g = erdos_renyi_graph(n=n, p=p, seed=seed)
    elif net_type == 'barabasi':
        g = barabasi_albert_graph(n=n, m=m, seed=seed)
    if drop_prob is not None:
        np.random.seed(seed)
        for e in g.edges:
            if np.random.random(size=1)<=drop_prob:
                g.remove_edge(e[0],e[1])
    N_i_j = nx.to_numpy_array(g, dtype=int)

    return N_i_j, g

if __name__ == '__main__':
    #seed=142, net_type='erdos', m=None, p=.03, drop_prob=.95), False
    Us = [100,100,200,500,700,1000,1000,2000,3000,4000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000]
    for U in Us:
        #seed=142, net_type='erdos', m=None, p=.003, drop_prob=.995
        a_uv,_ = gen_network(seed=142, p=0.003, drop_prob=.60, n=U, m=None, net_type='erdos')
        print(f"{U} -> {a_uv.sum()}")

    
    
