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
    N_i_j = nx.to_numpy_matrix(g, dtype=np.int0)

    return N_i_j, g

if __name__ == '__main__':
    net = gen_network(0.000, 1000)
    print(net.shape)
    print(net)
