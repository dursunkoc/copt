import numpy as np
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph, barabasi_albert_graph

def gen_network(p, n):
    #TODO: set seed
#    g = erdos_renyi_graph(n=n, p=p)
    g = barabasi_albert_graph(n=n, m=1)
    N_i_j = nx.to_numpy_matrix(g, dtype=np.int0)
    return N_i_j

if __name__ == '__main__':
    net = gen_network(0.000, 1000)
    print(net.shape)
    print(net)
