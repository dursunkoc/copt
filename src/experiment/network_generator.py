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
    print(f"+--------------+---------------+----------------+----------------+------------+------------------------+----------------------------+---------------------------+")
    print(f"| Nodes        | Edges         | Maximum Degree | Average Degree | Mod Degree | Nodes with Degree 0    | Nodes with Maximum Degree  | Connected Components      |")
    print(f"+--------------+---------------+----------------+----------------+------------+------------------------+----------------------------+---------------------------+")
    for U in Us:
        #seed=142, net_type='erdos', m=None, p=.003, drop_prob=.995
        a_uv, G = gen_network(seed=142, p=0.003, drop_prob=.60, n=U, m=None, net_type='erdos')
        #Find the number of nodes in the network
        num_nodes = len(G.nodes())
        #Find the number of edges in the network
        num_edges = len(G.edges())
        #Find the maximum degree of the network
        max_degree = max([d for n, d in G.degree()])
        #Find the average degree of the network
        avg_degree = sum(G.degree(n) for n in G.nodes()) / G.number_of_nodes()
        #Find the mod degree of the network
        mod_degree = np.median([d for n, d in G.degree()])
        #Find the number of nodes with degree 0
        num_zero_degree_nodes = len([n for n, d in G.degree() if d == 0])
        #Find the number of nodes with degree ==MAX
        num_max_degree_nodes = len([n for n, d in G.degree() if d == max_degree])
        #Find the number of connected components in the network
        num_connected_components = nx.number_connected_components(G)
        #print in tabular format
        print(f"| {num_nodes:<12} | {num_edges:<13} | {max_degree:<14} | {avg_degree:<14.2f} | {mod_degree:<10} | {num_zero_degree_nodes:<22} | {num_max_degree_nodes:<26} | {num_connected_components:<25} |")
        print(f"+--------------+---------------+----------------+----------------+------------+------------------------+----------------------------+---------------------------+")


    
    
