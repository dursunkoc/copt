from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
from mip_core import MipCore
import numpy as np
from time import time
from network_generator import gen_network

class RandomNetworkGenerator():
    def __init__(self, seed, net_type, m=1, p=.04, drop_prob=.10):
        super().__init__("MIP With Network")
        self.seed = seed
        self.m = m
        self.p = p
        self.drop_prob = drop_prob
        self.net_type = net_type


    def gen_net(self, U):
        start_time = time()
        nw_start_time = time()
        print("Building Network", nw_start_time)
        a_uv, _ = gen_network(seed=self.seed, p=self.p, n=U, m=self.m, drop_prob=self.drop_prob, net_type=self.net_type)
        nw_end_time = time()
        nw_duration = nw_end_time - nw_start_time
        print("Built Network", nw_end_time, " duration:", nw_duration)
        return a_uv

if __name__ == '__main__':
    cases = [100,#1 #2
             200,#3
             1000,#4 #5
             2000,#6
             3000,#7
             4000,#8
             5000,#9
            ]
    expr = Experiment(cases)
    for c in cases:
        RandomNetworkGenerator(seed=142, net_type='erdos', m=None, p=.03, drop_prob=None)
