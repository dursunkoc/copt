from network_generator import gen_network
import numpy as np

class Case():
    def __init__(self, indiv_size) -> None:
        self.indiv_size=indiv_size
        self.a_uv, self.e_u = case_gen(indiv_size)

def case_gen(indiv_size):
    a_uv, _ = gen_network(seed=12, p=None, n=indiv_size,
                          m=3, drop_prob=.8, net_type='barabasi')
    np.random.seed(12)
    e_u = np.random.choice(2, indiv_size)
    return a_uv, e_u


cases = [
         Case(10),
         Case(10),
         Case(20),
         Case(50),
         Case(100),
         Case(200),
         Case(500),
         Case(1000),
#         Case(5000),
#         Case(10000),
        ]
if __name__ =='__main__':
    for case in cases:
        print(case.indiv_size)
        print(case.e_u)
        print(case.a_uv)