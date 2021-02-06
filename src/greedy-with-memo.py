import numpy as np
from tqdm.notebook import trange
from tqdm.notebook import tqdm
from dask import delayed
import dask.array as da

class Constraint:
    def __init__(self, fnc, fixed_indicies):
        self.fnc = fnc
        self.fixed_indicies = fixed_indicies
        
    def __variable_matched(self, indicies):
        for (k,v) in self.fixed_indicies.items():
            if(indicies[k]!=v):
                return False
        return True
    
    def get_func(self):
        return self.fnc
    
    def execute_considering(self, X, indicies):
        '''
        indicies: tuple of indicies indicating that would change from 0->1
        X: is the variables array
        indicies = (4, 100, 0, 1 )
        X_cuhd[indicies]
        '''
        if not self.__variable_matched(indicies):
            return True
        else:
            params = tuple([X]) + tuple([(v)for (k,v) in sorted(self.fixed_indicies.items())])
            return self.fnc(*params)
        
        
class Model:
    def __init__(self, constraints):
        self.constraints = constraints
        
    def execute(self, X, indicies):
        for c in self.constraints:
            res = c.execute_considering(X, indicies)
            if not res.all():
                return False
        return True

np.get_printoptions()#["threshold"]
np.set_printoptions(edgeitems=10, linewidth=150)

np.random.seed(13)

C = 5 # number of campaigns
U = 10 # number of customers.
H = 2 # number of channels.
D = 7 # number of planning days.
I = 3 # number of quota categories.
P = 10 # number of priority categories.

e_cu = np.random.choice(2,(C, U))

r_p = np.random.choice(100, P)
rp_c = np.array([r_p[r] for r in np.random.choice(P, C)])


X_cuhd = np.zeros((C,U,H,D), dtype='int')

cons = [Constraint(lambda X, h, d: X[:,:,h,d]<=e_cu, {2:h, 3:d}) for h in range(H) for d in range(D)]
m=Model(cons)
for d in trange(D, desc=f"Days Loop for campaign"):
    for c in tqdm(np.argsort(-rp_c), desc="Campaigns Loop"):
        for h in range(H):#trange(H, desc=f"Channels Loop at Day-{d}"):
            for u in range(U):#trange(U, desc=f"Users Loop On Campaign-{c}"):
                X_cuhd[c,u,h,d]=1
                m.execute(X_cuhd, (c, u, h, d))