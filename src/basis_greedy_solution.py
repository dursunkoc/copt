
#import sys
#import os
#sys.path.insert(0, os.path.abspath('../src'))

import numpy as np
from tqdm import trange
from tqdm import tqdm
from dask import delayed

import greedy
import model_parameters as MP
#import dask.array as da

#np.get_printoptions()#["threshold"]
#np.set_printoptions(edgeitems=10, linewidth=150)

#np.random.seed(13)
c_i = 0
u_i = 1
h_i = 2
d_i = 3

mdl = greedy.Model([
    greedy.Constraint('eligibility',MP.eligibility, (c_i, u_i, h_i, d_i,)),
    greedy.Constraint('channel_capacity',MP.channel_capacity, (h_i, d_i,)),
    greedy.Constraint('daily_limitation',MP.daily_limitation, (u_i, d_i,)),
    greedy.Constraint('weekly_limitation',MP.weekly_limitation, (u_i,)),
    greedy.Constraint('campaign_limitation',MP.campaign_limitation, (c_i, u_i,)),
    greedy.Constraint('daily_quota',MP.daily_quota, (u_i, d_i,)),
    greedy.Constraint('one_channel',MP.one_channel, (c_i, u_i, d_i,)),
    greedy.Constraint('weekly_quota',MP.weekly_quota, (u_i,))
], MP.objective_fn)

X_cuhd = np.zeros((MP.C,MP.U,MP.H,MP.D), dtype='int')
for c in tqdm(np.argsort(-MP.rp_c), desc="Campaigns Loop"):
    for d in trange(MP.D, desc=f"Days Loop for campaign-{c}"):
        for h in range(MP.H):#trange(H, desc=f"Channels Loop at Day-{d}, Campapaign-{c}"):
            for u in range(MP.U):#trange(U, desc=f"Users Loop On Campaign-{c}"):
                X_cuhd[c,u,h,d]=1
                if not mdl.execute(X_cuhd, (c, u, h, d)):
                    X_cuhd[c,u,h,d]=0

print("Final Obj Value:", (mdl.calc_value(X_cuhd)))