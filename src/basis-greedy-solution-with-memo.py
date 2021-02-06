
import numpy as np
from tqdm.notebook import trange
from tqdm.notebook import tqdm
from dask import delayed
import dask.array as da

np.random.seed(13)


C = 10 # number of campaigns
U = 1000 # number of customers.
H = 3 # number of channels.
D = 7 # number of planning days.
I = 3 # number of quota categories.
P = 10 # number of priority categories.


e_cu = np.random.choice(2,(C, U))

q_ic = np.random.choice(2, (I,C))

r_p = np.random.choice(100, P)
rp_c = np.array([r_p[r] for r in np.random.choice(P, C)])

b = 4

k = 2

l_c = np.random.choice([2,3,4],C)

m_i = np.random.choice([10,15,18],I)
n_i = np.random.choice([2,3,4],I)

t_hd = np.random.choice([70], (H, D))

X_cuhd = np.zeros((C,U,H,D), dtype='int')


eligibility = lambda X, c, u, h, d: X[c,u,h,d]<=e_cu[c,u]
one_channel = lambda X, c, u, d: X[c,u,:,d].sum() <= 1
weekly_limitation = lambda X, u: X[:,u,:,:].sum() <= b
daily_limitation = lambda X, u, d: X[:,u,:,d].sum() <= k
campaign_limitation = lambda X, c, u: X[c,u,:,:].sum() <= l_c[c]
weekly_quota = lambda X, u: np.all(q_ic[:,c] * X[c,u,:,:].sum()<=m_i)
daily_quota = lambda X, u, d: np.all(q_ic[:,c] * X[c,u,:,d].sum()<=n_i)
channel_capacity = lambda X, h, d: X[:,:,h,d].sum() <= t_hd[h,d]

X_cuhd = np.zeros((C,U,H,D), dtype='int')
for c in tqdm(np.argsort(-rp_c), desc="Campaigns Loop"):
    for d in trange(D, desc=f"Days Loop for campaign-{c}"):
        for h in trange(H, desc=f"Channels Loop at Day-{d}, Campapaign-{c}"):
            for u in range(U):#trange(U, desc=f"Users Loop On Campaign-{c}"):
                X_cuhd[c,u,h,d]=1
                if not eligibility(X_cuhd, c, u, h, d):
                    X_cuhd[c,u,h,d]=0
                if not one_channel(X_cuhd, c, u, d):
                    X_cuhd[c,u,h,d]=0
                if not weekly_limitation(X_cuhd, u):
                    X_cuhd[c,u,h,d]=0
                if not daily_limitation(X_cuhd, u, d):
                    X_cuhd[c,u,h,d]=0
                if not campaign_limitation(X_cuhd, c, u):
                    X_cuhd[c,u,h,d]=0
                if not weekly_quota(X_cuhd, u):
                    X_cuhd[c,u,h,d]=0
                if not daily_quota(X_cuhd, u, d):
                    X_cuhd[c,u,h,d]=0
                if not channel_capacity(X_cuhd, h, d):
                    X_cuhd[c,u,h,d]=0
print(np.matmul(rp_c, X_cuhd.sum(axis=(1,2,3))))