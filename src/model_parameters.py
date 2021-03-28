import numpy as np

#seed randomization
np.random.seed(23)

C = 2 # number of campaigns
U = 1000 # number of customers.
H = 3 # number of channels.
D = 7 # number of planning days.
I = 3 # number of quota categories.
P = 3 # number of priority categories.

#Parameters

##eligibility
e_cu = np.random.choice(2,(C, U)) #e_cu = np.ones((C, U), dtype='int8')
p_x_cud = np.random.choice(2,(C,U,D)) 

##previous period planning
s_cuhd = np.random.choice(2,(C,U,H,D))

##quota categories
q_ic = np.random.choice(2, (I,C)) #q_ic = np.zeros((I,C), dtype='int8')

##priority categories
r_p = np.random.choice(100, P) #r_p = np.ones(P, dtype='int8')
rp_c = np.array([r_p[r] for r in np.random.choice(P, C)])

##blokage
b = 7

##daily blokage
k = 3

##campaign blockage
l_c = np.random.choice([2,3,4],C)

##quota limitations daily/weekly
m_i = np.random.choice([4,3,5],I)#m_i = np.ones((I), dtype='int8')*10
n_i = np.random.choice([1,3,2],I)#n_i = np.ones((I), dtype='int8')*10

##capacity for channel
t_hd = np.random.choice([U*.7, U*.6, U*.5], (H, D))

#Constraint Functions
eligibility = lambda X, c, u, h, d: X[c,u,h,d]<=e_cu[c,u]
one_channel = lambda X, c, u, d: X[c,u,:,d].sum() <= 1
weekly_limitation = lambda X, u: (X[:,u,:,:]).sum() <= b
weekly_limitation_rh = lambda f_d: lambda X, s, u : X[:,u,:,:f_d].sum() + s[:,u,:,f_d:].sum() <= b
weekly_limitation_rh1 = weekly_limitation_rh(1)
weekly_limitation_rh2 = weekly_limitation_rh(2)
weekly_limitation_rh3 = weekly_limitation_rh(3)
weekly_limitation_rh4 = weekly_limitation_rh(4)
weekly_limitation_rh5 = weekly_limitation_rh(5)
weekly_limitation_rh6 = weekly_limitation_rh(6)
daily_limitation = lambda X, u, d: X[:,u,:,d].sum() <= k
campaign_limitation = lambda X, c, u: X[c,u,:,:].sum() <= l_c[c]
campaign_limitation_rh =  lambda f_d: lambda X, s, c, u: X[c,u,:,:f_d].sum() + s[c,u,:,f_d:].sum() <=l_c[c]
campaign_limitation_rh1=campaign_limitation_rh(1)
campaign_limitation_rh2=campaign_limitation_rh(2)
campaign_limitation_rh3=campaign_limitation_rh(3)
campaign_limitation_rh4=campaign_limitation_rh(4)
campaign_limitation_rh5=campaign_limitation_rh(5)
campaign_limitation_rh6=campaign_limitation_rh(6)
weekly_quota = lambda X, u: all((q_ic * X[:,u,:,:].sum(axis=(1,2))).sum(axis=1)<=m_i)
weekly_quota_rh = lambda f_d :lambda X, s, u: all((q_ic * X[:,u,:,:f_d].sum(axis=(1,2))).sum(axis=1) + (q_ic * s[:,u,:,f_d:].sum(axis=(1,2))).sum(axis=1) <= m_i)
weekly_quota_rh1 = weekly_quota_rh(1)
weekly_quota_rh2 = weekly_quota_rh(2)
weekly_quota_rh3 = weekly_quota_rh(3)
weekly_quota_rh4 = weekly_quota_rh(4)
weekly_quota_rh5 = weekly_quota_rh(5)
weekly_quota_rh6 = weekly_quota_rh(6)
daily_quota = lambda X, u, d: all((q_ic * X[:,u,:,d].sum(axis=(1))).sum(axis=1)<=n_i)
channel_capacity = lambda X, h, d: X[:,:,h,d].sum() <= t_hd[h,d]

#objective function
objective_fn = lambda X: np.matmul(rp_c, X.sum(axis=(1,2,3)))


e_cu_X = np.stack([np.stack([e_cu for _ in range(H)], axis=2) for _ in range(D)], axis=3)
m_i_X = np.stack([m_i for _ in range(U)], axis=1)
n_i_X = np.stack([n_i for _ in range(U)], axis=1)


def X_eligibility (X):
    return (X <= e_cu_X).all()

def X_weekly_limitation (X):
    return (X.sum(axis=(0,2,3))<=b).all()

def X_daily_limitation (X):
    return (X.sum(axis=(0)).sum(axis=(1))<=k).all()

def X_campaign_limitation (X):
    return np.all(X.sum(axis=(2,3)).T<=l_c)

def X_weekly_quota (X):
    for i in range(I):
        quota_i = np.all(X.sum(axis=(2,3)).T * q_ic[i, ].T  <= m_i[i])
        if not quota_i:
            return False
    return True

def X_channel_capacity (X):
    return np.all(X.sum(axis=(0,1))<=t_hd)

def X_daily_quota (X):
    for i in range(I):
        quota_i = np.all((q_ic[i,].T * X.sum(axis=2).T).sum(2) <= n_i[i])
        if not quota_i:
            return False
    return True

def check(X):
    if not X_eligibility(X):
        return False
    if not X_weekly_limitation(X):
        return False
    if not X_daily_limitation(X):
        return False
    if not X_campaign_limitation(X):
        return False
    if not X_weekly_quota(X):
        return False
    if not X_daily_quota(X):
        return False
    if not X_channel_capacity(X):
        return False
    return True