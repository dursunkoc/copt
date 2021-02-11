import numpy as np

#seed randomization
np.random.seed(13)

C = 10 # number of campaigns
U = 10000 # number of customers.
H = 3 # number of channels.
D = 7 # number of planning days.
I = 3 # number of quota categories.
P = 10 # number of priority categories.

#Parameters

##eligibility
e_cu = np.random.choice(2,(C, U)) #e_cu = np.ones((C, U), dtype='int8')

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
weekly_limitation = lambda X, u: X[:,u,:,:].sum() <= b
daily_limitation = lambda X, u, d: X[:,u,:,d].sum() <= k
campaign_limitation = lambda X, c, u: X[c,u,:,:].sum() <= l_c[c]
weekly_quota = lambda X, u: all((q_ic * X[:,u,:,:].sum(axis=(1,2))).sum(axis=1)<=m_i)
daily_quota = lambda X, u, d: all((q_ic * X[:,u,:,d].sum(axis=(1))).sum(axis=1)<=n_i)
channel_capacity = lambda X, h, d: X[:,:,h,d].sum() <= t_hd[h,d]

#objective function
objective_fn = lambda X: np.matmul(rp_c, X.sum(axis=(1,2,3)))
