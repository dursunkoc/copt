from time import time
from typing import Callable, List, Tuple
import numpy as np
from numba import njit, jit
from experiment import Solution, Case, Parameters

@njit
def eligibility(e_cu:np.ndarray, X:np.ndarray, c:int, u:int, h:int, d:int)->np.ndarray:
        return X[c,u,h,d]<=e_cu[c,u]

@njit
def weekly_limitation(b, X, u):
    return X[:,u,:,:].sum() <= b

@njit
def weekly_limitation_rh(b, X, s, u, f_d):
    return X[:,u,:,:f_d].sum() + s[:,u,:,f_d:].sum() <= b

@njit
def daily_limitation (k, X, u, d):
    return X[:,u,:,d].sum() <= k

@njit
def campaign_limitation(l_c, X, c, u):
    return X[c,u,:,:].sum() <=l_c[c]

@njit
def campaign_limitation_rh(l_c, X, s, c, u, f_d):
    return X[c,u,:,:f_d].sum() + s[c,u,:,f_d:].sum() <=l_c[c]

@njit
def weekly_quota(m_i, q_ic, X, u):
    return np.all((q_ic * X[:,u,:,:].sum(axis=(1,2))).sum(axis=1)<=m_i)

@njit
def weekly_quota_rh(m_i, q_ic, X, s, u, f_d):
    return np.all((q_ic * X[:,u,:,:f_d].sum(1).sum(1)).sum(axis=1) + (q_ic * s[:,u,:,f_d:].sum(1).sum(1)).sum(axis=1)<=m_i)

@njit
def daily_quota(n_i, q_ic, X, u, d):
    return np.all((q_ic * X[:,u,:,d].sum(axis=(1))).sum(axis=1)<=n_i)

@njit
def channel_capacity(t_hd, X, h, d):
    return X[:,:,h,d].sum() <= t_hd[h,d]

@njit
def check(X, b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd, indicies):
    c_i = 0
    u_i = 1
    h_i = 2
    d_i = 3
    if not eligibility(e_cu, X, indicies[c_i],indicies[u_i],indicies[h_i],indicies[d_i]):
        return False
    for f_d in range(1, cuhd[d_i]+1):
        if not weekly_limitation_rh(b, X, s_cuhd, indicies[u_i], f_d):
            return False
    if not daily_limitation(k, X, indicies[u_i],indicies[d_i]):
        return False
    for f_d in range(1, cuhd[d_i]+1):
        if not campaign_limitation_rh(l_c, X, s_cuhd, indicies[c_i],indicies[u_i], f_d):
            return False
    for f_d in range(1, cuhd[d_i]+1):
        if not weekly_quota_rh(m_i, q_ic, X, s_cuhd, indicies[u_i], f_d):
            return False
    if not daily_quota(n_i, q_ic, X, indicies[u_i],indicies[d_i]):
        return False
    if not channel_capacity(t_hd, X, indicies[h_i],indicies[d_i]):
        return False
    return True

@njit
def greedy(X_cuhd:np.ndarray, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd):
    for c in range(cuhd[0]):
        for d in range(cuhd[3]):
            for u in range(cuhd[1]):
                for h in range(cuhd[2]):
                    _rh(m_i,q_ic,X_cuhd,None,u,1)
                    #X_cuhd[c,u,h,d]=1
                    #if not check(X_cuhd, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                    #    X_cuhd[c,u,h,d]=0

@njit
def check_on(X, b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd, indicies):
    c_i = 0
    u_i = 1
    h_i = 2
    d_i = 3
    if not eligibility(e_cu, X, indicies[c_i],indicies[u_i],indicies[h_i],indicies[d_i]):
        return False
    for f_d in range(1, cuhd[d_i]+1):
        if not weekly_limitation_rh(b, X, s_cuhd, indicies[u_i], f_d):
            return False
    if not daily_limitation(k, X, indicies[u_i],indicies[d_i]):
        return False
    for f_d in range(1, cuhd[d_i]+1):
        if not campaign_limitation_rh(l_c, X, s_cuhd, indicies[c_i],indicies[u_i], f_d):
            return False
    for f_d in range(1, cuhd[d_i]+1):
        if not weekly_quota_rh(m_i, q_ic, X, s_cuhd, indicies[u_i], f_d):
            return False
    if not daily_quota(n_i, q_ic, X, indicies[u_i],indicies[d_i]):
        return False
    if not channel_capacity(t_hd, X, indicies[h_i],indicies[d_i]):
        return False
    return True

@njit
def greedy_on(X, b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd):
    for c in range(cuhd[0]):
        for d in range(cuhd[3]):
            for u in range(cuhd[1]):
                for h in range(cuhd[2]):
                    X[c,u,h,d]=1
                    if not check_on(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                        X[c,u,h,d]=0
@njit
def do_check(X, b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd):
    for c in range(cuhd[0]):
        print(f"C{c}")
        for d in range(cuhd[3]):
            for u in range(cuhd[1]):
                for h in range(cuhd[2]):
                    if not check_on(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                        return False
    return True

@njit
def objective_fn_no_net(rp_c, X):
    return (rp_c * X.sum(1).sum(1).sum(1)).sum()

def objective_fn_no_net2(rp_c, X):
    return np.matmul(rp_c, X.sum(axis=(1,2,3)))

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.DEBUG)

np.random.seed(13)
r_p = np.random.choice(100, 3) #r_p = np.ones(P, dtype='int8')
rp_c = np.array([r_p[r] for r in np.random.choice(3, 10)])
X = np.ones((10,1000,3,7))
print(objective_fn_no_net(rp_c, X))
print(objective_fn_no_net2(rp_c, X))


#solution = Solution("NUMBA TEST")
#PMS0 = solution.generate_parameters(Case({"C":2,"U":10,"H":3, "D":7, "I":3, "P":3}))
#X_cuhd = np.zeros(PMS0.cuhd, dtype='int')
#greedy_on(X_cuhd, PMS0.b, PMS0.cuhd, PMS0.e_cu, PMS0.k, PMS0.l_c, PMS0.m_i, PMS0.n_i, PMS0.q_ic, PMS0.s_cuhd, PMS0.t_hd)
#print("STARITING GAME")
#
#PMS1 = solution.generate_parameters(Case({"C":10,"U":500,"H":3, "D":7, "I":3, "P":3}))
#X_cuhd = np.zeros(PMS1.cuhd, dtype='int')
#start = time()
#greedy_on(X_cuhd, PMS1.b, PMS1.cuhd, PMS1.e_cu, PMS1.k, PMS1.l_c, PMS1.m_i, PMS1.n_i, PMS1.q_ic, PMS1.s_cuhd, PMS1.t_hd)
#end = time()
#print(end - start)
#
#start = time()
#greedy_on(X_cuhd, PMS1.b, PMS1.cuhd, PMS1.e_cu, PMS1.k, PMS1.l_c, PMS1.m_i, PMS1.n_i, PMS1.q_ic, PMS1.s_cuhd, PMS1.t_hd)
#end = time()
#print(end - start)
#
#start = time()
#do_check(X_cuhd, PMS1.b, PMS1.cuhd, PMS1.e_cu, PMS1.k, PMS1.l_c, PMS1.m_i, PMS1.n_i, PMS1.q_ic, PMS1.s_cuhd, PMS1.t_hd)
#end = time()
#print(end - start)