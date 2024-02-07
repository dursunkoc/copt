import multiprocessing
import numpy as np
from numba import njit, prange, objmode
import logging
import time

logger = logging.getLogger("CONSTRAINTS-LOGGER")
logger.setLevel(logging.INFO)

import sys
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S'))
logger.addHandler(log_handler)

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
def check_indicies(X, b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd, indicies):
    c_i = 0
    u_i = 1
    h_i = 2
    d_i = 3
    if not eligibility(e_cu, X, indicies[c_i],indicies[u_i],indicies[h_i],indicies[d_i]):
        return False
#    for f_d in range(1, cuhd[d_i]+1):
#        if not weekly_limitation_rh(b, X, s_cuhd, indicies[u_i], f_d):
#            return False
#    if not daily_limitation(k, X, indicies[u_i],indicies[d_i]):
#        return False
#    for f_d in range(1, cuhd[d_i]+1):
#        if not campaign_limitation_rh(l_c, X, s_cuhd, indicies[c_i],indicies[u_i], f_d):
#            return False
#    for f_d in range(1, cuhd[d_i]+1):
#        if not weekly_quota_rh(m_i, q_ic, X, s_cuhd, indicies[u_i], f_d):
#            return False
#    if not daily_quota(n_i, q_ic, X, indicies[u_i],indicies[d_i]):
#        return False
    if not channel_capacity(t_hd, X, indicies[h_i],indicies[d_i]):
        return False
    return True

@njit
def do_check_all(X, b,cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd):
    for c in range(cuhd[0]):
        for d in range(cuhd[3]):
            for u in range(cuhd[1]):
                for h in range(cuhd[2]):
                    if not check_indicies(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                        return False
    return True

@njit
def greedy_on(X, rp_c, b, cuhd,e_cu,k,l_c,m_i,n_i,q_ic,s_cuhd,t_hd):
    for c in np.argsort(-(rp_c)):
        for d in range(cuhd[3]):
            for u in range(cuhd[1]):
                for h in range(cuhd[2]):
                    X[c,u,h,d]=1
                    if not check_indicies(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                        X[c,u,h,d]=0

@njit(parallel=True)
def do_greedy_loop(X, sorted_camps, D, H, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd):
    for c_i in prange(sorted_camps.size):
        c = sorted_camps[c_i]
        for d in prange(D):
            Ur = X.sum(0).sum(1).sum(1).argsort()
            overall= sorted_camps.size*D*Ur.size*H
            for h in prange(H):
                with objmode():
                    print(f"-->C:{c_i}, D:{d}, H:{h}  @{time.time()}")
                for u_i in prange(Ur.size):
                    u = Ur[-u_i-1]
                    X[c,u,h,d]=1
                    if not check_indicies(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                        X[c,u,h,d]=0
                with objmode():
                    print(f"<--C:{c_i}, D:{d}, H:{h}  @{time.time()}")

@njit(parallel=True)
def do_greedy_loop_for_net(X, ee_cu, D, H, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, U_ranges):
    for c in np.argsort(-ee_cu.sum(1)):
        for u in U_ranges[c]:
            for d in range(D):
                for h in range(H):
                    if(ee_cu[c][u]>0):
                        X[c,u,h,d]=1
                        if not check_indicies(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                            X[c,u,h,d]=0

@njit(parallel=True)
def do_greedy_loop_rnd(X, sorted_camps, D, H, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd):
    for c_i in prange(sorted_camps.size):
        c = sorted_camps[c_i]
        for d in prange(D):
            Ur = X.sum(0).sum(1).sum(1).argsort()
            for h in prange(H):
                with objmode():
                    print(f"-->C:{c_i}, D:{d}, H:{h}  @{time.time()}")
                for u_i in prange(Ur.size):
                    u = Ur[-u_i-1]
                    X[c,u,h,d]=1
                    if not check_indicies(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (c,u,h,d)):
                        X[c,u,h,d]=0
                with objmode():
                    print(f"<--C:{c_i}, D:{d}, H:{h}  @{time.time()}")


def do_greedy_loop_in_parallel(X, sorted_camps, D, H, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, sem_count):
    semaphore = multiprocessing.Semaphore(sem_count)
    jobs = []
    number_of_camps = sorted_camps.size
    for camp in sorted_camps:
        p = multiprocessing.Process(target=do_greedy_loop_for_camp, args=(X, number_of_camps, camp, D, H, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, semaphore))
        jobs.append(p)
        p.start()
    print("\nstarted all!")
    for proc in jobs:
        proc.join()

def do_greedy_loop_for_camp(X, number_of_camps, camp, D, H, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, semaphore):
    with semaphore:
        index=0
        for d in range(D):
            Ur = X.sum(0).sum(1).sum(1).argsort()
            overall= number_of_camps*D*Ur.size*H
            for u in Ur[::-1]:
                for h in range(H):
                    index=index+1
                    if index % 10000 == 0:
                        print(f"{index}/{overall}@{time.time()}")
                    X[camp,u,h,d]=1
                    if not check_indicies(X, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, (camp,u,h,d)):
                        X[camp,u,h,d]=0


@njit(parallel=True)
def fn_fix_to_fitness(population, rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, e_cu_X):
    pop_size = len(population)
    for index in prange(pop_size):
        if not X_check(population[index], e_cu_X):
            descrease_to_fit(population[index], rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, e_cu_X)
        increase_to_fit(population[index], rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd)

@njit
def increase_to_fit(indiv, rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd):
    D=cuhd[3]
    H=cuhd[2]
    U=cuhd[1]
    for c in np.argsort(-rp_c):
        for d in range(D):
            for h in range(H):
                for u in range(U):
                    if indiv[c,u,h,d]==0:
                        indiv[c,u,h,d]=1
                        if not check_indicies(indiv, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd,(c,u,h,d)):
                            indiv[c,u,h,d]=0

@njit
def descrease_to_fit(indiv, rp_c, b, cuhd, e_cu, k, l_c, m_i, n_i, q_ic, s_cuhd, t_hd, e_cu_X):
    D=cuhd[3]
    H=cuhd[2]
    U=cuhd[1]
    for c in np.argsort(rp_c):
        for d in range(D):
            for h in range(H):
                for u in range(U):
                    if indiv[c,u,h,d]==1:
                        indiv[c,u,h,d]=0
                        if X_check(indiv, e_cu_X):
                            return

#Bulk Constraint Functions
@njit
def X_eligibility (e_cu_X, X):
    return (X <= e_cu_X).all()
def X_weekly_limitation (self, b, X):
    return (X.sum(axis=(0,2,3))<=b).all()
def X_weekly_limitation_rh (self, b, X):
    return (X.sum(axis=(0,2,3))<=b).all()
def X_daily_limitation (self, k, X):
    return (X.sum(axis=(0)).sum(axis=(1))<=k).all()
def X_campaign_limitation (self, l_c, X):
    return np.all(X.sum(axis=(2,3)).T<=l_c)
def X_weekly_quota (self, m_i, q_ic, X):
    q_range = range(q_ic.shape[0])
    for i in q_range:
        quota_i = np.all(X.sum(axis=(2,3)).T * q_ic[i, ].T  <= m_i[i])
        if not quota_i:
            return False
    return True
def X_channel_capacity (self, t_hd, X):
    return np.all(X.sum(axis=(0,1))<=t_hd)

def X_daily_quota (self, n_i, q_ic, X):
    q_range = range(q_ic.shape[0])
    for i in q_range:
        quota_i = np.all((q_ic[i,].T * X.sum(axis=2).T).sum(2) <= n_i[i])
        if not quota_i:
            return False
    return True

@njit
def X_check(X, e_cu_X):
    if not X_eligibility(e_cu_X, X):
        return False
#    if not X_weekly_limitation(b, X):
#        return False
#    if not X_daily_limitation(k, X):
#        return False
#    if not X_campaign_limitation(l_c, X):
#        return False
#    if not X_weekly_quota(m_i, q_ic, X):
#        return False
#    if not X_daily_quota(n_i, q_ic, X):
#        return False
#    if not X_channel_capacity(t_hd, X):
#        return False
    return True
