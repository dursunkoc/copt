from typing import List, Dict, Tuple
import numpy as np
from tqdm import trange
from tqdm import tqdm

class Case:
    def __init__(self, arguments:Dict[str, int]):
        self.arguments = arguments
    
    def __str__(self):
        return str(self.arguments)

class Parameters:
    def __init__(self, e_cu, e_cu_X, q_ic, rp_c, b, k, l_c, m_i, n_i, m_i_X, n_i_X, t_hd):
        self.e_cu = e_cu
        self.e_cu_X = e_cu_X
        self.q_ic = q_ic
        self.rp_c = rp_c
        self.b = b
        self.k = k
        self.l_c = l_c
        self.m_i = m_i
        self.n_i = n_i
        self.m_i_X = m_i_X
        self.n_i_X = n_i_X
        self.t_hd = t_hd

class SolutionResult:
    def __init__(self, case: Case, value: float, duration:int):
        self.case = case
        self.value = value
        self.duration = duration
    def __str__(self):
        return f"<case: {self.case}, value: {self.value}, duration: {self.duration}>"


class Experiment:
    import numpy as np
    def __init__(self, cases: List[Case]):
        self.cases = cases
    
    def run_cases_with(self, solution) -> List[Tuple[Case, SolutionResult]]:
        return [solution.run(case) for case in tqdm(self.cases, "Cases")]

class Solution:
    c_i = 0
    u_i = 1
    h_i = 2
    d_i = 3
    import numpy as np
    def __init__(self, name: str):
        self.name = name

    def run(self, case:Case)->SolutionResult:
        #TODO solve with a solution algorith
        parameters = self.generate_parameters(case)
        print(parameters)
        duration = 10
        value = 2.1
        return SolutionResult(case, value, duration)

    def generate_parameters(self, case: Case) -> Parameters:
        import numpy as np
        np.random.seed(23)
        C = case.arguments["C"] # number of campaigns
        U = case.arguments["U"]  # number of customers.
        H = case.arguments["H"]  # number of channels.
        D = case.arguments["D"]  # number of planning days.
        I = case.arguments["I"]  # number of quota categories.
        P = case.arguments["P"]  # number of priority categories.
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
        e_cu_X = np.stack([np.stack([e_cu for _ in range(H)], axis=2) for _ in range(D)], axis=3)
        m_i_X = np.stack([m_i for _ in range(U)], axis=1)
        n_i_X = np.stack([n_i for _ in range(U)], axis=1)
        return Parameters(e_cu,e_cu_X,q_ic,rp_c,b,k,l_c,m_i, n_i, m_i_X, n_i_X, t_hd)

#Single Constraint Functions
    def eligibility(self, e_cu, X, c, u, h, d):
        return X[c,u,h,d]<=e_cu[c,u]
    def weekly_limitation(self, b, X, u):
        return X[:,u,:,:].sum() <= b
    def daily_limitation (self, k, X, u, d):
        return X[:,u,:,d].sum() <= k
    def campaign_limitation(self, l_c, X, c, u):
        return X[c,u,:,:].sum() <= l_c[c]
    def weekly_quota(self, m_i, q_ic, X, u):
        return all((q_ic * X[:,u,:,:].sum(axis=(1,2))).sum(axis=1)<=m_i)
    def daily_quota(self, n_i, q_ic, X, u, d):
        return all((q_ic * X[:,u,:,d].sum(axis=(1))).sum(axis=1)<=n_i)
    def channel_capacity(self, t_hd, X, h, d):
        return X[:,:,h,d].sum() <= t_hd[h,d]

    def check(self, X, PMS, indicies):
        if not self.eligibility(PMS.e_cu, X, indicies[self.c_i],indicies[self.u_i],indicies[self.h_i],indicies[self.d_i]):
            return False
        if not self.weekly_limitation(PMS.b, X, indicies[self.u_i]):
            return False
        if not self.daily_limitation(PMS.k, X, indicies[self.u_i],indicies[self.d_i]):
            return False
        if not self.campaign_limitation(PMS.l_c, X, indicies[self.c_i],indicies[self.u_i]):
            return False
        if not self.weekly_quota(PMS.m_i, PMS.q_ic, X, indicies[self.u_i]):
            return False
        if not self.daily_quota(PMS.n_i, PMS.q_ic, X, indicies[self.u_i],indicies[self.d_i]):
            return False
        if not self.channel_capacity(PMS.t_hd, X, indicies[self.h_i],indicies[self.d_i]):
            return False
        return True
    
    def objective_fn(self, rp_c, X):
        return np.matmul(rp_c, X.sum(axis=(1,2,3)))

#Bulk Constraint Functions
    def X_eligibility (self, e_cu_X, X):
        return (X <= e_cu_X).all()
    def X_weekly_limitation (self, b, X):
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




if __name__ == '__main__':
    cases = [
            Case({"C":1,"U":10,"H":3, "D":7}),
            Case({"C":1,"U":100,"H":3, "D":7}),
            Case({"C":1,"U":1000,"H":3, "D":7})
            ]
    expr = Experiment(cases)
    solutions = expr.run_cases_with(Solution("mip"))
    for solution in solutions:
        print(solution)