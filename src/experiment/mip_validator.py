from experiment import Solution,  SolutionResult, Case, Experiment, Parameters
import numpy as np
from time import time
from docplex.mp.model import Model

class MipValidator():
    def __init__(self) -> None:
        mdl = Model(name='Campaign Optimization')
        #variables
        V_cuhd = {(c,u,h,d): mdl.binary_var(f"X_c:{c}_u:{u}_h:{h}_d:{d}")
            for c in range(0,C)
            for u in range(0,U)
            for h in range(0,H)
            for d in range(0,D)}
        #objectivefunction
        maximize = mdl.maximize(mdl.sum([V_cuhd[(c,u,h,d)] * PMS.rp_c[c]
                  for c in range(0,C)
                  for u in range(0,U) 
                  for h in range(0,H) 
                  for d in range(0,D)]))
        #solution for greedy
        indicies = np.where(X_cuhd == 1)
        mdl.add_constraints((V_cuhd[(c,u,h,d)] == 1
                            for c in set(indicies[0])
                            for u in set(indicies[1])
                            for h in set(indicies[2])
                            for d in set(indicies[3])))
        #constraints
        eligibilitiy = self.mip_eligibility(mdl, V_cuhd, PMS, C, U, H, D)
        
        weekly_communication = [self.mip_weekly_communication_rh(mdl, V_cuhd, PMS, C, U, H, D, f_d) for f_d in range(1, D+1)]

        daily_communication = self.mip_daily_communication(mdl, V_cuhd, PMS, C, U, H, D)
        
        campaign_communication = [self.mip_campaign_communication_rh(mdl, V_cuhd, PMS, C, U, H, D, f_d) for f_d in range(1, D+1)]
        
        weekly_quota = [self.mip_weekly_quota_rh(mdl, V_cuhd, PMS, C, U, H, D, I, f_d) for f_d in range(1, D+1)]

        daily_quota = self.mip_daily_quota(mdl, V_cuhd, PMS, C, U, H, D, I)

        channel_capacity = self.mip_channel_capacity(mdl, V_cuhd, PMS, C, U, H, D)
        
        result = mdl.solve(log_output=False)
