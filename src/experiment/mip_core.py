from docplex.mp.model import Model

class MipCore:
    def mip_eligibility(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints(
            (X_cuhd[(c,u,h,d)] <= PMS.e_cu[c,u]
            for c in range(0,C)
            for u in range(0,U) 
            for h in range(0,H) 
            for d in range(0,D)))

    def mip_daily_communication(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]  
                    for c in range(0,C) 
                    for h in range(0,H)) <= PMS.k)
                for d in range(0,D)
                for u in range(0,U)))

    def mip_weekly_communication_rh(self, mdl, X_cuhd, PMS, C, U, H, D, f_d):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)] if d<f_d else PMS.s_cuhd[(c,u,h,d)]
                for d in range(0,D) 
                for c in range(0,C) 
                for h in range(0,H)) <= PMS.b)
            for u in range(0,U)))

    def mip_weekly_communication(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]
                for d in range(0,D) 
                for c in range(0,C) 
                for h in range(0,H)) <= PMS.b)
            for u in range(0,U)))

    def mip_campaign_communication_rh(self, mdl, X_cuhd, PMS, C, U, H, D, f_d):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)] if d< f_d else PMS.s_cuhd[(c,u,h,d)]
                for d in range(0,D)
                for h in range(0,H)) <= PMS.l_c[c] )
            for c in range(0,C)
            for u in range(0,U)))

    def mip_campaign_communication(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]
                for d in range(0,D)
                for h in range(0,H)) <= PMS.l_c[c] )
            for c in range(0,C)
            for u in range(0,U)))

    def mip_weekly_quota_rh(self, mdl, X_cuhd, PMS, C, U, H, D, I, f_d):
        return mdl.add_constraints((
            (mdl.sum( (X_cuhd[(c,u,h,d)] if d < f_d else PMS.s_cuhd[(c,u,h,d)]) * PMS.q_ic[i,c]
                for d in range(0,D)
                for c in range(0,C)
                for h in range(0,H)) <= PMS.m_i[i])
            for u in range(0,U)
            for i in range(0,I)))

    def mip_weekly_quota(self, mdl, X_cuhd, PMS, C, U, H, D, I):
        return mdl.add_constraints((
            (mdl.sum( (X_cuhd[(c,u,h,d)]) * PMS.q_ic[i,c]
                for d in range(0,D)
                for c in range(0,C)
                for h in range(0,H)) <= PMS.m_i[i])
            for u in range(0,U)
            for i in range(0,I)))

    def mip_daily_quota(self, mdl, X_cuhd, PMS, C, U, H, D, I):
        return mdl.add_constraints((
                (mdl.sum(X_cuhd[(c,u,h,d)]*PMS.q_ic[i,c]
                    for c in range(0,C) 
                    for h in range(0,H)) <= PMS.n_i[i])
                for u in range(0,U)
                for d in range(0,D)
                for i in range(0,I)))

    def mip_channel_capacity(self, mdl, X_cuhd, PMS, C, U, H, D):
        return mdl.add_constraints((
            (mdl.sum(X_cuhd[(c,u,h,d)]
                for u in range(0,U) 
                for c in range(0,C)) <= PMS.t_hd[h,d])
            for h in range(0,H)
            for d in range(0,D)))

    def mip_network_coverage(self, mdl, Y_cuhd, X_cuhd, a_uv, C, U, H, D):
        return mdl.add_constraints(
            Y_cuhd[(c,u)] <= mdl.sum((X_cuhd[(c,u,h,d)]) + 
            (mdl.sum(X_cuhd[(c,v,h,d)] * a_uv[u,v] for v in range(0,U)))
                for h in range(0,H)
                for d in range(0,D))
            for c in range(0,C)
            for u in range(0,U)
        )

    def start_trivial_model(self, binary:bool, PMS, C, U, H, D, I):
        mdl = Model(name='Campaign Optimization With Trivial Solution')
        #variables
        if binary:
            X = {(c,u,h,d): mdl.binary_var(f"X[{c},{u},{h},{d}]")
                for c in range(0,C)
                for u in range(0,U)
                for h in range(0,H)
                for d in range(0,D)}
            if PMS.a_uv is not None:
                Y = {(c,u): mdl.binary_var(f"Y[{c},{u}]")
                    for c in range(0,C)
                    for u in range(0,U)}
        else:
            X = {(c,u,h,d): mdl.continuous_var(lb=0, ub=1, name=f"X[{c},{u},{h},{d}]")
            for c in range(0,C)
            for u in range(0,U) 
            for h in range(0,H)
            for d in range(0,D)}
            if PMS.a_uv is not None:
                Y = {(c,u): mdl.continuous_var(lb=0, ub=1, name=f"Y[{c},{u}]")
                for c in range(0,C)
                for u in range(0,U)}
        #objectivefunction
        if PMS.a_uv is None:
            maximize = mdl.maximize(mdl.sum([X[(c,u,h,d)] * PMS.rp_c[c]
                    for c in range(0,C)
                    for u in range(0,U) 
                    for h in range(0,H) 
                    for d in range(0,D)]))
        else:
            maximize = mdl.maximize(mdl.sum([Y[(c,u)] * PMS.rp_c[c]
                    for c in range(0,C)
                    for u in range(0,U)]))
        #constraints
        eligibilitiy = self.mip_eligibility(mdl, X, PMS, C, U, H, D)
        channel_capacity = self.mip_channel_capacity(mdl, X, PMS, C, U, H, D)
        if PMS.a_uv is not None:
            network_coverage = self.mip_network_coverage(mdl, Y, X, PMS.a_uv, C, U, H, D)
        return (mdl, X)

    def print_constraint(self, constraint):
        for c in constraint:
            print(c)

    def start_model(self, binary:bool, PMS, C, U, H, D, I, V_cuhd=None):
        mdl = Model(name='Campaign Optimization')
        print("Starting model")
        #variables
        if binary:
            X = {(c,u,h,d): mdl.binary_var(f"X_c:{c}_u:{u}_h:{h}_d:{d}")
                for c in range(0,C)
                for u in range(0,U)
                for h in range(0,H)
                for d in range(0,D)}
            if PMS.a_uv is not None:
                Y = {(c,u): mdl.binary_var(f"Y_c:{c}_u:{u}")
                    for c in range(0,C)
                    for u in range(0,U)}
        else:
            X = {(c,u,h,d): mdl.continuous_var(lb=0, ub=1, name=f"X_c:{c}_u:{u}_h:{h}_d:{d}")
            for c in range(0,C)
            for u in range(0,U) 
            for h in range(0,H)
            for d in range(0,D)}
            if PMS.a_uv is not None:
                Y = {(c,u): mdl.continuous_var(lb=0, ub=1, name=f"Y_c:{c}_u:{u}")
                for c in range(0,C)
                for u in range(0,U)}
        print("Variables Done!")
        #objectivefunction
        if PMS.a_uv is None:
            maximize = mdl.maximize(mdl.sum([X[(c,u,h,d)] * PMS.rp_c[c]
                    for c in range(0,C)
                    for u in range(0,U) 
                    for h in range(0,H) 
                    for d in range(0,D)]))
        else:
            maximize = mdl.maximize(mdl.sum([Y[(c,u)] * PMS.rp_c[c]
                    for c in range(0,C)
                    for u in range(0,U)]))
        print("Objective Done!")
        #constraints
        eligibilitiy = self.mip_eligibility(mdl, X, PMS, C, U, H, D)
        print("eligibilitiy Done!")
        if PMS.s_cuhd is not None:
            weekly_communication = [self.mip_weekly_communication_rh(mdl, X, PMS, C, U, H, D, f_d) for f_d in range(1, D+1)]
            campaign_communication = [self.mip_campaign_communication_rh(mdl, X, PMS, C, U, H, D, f_d) for f_d in range(1, D+1)]
            weekly_quota = [self.mip_weekly_quota_rh(mdl, X, PMS, C, U, H, D, I, f_d) for f_d in range(1, D+1)]
        else:
            weekly_communication = self.mip_weekly_communication(mdl, X, PMS, C, U, H, D)
            campaign_communication = self.mip_campaign_communication(mdl, X, PMS, C, U, H, D)
            weekly_quota = self.mip_weekly_quota(mdl, X, PMS, C, U, H, D, I)
        print("weekly_quota, campaign_communication, weekly_communication  Done!")

        daily_communication = self.mip_daily_communication(mdl, X, PMS, C, U, H, D)
        print("daily_communication  Done!")
        daily_quota = self.mip_daily_quota(mdl, X, PMS, C, U, H, D, I)
        print("daily_quota  Done!")
        channel_capacity = self.mip_channel_capacity(mdl, X, PMS, C, U, H, D)
        print("channel_capacity  Done!")
        if PMS.a_uv is not None:
            network_coverage = self.mip_network_coverage(mdl, Y, X, PMS.a_uv, C, U, H, D)
        print("network_coverage  Done!")
        if V_cuhd is not None:
            for c in range(C):
                for d in range(D):
                    for h in range(H):
                        for u in range(U):
                            if V_cuhd[c,u,h,d]==1:
                                self.fix_variable(mdl, X, c, u, h, d)
                            else:
                                self.fix_variable_0(mdl, X, c, u, h, d)
        print("ALL Done!")
        return (mdl, X)

    def fix_variable_0(self, mdl, X, c, u, h, d):
        return mdl.add_constraint(X[(c,u,h,d)] == 0)

    def fix_variable(self, mdl, X, c, u, h, d):
        return mdl.add_constraint(X[(c,u,h,d)] == 1)

    def unfix_variable(self, mdl, cons):
        mdl.remove_constraint(cons)
    
    def is_feasible(self, mdl:Model):
        result = mdl.solve(log_output=False)
        return result is not None