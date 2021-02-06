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
            if not res:
                return res
        return True

