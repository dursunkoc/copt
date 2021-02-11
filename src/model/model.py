class Model:
    def __init__(self, constraints, obj_fn):
        self.constraints = constraints
        self.obj_fn = obj_fn
        
    def execute(self, X, indicies):
        for c in self.constraints:
            if not c.execute(X, indicies):
                return False
        return True

    def calc_value(self, X):
        return self.obj_fn(X)
    
    def stats(self):
        return [(c.name, c.get_exec_count()) for c in self.constraints]