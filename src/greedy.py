class Constraint:
    def __init__(self, name, fnc, fixed_indicies):
        self.fnc = fnc
        self.name = name
        self.fixed_indicies = fixed_indicies
        self.exec_count = 0

    def execute(self, X, indicies):
        self.exec_count = self.exec_count + 1
        params = tuple([X]) + tuple([indicies[fi] for fi in self.fixed_indicies])
        return self.fnc(*params)
    
    def get_exec_count(self):
        return self.exec_count


class Model:
    def __init__(self, constraints):
        self.constraints = constraints
        
    def execute(self, X, indicies):
        for c in self.constraints:
            if not c.execute(X, indicies):
                return False
        return True
    
    def stats(self):
        return [(c.name, c.get_exec_count()) for c in self.constraints]
