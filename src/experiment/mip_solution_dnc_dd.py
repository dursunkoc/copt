from math import ceil
from docplex.mp.model import Model

class MySol():
    def start_partial_models(self, binary:bool, PMS, C, U, H, D, I, V_cuhd=None):
        batch_size = 3000
        number_of_models = ceil(U / batch_size)
        models = []
        for model_i in range(0, number_of_models):
            mdl = Model(name=f'Partial Campaign Optimization[{model_i}]')
            start = model_i * batch_size
            end = ((model_i + 1) * batch_size)
            print(range(start, end))
            models.append(mdl)
        return models
    

if __name__ == '__main__':
    m = MySol()
    mdls = m.start_partial_models(True, None, 10, 450000, 3, 7, 3, None )
#    for i in mdls:
#        print(i)
