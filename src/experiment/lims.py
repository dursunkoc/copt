import numpy as np
U=100
C=3
e_cu = np.random.choice(2,(C, U))
X = np.array([u for u in range(0,U)])
for c in range(C):
    e_u = e_cu[c]
    for e in e_u:
        print(e)