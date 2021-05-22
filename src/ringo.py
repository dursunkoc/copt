import numpy as np
from time import time
C=100
U=10000
H=3
D=7
X_cuhd = np.zeros((C,U,H,D))
start_time = time()
for c in range(C):
    for u in range(U):
        for h in range(H):
            for d in range(D):
                X_cuhd[c,u,h,d] = 1
end_time = time()
duration = end_time - start_time
print(f"duration: {duration}")
print(X_cuhd)
start_time = time()
for j in range(C*U*H*D):
    j_d = j%D
    j_d_r = j//D
    j_h = j_d_r%H
    j_h_r = j_d_r//H
    j_u = j_h_r%U
    j_u_r = j_h_r//U
    j_c = j_u_r%C
    j_c_r = j_u_r//C
    X_cuhd[j_c,j_u,j_h,j_d] = 2
end_time = time()
duration = end_time - start_time
print(f"duration: {duration}")
print(X_cuhd)