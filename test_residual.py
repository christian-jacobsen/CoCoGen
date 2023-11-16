import torch
import numpy as np
import matplotlib.pyplot as plt
from physics.operators import ViscousBurgers
from physics.operators import DarcyFlow

P = np.load("datasets/Darcy_n16/data/P_0.npy")
K = np.load("datasets/Darcy_n16/permeability/K_0.npy")

P = torch.from_numpy(P).unsqueeze(0)
K = torch.from_numpy(K).unsqueeze(0)

dx = 0.015625
eps = 1e-12

residual = DarcyFlow(dx, eps, 0.0, 0.0, 0.0, 0.0)

res, d2pdx2, d2pdy2, dkdx, dpdx, dkdy, dpdy = residual.r_diff(P, K)

P_r = P + 0.0
n=2000
for i in range(n):
    drdp = residual.drdp(P, K)
    P_r = P_r - eps*drdp
res_r, _, _, _, _, _, _ = residual.r_diff(P_r, K)

print("Residual L2 norm before: ", torch.mean(res**2))
print("Residual L2 norm after: ", torch.mean(res_r**2))

plt.figure(1)
plt.imshow(P[0])
plt.savefig("p_test.png", bbox_inches='tight', dpi=500)
plt.figure(2)
plt.imshow(dpdy[0])
plt.savefig("dpdy_test.png", bbox_inches='tight', dpi=500)
plt.figure(3)
plt.imshow(dpdx[0])
plt.savefig("dpdx_test.png", bbox_inches='tight', dpi=500)
plt.figure(7)
plt.imshow(d2pdx2[0])
plt.savefig("d2pdx2_test.png", bbox_inches='tight', dpi=500)
plt.figure(8)
plt.imshow(d2pdy2[0])
plt.savefig("d2pdy2_test.png", bbox_inches='tight', dpi=500)

plt.figure(4)
plt.imshow(K[0])
plt.savefig("k_test.png", bbox_inches='tight', dpi=500)
plt.figure(5)
plt.imshow(dkdy[0])
plt.savefig("dkdy_test.png", bbox_inches='tight', dpi=500)
plt.figure(6)
plt.imshow(dkdx[0])
plt.savefig("dkdx_test.png", bbox_inches='tight', dpi=500)

plt.figure(9)
plt.imshow(P_r[0])
plt.savefig("p_plus_r_test.png", bbox_inches='tight', dpi=500)

'''
nu = 0.01
dx = 0.03125
dt = 0.01
mu = -0.751762367
sigma = 8.041401807

residual = ViscousBurgers(nu, dx, dt, mu, sigma)

y = np.load("datasets/Burgers1D/Nu0.01_64/data/u_1.npy")
u = (y-mu)/sigma
u = torch.from_numpy(u).unsqueeze(0).unsqueeze(0)

print('y: ', y)
print('residual: ', residual(u))
print('norm of unscaled residual: ', torch.mean(residual(u)**2))
print('norm of scaled residual: ', torch.mean(((residual(u)-mu)/sigma)**2))
'''
