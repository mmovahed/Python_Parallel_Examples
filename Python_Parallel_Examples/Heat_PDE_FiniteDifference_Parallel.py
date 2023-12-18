import numpy as np
import numba as nb
from numba import set_num_threads, get_num_threads , config
import time

# Define the parallel function using numba JIT decorator
@nb.njit(parallel=True)
def solve_heat_pde_parallel(u, dt, dx, dy, nt, nx, ny, alpha):
    for n in nb.prange(nt):  # Parallelize over time
        un = u.copy()
        for i in nb.prange(1, nx-1):  # Parallelize over x
            for j in nb.prange(1, ny-1):  # Parallelize over y
                u[i, j] = un[i, j] + alpha * (un[i+1, j] - 2*un[i, j] + un[i-1, j] + un[i, j+1] - 2*un[i, j] + un[i, j-1])
        # Boundary conditions
        u[0, :] = 0
        u[nx-1, :] = 0
        u[:, 0] = 0
        u[:, ny-1] = 0
    
    return u


# Grid parameters
nx = 100  # Number of grid points in x
ny = 100  # Number of grid points in y
dx = 1.0 / (nx - 1)  # Grid spacing in x
dy = 1.0 / (ny - 1)  # Grid spacing in y
    
# Time step size
nt = 1000
    
# Diffusion coefficient
alpha = 0.05
    
# Initial conditions
u = np.zeros((nx, ny))
u[int(0.5/dx):int(1/dx+1), int(0.5/dy):int(1/dy+1)] = 2


set_num_threads(1)
time2 = time.perf_counter()

# Call the serial solver function
u = solve_heat_pde_parallel(u, dt=0.001, dx=dx, dy=dy, nt=nt, nx=nx, ny=ny, alpha=alpha)
    
print("serial time : ", time.perf_counter() - time2)

u2 = np.zeros((nx, ny))
u2[int(0.5/dx):int(1/dx+1), int(0.5/dy):int(1/dy+1)] = 2
set_num_threads(2)
time2 = time.perf_counter()

# Call the parallel solver function
u2 = solve_heat_pde_parallel(u2, dt=0.001, dx=dx, dy=dy, nt=nt, nx=nx, ny=ny, alpha=alpha)

print("Parallel time : ", time.perf_counter() - time2)

print(u)
print("-------------")
print(u2)

import matplotlib.pyplot as plt
f, axarr = plt.subplots(1,2)
axarr[0].imshow(u)
axarr[1].imshow(u2)

plt.show()
