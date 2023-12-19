from mpi4py import MPI
import numpy as np
#import matplotlib.pyplot as plt
import time

# mpiexec -n 4 py Heat_PDE_FiniteDifference_DM.py


def solve_fractional_pde(alpha, nx, nt, tend):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    dx = 1.0 / nx
    dt = tend / nt

    x = np.linspace(rank * dx, (rank + 1) * dx, nx)
    u = np.sin(np.pi * x)
    
    for t in range(nt):
        if rank == 0:
            comm.Send(u[-1], dest=rank+1)
            comm.Recv(u[-1], source=rank+1)
        elif rank == size - 1:
            comm.Recv(u[0], source=rank-1)
            comm.Send(u[0], dest=rank-1)
        else:
            comm.Sendrecv(u[-1], dest=rank+1, recvbuf=u[0], source=rank-1)
            comm.Sendrecv(u[0], dest=rank-1, recvbuf=u[-1], source=rank+1)
        
        u_new = np.zeros_like(u)
        for i in range(1, nx-1):
            u_new[i] = u[i] + alpha * dt * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        
        u = u_new

    u_all = comm.gather(u, root=0)
    if rank == 0:
        u_global = np.concatenate(u_all)
        return x, u_global
    else:
        return None, None


alpha = 0.5
nx = 1000
nt = 100
tend = 1.0


time2 = time.perf_counter()

x, u = solve_fractional_pde(alpha, nx, nt, tend)

print("Execution time : ", time.perf_counter() - time2)
    
#if x is not None:
#    plt.plot(x, u)
#    plt.xlabel('x')
#    plt.ylabel('u')
#    plt.title('Solution of Heat Equation')
#    plt.show()