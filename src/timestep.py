import numpy as np
from states import State
import params
from physics import speed_of_sound


def cell_timestep(q: State) -> float:
    dx = params.dx
    dy = params.dy
    cs = speed_of_sound(q)
    inv_dt = (cs + abs(q.u)) / dx + (cs + abs(q.v)) / dy
    gr = cs**2 * q.r
    Bt2 = [q.by**2 + q.bz**2, q.bx**2 + q.bz**2, q.bx**2 + q.by**2]
    b2 = q.bx**2 + q.by**2 + q.bz**2
    cf1 = gr - b2
    V = [q.u, q.v, q.w]
    D = [dx, dy]
    for i in range(2):
        cf2 = gr + b2 + np.sqrt(cf1**2 + 4 * gr * Bt2[i])
        cf = np.sqrt(0.5 * cf2 / q.r)
        cmax = np.max(abs(V[i] - cf), abs(V[i] + cf))
        inv_dt = max(inv_dt, cmax/D[i])
    return inv_dt

def timestep(Q: np.ndarray) -> float:
    all_inv_dt = 0.0
    for i in range(params.Nx):
        for j in range(params.Ny):
            q = Q[i, j]
            all_inv_dt = max(cell_timestep(q), all_inv_dt)
    return params.cfl / np.max(all_inv_dt)