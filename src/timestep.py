import numpy as np
from states import State, get_state_from_array
import params
from physics import speed_of_sound
from pycfd_types import Array, real_t, VarIndex

IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI = VarIndex.__members__

def cell_timestep(q: State) -> float:
    dx = params.dx
    dy = params.dy
    cs = speed_of_sound(q)
    inv_dt = (cs + abs(q[IU])) / dx + (cs + abs(q[IV])) / dy
    gr = cs**2 * q[IR]
    Bt2 = [q[IBY]**2 + q[IBZ]**2, q[IBX]**2 + q[IBZ]**2, q[IBX]**2 + q[IBY]**2]
    b2 = q[IBX]**2 + q[IBY]**2 + q[IBZ]**2
    cf1 = gr - b2
    V = [q[IU], q[IV], q[IW]]
    D = [dx, dy]
    for i in range(2):
        cf2 = gr + b2 + np.sqrt(cf1**2 + 4 * gr * Bt2[i])
        cf = np.sqrt(0.5 * cf2 / q[IR])
        cmax = np.max(abs(V[i] - cf), abs(V[i] + cf))
        inv_dt = max(inv_dt, cmax/D[i])
    return inv_dt


def compute_dt(Q: Array) -> real_t:
    all_inv_dt = 0.0
    for i in range(params.Ntx):
        for j in range(params.Nty):
            q_loc = get_state_from_array(Q, i, j)
            all_inv_dt = max(cell_timestep(q_loc), all_inv_dt)
    return params.cfl / np.max(all_inv_dt)