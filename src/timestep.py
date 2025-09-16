import numpy as np
from src.states import State, get_state_from_array
import src.params as params
from src.physics import speed_of_sound
from src.pycfd_types import Array, real_t
from src.varindexes import IR, IU, IV, IBX, IBY, IBZ
# IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI = VarIndex.__members__

def cell_timestep(q: State) -> real_t:
    cs = speed_of_sound(q)
    inv_dt_hyp_loc = (cs + abs(q[IU]))/params.dx + (cs + abs(q[IV]))/params.dy
    #ifdef MHD
    if params.MHD:
        B2: real_t = q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]
        ca2: real_t = B2 / q[IR]
        cap2x: real_t = q[IBX]*q[IBX]/q[IR]
        cap2y: real_t = q[IBY]*q[IBY]/q[IR]
        # cap2z = q[IBZ]*q[IBZ]/q[IR]

        cf_x: real_t = np.sqrt(0.5*(cs**2+ca2)+0.5*np.sqrt((cs**2+ca2)*(cs**2+ca2)-4.0*cs**2*cap2x))
        cf_y: real_t = np.sqrt(0.5*(cs**2+ca2)+0.5*np.sqrt((cs**2+ca2)*(cs**2+ca2)-4.0*cs**2*cap2y))
        # real_t c_jz = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2z))
        inv_dt_hyp_loc = max((cf_x + abs(q[IU])) / params.dx + (cf_y + abs(q[IV])) / params.dy, inv_dt_hyp_loc)
    return inv_dt_hyp_loc


def compute_dt(Q: Array, t: real_t, verbose: bool) -> real_t:
    all_inv_dt = 0.0
    for (i, j) in params.range_dom:
        q_loc = get_state_from_array(Q, i, j)
        all_inv_dt = max(cell_timestep(q_loc), all_inv_dt)
    if verbose: 
      print(f"Computing dts at ({t=:.2f}): dt_hyp={params.CFL/all_inv_dt}")
    return params.CFL / np.max(all_inv_dt)
