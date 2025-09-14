from dataclasses import dataclass
import numpy as np
from pycfd_types import Array, IDir, real_t, VarIndex
from states import State, get_state_from_array, swap_components, set_state_into_array, consToPrim
from riemann import riemann
import params

IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI = VarIndex.__members__
def reconstruct(Q: Array, slopes: Array, i: int, j: int, sign: real_t, idir: IDir) -> State:
    q: State = get_state_from_array(Q, i, j);
    slope: State = get_state_from_array(slopes, i, j);
    res: State = State()
    match (params.reconstruction):
      case "PLM": res = q + slope * sign * 0.5 # Piecewise Linear
      case "PCM_WB": # Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        # res[IP] = q[IP] + sign * q[IR] * getGravity(i, j, dir, params) * params.dy * 0.5;
        #ifdef MHD
        res[IW] = q[IW];
        res[IBX] = q[IBX];
        res[IBY] = q[IBY];
        res[IBZ] = q[IBZ];
        res[IPSI] = q[IPSI];
        #endif // MHD
      case _:  res = q; #// Piecewise Constant

    return swap_components(res, idir);


def minmod(dL: real_t, dR: real_t) -> real_t:
    if (dL*dR < 0.0):
        return 0.0;
    elif (abs(dL) < abs(dR)):
        return dL;
    else:
        return dR;



slopesX: Array = np.empty((params.Ntx, params.Nty, params.Nfields))
slopesY: Array = np.empty((params.Ntx, params.Nty, params.Nfields))

def compute_slopes(Q: Array) -> None:
    # res.device_params.ibeg-1, res.device_params.jbeg-1}, {res.device_params.iend+1, res.device_params.jend+1});
    for i in range(params.ibeg-1, params.iend+1):
        for j in range(params.jbeg-1, params.jend+1):
            for ivar in range(params.Nfields):
                dL: real_t = Q[i, j, ivar]   - Q[i-1, j, ivar];
                dR: real_t = Q[i+1, j, ivar] - Q[i, j, ivar];
                dU: real_t = Q(i, j, ivar)   - Q(i, j-1, ivar)
                dD: real_t = Q(i, j+1, ivar) - Q(i, j, ivar);

                slopesX[i, j, ivar] = minmod(dL, dR);
                slopesY[i, j, ivar] = minmod(dU, dD);


def compute_fluxes_and_update(self, Q: Array, Unew: Array, dt: real_t) -> None:
        # const real_t ch_derigs = params.GLM_scale * GLM_ch1/dt;
        # const real_t ch_dedner = 0.5 * params.CFL * fmin(params.dx, params.dy)/dt;
    def updateAlongDir(self, i: int, j: int, idir: IDir) :
        slopes: Array = self.slopesX if idir == IDir.IX else self.slopesY
        dxm: int = -1 if idir == IDir.IX else 0
        dxp: int =  1 if idir == IDir.IX else 0
        dym: int = -1 if idir == IDir.IY else 0
        dyp: int =  1 if idir == IDir.IY else 0
        
        qCL: State = reconstruct(Q, slopes, i, j, -1.0, idir)
        qCR: State = reconstruct(Q, slopes, i, j,  1.0, idir);
        qL:  State = reconstruct(Q, slopes, i+dxm, j+dym, 1.0, idir);
        qR:  State = reconstruct(Q, slopes, i+dxp, j+dyp, -1.0, idir);
        
    # Calculating flux left and right of the cell
        fluxL = riemann(qL, qCL);
        fluxR = riemann(qCR, qR);

        fluxL = swap_components(fluxL, dir);
        fluxR = swap_components(fluxR, dir);

        
        un_loc: State = get_state_from_array(Unew, i, j)
        un_loc += dt*(fluxL - fluxR)/(params.dx if idir == IDir.IX else params.dy)

        set_state_into_array(Unew, i, j, un_loc)
    
    for i in range(params.ibeg, params.iend):
        for j in range(params.jbeg, params.jend):

            updateAlongDir(i, j, IDir.IX)
            updateAlongDir(i, j, IDir.IY)

    Unew[i, j, IR] = max(params.smallr, Unew[i, j, IR])
  

def euler_step(Q: Array, Unew: Array, dt: real_t, ite: int) -> None:
    # // First filling up boundaries for ghosts terms
    # bc_manager.fillBoundaries(Q);
    # // Hyperbolic update
    if (params.reconstruction == "PLM"):
        compute_slopes(Q)
    
    compute_fluxes_and_update(Q, Unew, dt)


def update(Q: Array, Unew: Array, dt: real_t) -> None:
    if params.time_stepping == "euler":
        euler_step(Q, Unew, dt)
    elif params.time_stepping == "RK2":
        U0: Array = np.empty((params.Ntx, params.Nty, params.Nfields));
        Ustar: Array = np.empty((params.Ntx, params.Nty, params.Nfields))

    # Step 1
    U0 = np.copy(Unew)
    Ustar = np.copy(Unew)
    euler_step(Q, Ustar, dt);
    # Step 2
    Unew = np.copy(Ustar)
    consToPrim(Ustar, Q);
    euler_step(Q, Unew, dt)
    # SSP-RK2
    for i in range(params.ibeg, params.iend):
        for j in range(params.jbeg, params.jend):
            for ivar in range(params.Nfields):
                Unew[i, j, ivar] = 0.5 * (U0[i, j, ivar] + Unew[i, j, ivar])
