"""Compute update."""

import numpy as np
from states import State, swap_components, cons_to_prim_array
from riemann import IdealGLM
import params
from timestep import timestep


def computeGlobalDivSpeed(Q: np.ndarray) -> float:
    umax = 0.0
    lmax = 0.0
    for i in range(params.Nx):
        for j in range(params.Ny):
            q = Q[i, j]
            
def reconstruct(Q: np.ndarray, i: int, j: int) -> np.ndarray[State]:
    q = Q[i, j]
    match params.reconstruction:
        case "PCM":
            return Q
        case "PLM":
            raise NotImplementedError("Piecewise Linear Method (PLM) reconstruction is not implemented.")
        case "PPM":
            raise NotImplementedError("Piecewise Parabolic Method (PPM) reconstruction is not implemented.")
    raise ValueError(f"Unknown reconstruction method: {params.reconstruction}")

def computeFluxesAndUpdate(Q: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray[State]:
    """
    Compute the fluxes for the given state array Q using the Riemann solver.
    
    Parameters:
    Q (np.ndarray): The state array.
    U (np.ndarray): The conservative variables array.
    Returns:
    np.ndarray: The computed fluxes.
    """
    def update_along_dir(i, j, idir: int) -> State:
        """
        Update the state along a specified direction.
        
        Parameters:
        i (int): The x index in the grid.
        j (int): The y index in the grid.
        dir (int): The direction to update (0 for x, 1 for y).
        
        Returns:
        State: The updated state.
        """

        dxm = -1 if idir == 0 else 0
        dxp = 1 if idir == 0 else 0
        dym = -1 if idir == 1 else 0
        dyp = 1 if idir == 1 else 0
        qCL = reconstruct(Q, i, j)
        qCR = reconstruct(Q, i, j)
        qL = reconstruct(Q, i + dxm, j + dym)
        qR = reconstruct(Q, i + dxp, j + dyp)
        fluxL = IdealGLM(qL, qCL)
        fluxR = IdealGLM(qCR, qR)
        fluxL = swap_components(fluxL, idir)
        fluxR = swap_components(fluxR, idir)
        un_loc = U[i, j]
        un_loc += dt * (fluxL - fluxR)
        U[i, j] = un_loc

    for i in range(params.Nx):
        for j in range(params.Ny):
            update_along_dir(i, j, 0)  # Update along x direction
            update_along_dir(i, j, 1)
    
    return U


def euler_step(Q: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray[State]:
    """
    Perform a single Euler step to update the state array Q.
    
    Parameters:
    Q (np.ndarray): The state array.
    U (np.ndarray): The conservative variables array.
    dt (float): The time step size.
    
    Returns:
    np.ndarray: The updated state array.
    """
    U = computeFluxesAndUpdate(Q, U, dt)
    # Assurer que la pression est positive et appliquer les sources termes
    return U


def update(Q: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray[State]:
    if params.timestep.lower() == 'euler':
        return euler_step(Q, U, dt)
    elif params.timestep.lower() == 'rk2':
        # Step 1
        U0 = np.copy(U)
        Ustar = euler_step(Q, U, dt)
        # Step 2
        Q = cons_to_prim_array(Ustar, Q)
        Unew = np.copy(Ustar)
        Unew = euler_step(Q, Unew, dt)

        for i in range(params.Nx):
            for j in range(params.Ny):
                U[i, j] = 0.5 * (U0[i, j] + Unew[i, j])
    return U
