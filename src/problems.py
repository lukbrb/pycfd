"""Module contenant les problèmes physiques à étudier."""
import numpy as np
import params
from states import State

def get_pos(i, j) -> tuple[float, float]:
    """
    Get the physical position corresponding to grid indices.
    
    Parameters:
    i (int): The x index in the grid.
    j (int): The y index in the grid.
    
    Returns:
    tuple[float, float]: The physical coordinates (x, y).
    """
    x = params.xmin + i * params.dx
    y = params.ymin + j * params.dy
    return x, y


def init_orszag_tang(grille: np.ndarray[State]) -> State:
    """
    Initialize the Orszag-Tang vortex problem.
    
    Parameters:
    grille ndarray(State): The initial state of the system.
    
    Returns:
    ndarray(State): The initialized state for the Orszag-Tang problem.
    """
    B0 = 1/ np.sqrt(4 * np.pi)
    for i in range(params.Nx):
        for j in range(params.Ny):
            x, y = get_pos(i, j)
            q = State(
                params.gamma**2 * B0**2,
                -np.sin(2 * np.pi * y),
                np.sin(2 * np.pi * x),
                0.0,
                params.gamma * B0**2,
                -B0 * np.sin(2 * np.pi * y),
                B0 * np.sin(4 * np.pi * x),
                0.0,
                0.0
            )
            grille[i, j] = q
    return grille
