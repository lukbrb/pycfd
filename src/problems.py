"""Module contenant les problèmes physiques à étudier."""

from typing import Callable
import numpy as np
import src.params as params
from src.pycfd_types import Array, real_t, IDir

# from pycfd_types import VarIndex as C
from src.varindexes import IR, IU, IV, IW, IP, IBX, IBY, IBZ, IPSI


def get_pos(i, j) -> tuple[real_t, real_t]:
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


def init_sod_x(Q: Array, i: int, j: int) -> None:
    if get_pos(i, j)[IDir.IX] <= 0.5:
        Q[i, j, IR] = 1.0
        Q[i, j, IP] = 1.0
        Q[i, j, IU] = 0.0

    else:
        Q[i, j, IR] = 0.125
        Q[i, j, IP] = 0.1
        Q[i, j, IU] = 5


def init_orszag_tang(Q: Array, i: int, j: int) -> None:
    """
    Initialize the Orszag-Tang vortex problem.

    Parameters:
    grille ndarray(State): The initial state of the system.

    Returns:
    ndarray(State): The initialized state for the Orszag-Tang problem.
    """
    B0 = 1 / np.sqrt(4 * np.pi)
    x, y = get_pos(i, j)
    Q[i, j, IR] = params.gamma**2 * B0**2
    Q[i, j, IU] = -np.sin(2 * np.pi * y)
    Q[i, j, IV] = np.sin(2 * np.pi * x)
    Q[i, j, IW] = 0.0
    Q[i, j, IP] = params.gamma * B0**2
    Q[i, j, IBX] = -B0 * np.sin(2 * np.pi * y)
    Q[i, j, IBY] = B0 * np.sin(4 * np.pi * x)
    Q[i, j, IBZ] = 0.0
    Q[i, j, IPSI] = 0.0


def init_test(Q: Array, i: int, j: int) -> None:
    Q[i, j, IR] = 1
    Q[i, j, IU] = 0
    Q[i, j, IV] = 0
    Q[i, j, IP] = params.gamma


problems: dict[str, Callable] = {
    "orszag-tang": init_orszag_tang,
    "sod_x": init_sod_x,
}


def init_problem(Q: Array, problem_name: str) -> None:
    """Initialise the whole grid given an initial problem setup

    Q (np.ndarray) : Array containing all the data of the grid
    problem_name (str): Key to the function to call to fill each cell.
    """
    assert problem_name in problems, "The chosen problem is not referenced."
    for i, j in params.range_dom:
        problems[problem_name](Q, i, j)
