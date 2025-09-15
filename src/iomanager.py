"""Handle the input/oyput operations"""

from pycfd_types import Array, real_t
import numpy as np


def save_solution(Q: Array, ite: int, t: real_t) -> None:
    
    fields = ["rho", "u", "v", "w", "p", "bx", "by", "bz", "psi"]
    for ivar, f in enumerate(fields):
        fname = f"run_{f}_{ite:04d}.dat"
        np.savetxt(fname, Q[..., ivar].T, comments=str(t))
