"""Handle the input/oyput operations"""
from pathlib import Path
from src.pycfd_types import Array, real_t
import numpy as np


class IOManager:
    def __init__(self, outname: str="run", dirname: str="data/") -> None:
        self.dirname: Path = Path(dirname)
        self.outname: str = outname

    def setup_dirdata(self) -> None:
        self.dirname.mkdir(exist_ok=True)

    def save_solution(self, Q: Array, ite: int, t: real_t) -> None:
        fields = ["rho", "u", "v", "w", "p", "bx", "by", "bz", "psi"]
        for ivar, f in enumerate(fields):
            fname = self.dirname / Path(f"{self.outname}_{f}_{ite:04d}.dat")
            np.savetxt(fname, Q[..., ivar].T, comments=str(t))
