import numpy as np
from src.varindexes import IBX as IBX, IBY as IBY, IBZ as IBZ, IP as IP, IPSI as IPSI, IR as IR, IU as IU, IV as IV, IW as IW

class IOManager:
    outname: str
    dirname: str
    ite_nzeros: int
    Nx: int
    Ny: int
    Ntx: int
    Nty: int
    ibeg: int
    iend: int
    jbeg: int
    jend: int
    dx: float
    dy: float
    xmin: float
    ymin: float
    MHD: bool
    write_ghost_cells: bool
    def __init__(self, outname: str = 'run', dirname: str = 'data') -> None: ...
    def setup_dirdata(self) -> None: ...
    def save_solution(self, Q: np.ndarray, iteration: int, t: float, unique_output: bool = False) -> None: ...
    def load_solution(self, iteration: int) -> dict[str, np.ndarray]: ...
