import numpy as np
from typing import Sequence, Union
from pycfd_types import real_t, VarIndex
import params

IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI = VarIndex.__members__

class State(np.ndarray):
    """Classe de base pour PrimState et ConsState."""

    def __new__(cls, data: Union[Sequence[real_t], None] = None) -> 'State':
        arr = np.array(data, dtype=real_t) if data else np.zeros(params.Nfields)
        if arr.shape != (params.Nfields,):
            raise ValueError(f"Expected {params.Nfields} fields, got {arr.shape}")
        # vue ndarray avec type State
        return arr.view(cls)

    def __array_finalize__(self, obj):
        # appelé à chaque fois qu’un nouvel ndarray-compat est créé
        if obj is None:
            return


def primToCons(q: State) -> State:
    u: State = State()
    u[IR]   = q[IR]
    u[IU]   = q[IR] * q[IU]
    u[IV]   = q[IR] * q[IV]
    u[IW]   = q[IR] * q[IW]
    u[IBX]  = q[IBX]
    u[IBY]  = q[IBY]
    u[IBZ]  = q[IBZ]
    u[IPSI] = q[IPSI]
    Ek = q[IR] * 0.5 * (q[IU]**2 + q[IV]**2 + q[IW]**2)
    Emag = 0.5 * (q[IBX]**2 + q[IBY]**2 + q[IBZ]**2)
    Epsi = 0.5 * q[IPSI]**2
    u[IE] = q[IP]/(params.gamma - 1) + Ek + Emag + Epsi
    return u


def consToPrim(u: State) -> State:
    q: State = State()
    q[IR]   = u[IR]
    q[IU]   = u[IU] / u[IR]
    q[IV]   = u[IV] / u[IR]
    q[IW]   = u[IW] / u[IR]
    q[IBX]  = u[IBX]
    q[IBY]  = u[IBY]
    q[IBZ]  = u[IBZ]
    q[IPSI] = u[IPSI]
    Ek = q[IR] * 0.5 * (q[IU]**2 + q[IV]**2 + q[IW]**2)
    Emag = 0.5 * (q[IBX]**2 + q[IBY]**2 + q[IBZ]**2)
    Epsi = 0.5 * q[IPSI]**2
    q[IP] = (q[IE] - Ek - Emag - Epsi) * (params.gamma -1)
    return q


# class PrimState(State):
#     pass

# class ConsState(State):
#     pass