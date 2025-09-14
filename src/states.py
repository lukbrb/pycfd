from functools import singledispatch
from typing import Sequence, Union, overload
import numpy as np
import params
from pycfd_types import real_t, VarIndex, Array, IDir

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


def get_state_from_array(Q: Array, i: int, j: int) -> State:
    return Q[i, j, :]


def set_state_into_array(Q: Array, i: int, j: int, s: State) -> None:
    Q[i, j, IR]   = s[IR]
    Q[i, j, IU]   = s[IU]
    Q[i, j, IV]   = s[IV]
    Q[i, j, IW]   = s[IW]
    Q[i, j, IP]   = s[IP]
    Q[i, j, IBX]  = s[IBX]
    Q[i, j, IBY]  = s[IBY]
    Q[i, j, IBZ]  = s[IBZ]
    Q[i, j, IPSI] = s[IPSI]


@overload
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

@overload
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

@overload
def consToPrim(U: Array, Q: Array) -> Array:
    for i in range(params.Ntx):
        for j in range(params.Nty):
            u_loc = get_state_from_array(U, i, j)
            q_loc = consToPrim(u_loc)
            set_state_into_array(Q, i, j, q_loc)


@overload
def primToCons(Q: Array, U: Array) -> Array:
    for i in range(params.Ntx):
        for j in range(params.Nty):
            q_loc = get_state_from_array(Q, i, j)
            u_loc = primToCons(q_loc)
            set_state_into_array(U, i, j, u_loc)


def swap_components(s: State, idir: IDir) -> State:
    if idir == IDir.IX:
        return s
    elif idir == IDir.IY:
        return State(s[IR], s[IV], s[IU], s[IW], s[IP], s[IBY], s[IBX], s[IBZ], s[IPSI])
    elif idir == IDir.IZ:
        return State(s[IR], s[IW], s[IV], s[IU], s[IP], s[IBZ], s[IBY], s[IBX], s[IPSI])
    else:
        raise ValueError("Chosen dir is not recognized.")


# class PrimState(State):
#     pass

# class ConsState(State):
#     pass