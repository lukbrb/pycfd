from typing import Union, Optional
from functools import singledispatch
import numpy as np
import src.params as params
from src.pycfd_types import real_t, Array, IDir
from src.varindexes import IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI

# IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI = VarIndex.__members__


class State(np.ndarray):
    """Classe de base pour PrimState et ConsState."""

    def __new__(cls, data: Union[Array, None] = None) -> "State":
        arr = np.zeros(params.Nfields) if data is None else np.array(data, dtype=real_t)
        if arr.shape != (params.Nfields,):
            raise ValueError(f"Expected {params.Nfields} fields, got {arr.shape}")
        # vue ndarray avec type State
        return arr.view(cls)

    def __array_finalize__(self, obj):
        # appelé à chaque fois qu’un nouvel ndarray-compat est créé
        if obj is None:
            return

        # --- Opérations arithmétiques binaires (u + v, u - v, etc.) ---

    def __add__(self, other):
        result = super().__add__(other)
        return result.view(State)

    def __sub__(self, other):
        result = super().__sub__(other)
        return result.view(State)

    def __mul__(self, other):
        result = super().__mul__(other)
        return result.view(State)

    def __truediv__(self, other):
        result = super().__truediv__(other)
        return result.view(State)

    def __floordiv__(self, other):
        result = super().__floordiv__(other)
        return result.view(State)

    def __pow__(self, other):
        result = super().__pow__(other)
        return result.view(State)

    # --- Opérations arithmétiques binaires inversées (5 + u, etc.) ---
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return State(other) - self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return State(other) / self

    def __rfloordiv__(self, other):
        return State(other) // self

    def __rpow__(self, other):
        return State(other) ** self

    # --- Opérations arithmétiques in-place (u += v, etc.) ---
    def __iadd__(self, other):
        super().__iadd__(other)
        return self

    def __isub__(self, other):
        super().__isub__(other)
        return self

    def __imul__(self, other):
        super().__imul__(other)
        return self

    def __itruediv__(self, other):
        super().__itruediv__(other)
        return self

    def __ifloordiv__(self, other):
        super().__ifloordiv__(other)
        return self

    def __ipow__(self, other):
        super().__ipow__(other)
        return self

    # --- Opérations unaires (-u, +u, abs(u), etc.) ---
    def __neg__(self):
        result = super().__neg__()
        return result.view(State)

    def __pos__(self):
        result = super().__pos__()
        return result.view(State)

    def __abs__(self):
        result = super().__abs__()
        return result.view(State)

    # --- Comparaisons (u == v, u < v, etc.) ---
    def __eq__(self, other):
        return super().__eq__(other)

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    def __ge__(self, other):
        return super().__ge__(other)

    def __ne__(self, other):
        return super().__ne__(other)


def get_state_from_array(Q: Array, i: int, j: int) -> State:
    s = State()
    s[IR] = Q[i, j, IR]
    s[IU] = Q[i, j, IU]
    s[IV] = Q[i, j, IV]
    s[IW] = Q[i, j, IW]
    s[IP] = Q[i, j, IP]
    s[IBX] = Q[i, j, IBX]
    s[IBY] = Q[i, j, IBY]
    s[IBZ] = Q[i, j, IBZ]
    s[IPSI] = Q[i, j, IPSI]
    return s


def set_state_into_array(Q: Array, i: int, j: int, s: State) -> None:
    Q[i, j, IR] = s[IR]
    Q[i, j, IU] = s[IU]
    Q[i, j, IV] = s[IV]
    Q[i, j, IW] = s[IW]
    Q[i, j, IP] = s[IP]
    Q[i, j, IBX] = s[IBX]
    Q[i, j, IBY] = s[IBY]
    Q[i, j, IBZ] = s[IBZ]
    Q[i, j, IPSI] = s[IPSI]


def cell_primToCons(q: State) -> State:
    u: State = State()
    u[IR] = q[IR]
    u[IU] = q[IR] * q[IU]
    u[IV] = q[IR] * q[IV]
    u[IW] = q[IR] * q[IW]
    u[IBX] = q[IBX]
    u[IBY] = q[IBY]
    u[IBZ] = q[IBZ]
    u[IPSI] = q[IPSI]
    Ek = q[IR] * 0.5 * (q[IU] ** 2 + q[IV] ** 2 + q[IW] ** 2)
    Emag = 0.5 * (q[IBX] ** 2 + q[IBY] ** 2 + q[IBZ] ** 2)
    Epsi = 0.5 * q[IPSI] ** 2
    u[IE] = q[IP] / (params.gamma - 1) + Ek + Emag + Epsi
    return u


def cell_consToPrim(u: State) -> State:
    q: State = State()
    q[IR] = u[IR]
    q[IU] = u[IU] / u[IR]
    q[IV] = u[IV] / u[IR]
    q[IW] = u[IW] / u[IR]
    q[IBX] = u[IBX]
    q[IBY] = u[IBY]
    q[IBZ] = u[IBZ]
    q[IPSI] = u[IPSI]
    Ek = q[IR] * 0.5 * (q[IU] ** 2 + q[IV] ** 2 + q[IW] ** 2)
    Emag = 0.5 * (q[IBX] ** 2 + q[IBY] ** 2 + q[IBZ] ** 2)
    Epsi = 0.5 * q[IPSI] ** 2
    q[IP] = (u[IE] - Ek - Emag - Epsi) * (params.gamma - 1)
    return q


# Note : grid_** function should be vectorized with numpy array arithmetic
def grid_consToPrim(U: Array, Q: Array) -> None:
    for i, j in params.range_dom:
        u_loc = get_state_from_array(U, i, j)
        q_loc = cell_consToPrim(u_loc)
        set_state_into_array(Q, i, j, q_loc)


def grid_primToCons(Q: Array, U: Array) -> None:
    for i, j in params.range_dom:
        q_loc = get_state_from_array(Q, i, j)
        u_loc = cell_primToCons(q_loc)
        set_state_into_array(U, i, j, u_loc)


@singledispatch
def primToCons(arg1: State | Array, arg2: Optional[Array] = None) -> Union[State, None]:
    """Fonction générique pour primToCons."""
    raise NotImplementedError("Type non supporté.")


@primToCons.register
def _(q: State, _: None = None) -> State:
    """Conversion cellule->cellule (State -> State)."""
    return cell_primToCons(q)


@primToCons.register
def _(Q: Array, U: Array) -> None:
    """Conversion grille->grille (Array -> Array)."""
    grid_primToCons(Q, U)


@singledispatch
def consToPrim(arg1: State | Array, arg2: Optional[Array] = None) -> Union[State, None]:
    """Fonction générique pour primToCons."""
    raise NotImplementedError("Type non supporté.")


@consToPrim.register
def _(q: State, _: None = None) -> State:
    """Conversion cellule->cellule (State -> State)."""
    return cell_consToPrim(q)


@consToPrim.register
def _(Q: Array, U: Array) -> None:
    """Conversion grille->grille (Array -> Array)."""
    grid_consToPrim(Q, U)


def swap_components(s: State, idir: IDir) -> State:
    if idir == IDir.IX:
        return s
    elif idir == IDir.IY:
        return State(
            np.array(
                [s[IR], s[IV], s[IU], s[IW], s[IP], s[IBY], s[IBX], s[IBZ], s[IPSI]]
            )
        )
    elif idir == IDir.IZ:
        return State(
            np.array(
                [s[IR], s[IW], s[IV], s[IU], s[IP], s[IBZ], s[IBY], s[IBX], s[IPSI]]
            )
        )
    else:
        raise ValueError("Chosen dir is not recognized.")
