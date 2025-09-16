from src.pycfd_types import Array, IDir
from src.states import State, get_state_from_array, set_state_into_array
import src.params as params
from src.varindexes import IU, IV, IBX, IBY


def fillAbsorbing(Q: Array, iref: int, jref: int, idir: IDir) -> State:
    q: State = get_state_from_array(Q, iref, jref)
    return q


def fillReflecting(Q: Array, i: int, j: int, iref: int, jref: int, idir: IDir) -> State:
    if idir == IDir.IX:
        ipiv: int = params.ibeg if i < iref else params.iend
        isym = 2*ipiv - i - 1
        jsym = j
    else:
        jpiv: int = params.jbeg if j < jref else params.jend
        isym = i
        jsym = 2*jpiv - j - 1

    q = get_state_from_array(Q, isym, jsym)

    if idir == IDir.IX:
        q[IU] *= -1.0
        q[IBX] *= -1.0

    else:
        q[IV] *= -1.0
        q[IBY] *= -1.0
    return q


def fillPeriodic(Q: Array, i: int, j: int, idir: IDir) -> State:
    if idir == IDir.IX:
        if i < params.ibeg:
            i += params.Nx
        else:
            i -= params.Nx
    else:
        if j < params.jbeg:
            j += params.Ny
        else:
            j -= params.Ny

    return get_state_from_array(Q, i, j)


def fillBoundaries(Q: Array) -> None:
    bc_x = params.boundary_x
    bc_y = params.boundary_y

    for (i, j) in params.range_xbound:
        ileft = i
        iright    = params.iend+i
        iref_left = params.ibeg
        iref_right = params.iend-1

        def fillX(i: int, iref: int) -> State:
            match (bc_x):
                case "BC_ABSORBING":
                    return fillAbsorbing(Q, iref, j, IDir.IX)
                case "BC_REFLECTING":
                    return fillReflecting(Q, i, j, iref, j, IDir.IX)
                case "BC_PERIODIC":
                      return fillPeriodic(Q, i, j, IDir.IX)
                case _:
                    return fillPeriodic(Q, i, j, IDir.IX)

        set_state_into_array(Q, ileft,  j, fillX(ileft, iref_left))
        set_state_into_array(Q, iright, j, fillX(iright, iref_right))

    for (i, j) in params.range_ybound:
        jtop: int     = j
        jbot: int     = params.jend+j
        jref_top: int = params.jbeg
        jref_bot: int = params.jend-1

        def fillY(j: int, jref: int) -> State:
            match (bc_y):
                case "BC_ABSORBING":
                    return fillAbsorbing(Q, i, jref, IDir.IY)
                case "BC_REFLECTING":
                    return fillReflecting(Q, i, j, i, jref, IDir.IY)
                case "BC_PERIODIC":
                      return fillPeriodic(Q, i, j, IDir.IY)
                case _:
                    return fillPeriodic(Q, i, j, IDir.IY)
                
        set_state_into_array(Q, i, jtop, fillY(jtop, jref_top))
        set_state_into_array(Q, i, jbot, fillY(jbot, jref_bot))