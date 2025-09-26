from enum import IntEnum
from typing import TypeAlias
from numpy import ndarray

Array: TypeAlias = ndarray
real_t: TypeAlias = float

class IDir(IntEnum):
    IX = 0
    IY = 1
    IZ = 2
