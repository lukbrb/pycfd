# Module used only for defining aliases for types
# use only for python type checking
from typing import TypeAlias
from enum import IntEnum
import numpy as np


Array: TypeAlias = np.ndarray
real_t: TypeAlias = float
# @verify(CONTINUOUS)
# class VarIndex(IntEnum):
#     IR   = 0
#     IU   = 1
#     IV   = 2
#     IW   = 3
#     IP   = 4
#     IE   = 4
#     IBX  = 5
#     IBY  = 6
#     IBZ  = 7
#     IPSI = 8

# VarIndex.IP._add_alias_("IE")


class IDir(IntEnum):
    IX = 0
    IY = 1
    IZ = 2
