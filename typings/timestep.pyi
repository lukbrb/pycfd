from src.physics import speed_of_sound as speed_of_sound
from src.pycfd_types import Array as Array, real_t as real_t
from src.states import State as State, get_state_from_array as get_state_from_array
from src.varindexes import IBX as IBX, IBY as IBY, IBZ as IBZ, IR as IR, IU as IU, IV as IV

def cell_timestep(q: State) -> real_t: ...
def compute_dt(Q: Array, t: real_t, verbose: bool) -> real_t: ...
