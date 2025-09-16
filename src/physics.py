"""Module for physics-related calculations."""
import numpy as np
from src.states import State
import src.params as params
from src.varindexes import IP, IR

def speed_of_sound(q: State) -> float:
    """
    Calculate the speed of sound in the medium described by the state q.
    
    Parameters:
    q (State): The state of the system.
    
    Returns:
    float: The speed of sound.
    """
    return np.sqrt(params.gamma * q[IP] / q[IR])


