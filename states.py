from dataclasses import dataclass
import params
import numpy as np

@dataclass
class State:
    """
    Represents the state of the Magnetohydrodynamics system, augmented with the \Psi field.
    """
    r: float
    u: float
    v: float
    w: float
    p: float
    bx: float
    by: float
    bz: float
    psi: float
    
    def __add__(self, other: 'State') -> 'State':
        return State(
            self.r + other.r,
            self.u + other.u,
            self.v + other.v,
            self.w + other.w,
            self.p + other.p,
            self.bx + other.bx,
            self.by + other.by,
            self.bz + other.bz,
            self.psi + other.psi
        )
    
    def __sub__(self, other: 'State') -> 'State':
        return State(
            self.r - other.r,
            self.u - other.u,
            self.v - other.v,
            self.w - other.w,
            self.p - other.p,
            self.bx - other.bx,
            self.by - other.by,
            self.bz - other.bz,
            self.psi - other.psi
        )
    
    def __mul__(self, scalar: float|'State') -> 'State':
        if isinstance(scalar, State):
            return State(
                self.r * scalar.r,
                self.u * scalar.u,
                self.v * scalar.v,
                self.w * scalar.w,
                self.p * scalar.p,
                self.bx * scalar.bx,
                self.by * scalar.by,
                self.bz * scalar.bz,
                self.psi * scalar.psi
            )
        
        elif isinstance(scalar, (int, float)):
            return State(
                self.r * scalar,
                self.u * scalar,
                self.v * scalar,
                self.w * scalar,
                self.p * scalar,
                self.bx * scalar,
                self.by * scalar,
                self.bz * scalar,
                self.psi * scalar
            )

def cons_to_prim(u: State) -> State:
    """
    Convert conservative variables to primitive variables.
    
    Parameters:
    u (State): The state of the system in conservative variables.
    
    Returns:
    State: The state of the system in primitive variables.
    """
    res =  State(
        r=u.r,
        u=u.u/u.r,
        v=u.v/u.r,
        w=u.w/u.r,
        p=0,
        bx=u.bx,
        by=u.by,
        bz=u.bz,
        psi=u.psi
    )

    Ek = 0.5 * res.r * (res.u**2 + res.v**2 + res.w**2)
    Em = 0.5 * (u.bx**2 + u.by**2 + u.bz**2)
    Epsi = 0.5 * u.psi**2
    res.p = (u.p - Ek - Em - Epsi) * (params.gamma - 1)
    return res

def prim_to_cons(q: State) -> State:
    """
    Convert primitive variables to conservative variables.
    
    Parameters:
    q (State): The state of the system in primitive variables.
    
    Returns:
    State: The state of the system in conservative variables.
    """
    Ek = 0.5 * q.r * (q.u**2 + q.v**2 + q.w**2)
    Em = 0.5 * (q.bx**2 + q.by**2 + q.bz**2)
    Epsi = 0.5 * q.psi**2
    res = State(
        r=q.r,
        u=q.r * q.u,
        v=q.r * q.v,
        w=q.r * q.w,
        p=Ek + Em + Epsi + q.p / (params.gamma - 1),
        bx=q.bx,
        by=q.by,
        bz=q.bz,
        psi=q.psi
    )
    return res

def cons_to_prim_array(u: np.ndarray[State], q: np.ndarray[State]) -> np.ndarray[State]:
    """
    Convert an array of conservative variables to primitive variables.
    
    Parameters:
    u (np.ndarray[State]): The array of states in conservative variables.
    
    Returns:
    np.ndarray[State]: The array of states in primitive variables.
    """
    for i in range(params.Nx):
        for j in range(params.Ny):
            q[i, j] = cons_to_prim(u[i, j])
    return q

def prim_to_cons_array(q: np.ndarray[State], u: np.ndarray[State]) -> np.ndarray[State]:
    """
    Convert an array of primitive variables to conservative variables.
    
    Parameters:
    q (np.ndarray[State]): The array of states in primitive variables.
    
    Returns:
    np.ndarray[State]: The array of states in conservative variables.
    """
    for i in range(params.Nx):
        for j in range(params.Ny):
            u[i, j] = prim_to_cons(q[i, j])
    return u

def swap_components(q: State, idir: int) -> State:
    """
    Swap the components of the state vector.
    
    Parameters:
    q (State): The state of the system.
    
    Returns:
    State: The state with swapped components.
    """
    if idir == 0:
        return State(
            r=q.r,
            u=q.u,
            v=q.v,
            w=q.w,
            p=q.p,
            bx=q.bx,
            by=q.by,
            bz=q.bz,
            psi=q.psi
        )
    else:
        return State(
            r=q.r,
            u=q.v,
            v=q.u,
            w=q.w,
            p=q.p,
            bx=q.by,
            by=q.bx,
            bz=q.bz,
            psi=q.psi
        )
