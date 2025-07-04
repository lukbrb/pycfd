import numpy as np
import params
from states import State, prim_to_cons_array, cons_to_prim_array
import problems
from update import update
from timestep import timestep


def main():
    print("PYCFD - Python Computational Fluid Dynamics")

    U = np.zeros((params.Nx, params.Ny), dtype=State)
    Q = np.zeros((params.Nx, params.Ny), dtype=State)

    t = 0.0
    ite = 0
    Q = problems.init_orszag_tang(Q)
    U = prim_to_cons_array(Q, U)
    print(Q)

    while t < params.tmax:
        dt = timestep(Q)
        U = update(Q, U, dt)
        t += dt
        if ite % 10 == 0:
            print(Q)


if __name__ == "__main__":
    main()
