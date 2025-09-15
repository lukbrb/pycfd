import numpy as np
import params
from pycfd_types import Array, real_t
from problems import init_problem
from states import primToCons, consToPrim
from timestep import compute_dt
from update import update
from iomanager import save_solution
from boundaries import fillBoundaries

def main() -> int:

    print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
    print("░        ░░   ░░░░░░░░░   ░░░░░░░░░░░░░░░░░   ░")
    print("▒   ▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒   ▒▒▒   ▒  ▒▒▒▒▒▒▒▒▒   ▒")
    print("▒   ▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒   ▒▒▒  ▒▒▒▒▒   ▒▒▒▒▒▒   ▒")
    print("▓       ▓▓▓▓▓▓   ▓▓▓   ▓▓▓▓▓▓▓▓▓   ▓▓▓▓   ▓   ▓")
    print("▓   ▓▓▓▓▓▓▓▓▓▓▓   ▓   ▓▓▓▓▓▓▓▓   ▓▓▓▓▓  ▓▓▓   ▓")
    print("▓   ▓▓▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓  ▓▓▓   ▓")
    print("█   █████████████   ███████         ███   █   █")
    print("███████████████████████████████████████████████")

    # Reading parameters from .ini file
    # auto params = readInifile(argv[1]);
    # auto device_params = params.device_params;

    U: Array = np.zeros((params.Ntx, params.Nty, params.Nfields), dtype=real_t)
    Q: Array = np.zeros((params.Ntx, params.Nty, params.Nfields), dtype=real_t)

    # // Misc vars for iteration
    t: real_t = 0.0
    ite: int = 0
    next_save: real_t = 0.0
    
    # // Initializing primitive variables
    # InitFunctor init(params);
    # UpdateFunctor update(params);
    # ComputeDtFunctor computeDt(params);
    # IOManager ioManager(params);

    # if (params.restart_file != "") {
    #   auto restart_info = ioManager.loadSnapshot(Q);
    #   t = restart_info.time;
    #   ite = restart_info.iteration;
    #   std::cout << "Restart at iteration " << ite << " and time " << t << std::endl;
    #   next_save = t + params.save_freq;
    #   ite++;
    # }
    # else

    init_problem(Q, params.problem_name)
    print(Q)
    
    fillBoundaries(Q)
    primToCons(Q, U)

    dt: real_t = 0.0
    next_log: int = 0
    while (t + params.epsilon < params.tend):
        save_needed: bool = (t + params.epsilon > next_save)
        dt = compute_dt(Q, t, next_log == 0)
        if (next_log == 0):
            next_log = params.log_frequency
        else:
            next_log -= 1

        if (save_needed):
            print(f" - Saving at time {t:.3f}")
            ite += 1
            save_solution(Q, ite, t)
            next_save += params.save_freq

        update(Q, U, dt)
        consToPrim(U, Q)
        # checkNegatives(Q, params)

    t += dt

    print(f"Time at end is {t:.3f}")
    ite += 1
    save_solution(Q, ite, t)

    print("    █     ▀██  ▀██         ▀██                              ▄█▄ ")
    print("   ███     ██   ██       ▄▄ ██    ▄▄▄   ▄▄ ▄▄▄     ▄▄▄▄     ███ ")
    print("  █  ██    ██   ██     ▄▀  ▀██  ▄█  ▀█▄  ██  ██  ▄█▄▄▄██    ▀█▀ ")
    print(" ▄▀▀▀▀█▄   ██   ██     █▄   ██  ██   ██  ██  ██  ██          █  ")
    print("▄█▄  ▄██▄ ▄██▄ ▄██▄    ▀█▄▄▀██▄  ▀█▄▄█▀ ▄██▄ ██▄  ▀█▄▄▄▀     ▄  ")
    print("                                                            ▀█▀ ")

    return 0

if __name__ == "__main__":
    main()