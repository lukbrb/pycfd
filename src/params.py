"""Paramètres de la simulation."""

from itertools import product
from numpy import ndindex, array

# Physics
problem_name = "orszag-tang"
MHD = True
gamma = 5 / 3
Nfields = 9 if MHD else 5

# Mesh
Nx = 128
Ny = 128
Nghosts = 2
Ntx = Nx + 2 * Nghosts
Nty = Ny + 2 * Nghosts
ibeg = Nghosts
iend = Nghosts + Nx
jbeg = Nghosts
jend = Nghosts + Ny

ParallelRange = product  # eventually change it for the numpy ndindex
range_tot = ndindex(Ntx, Nty)
range_dom = array(list(ParallelRange(range(ibeg, iend), range(jbeg, jend))))
range_xbound = array(list(ParallelRange(range(0, Nghosts), range(jbeg, jend))))
range_ybound = array(list(ParallelRange(range(0, Ntx), range(0, Nghosts))))
range_slopes = array(
    list(ParallelRange(range(ibeg - 1, ibeg + 1), range(jbeg - 1, jbeg + 1)))
)

xmax = 1.0
xmin = 0.0
ymax = 1.0
ymin = 0.0
dx = (xmax - xmin) / Nx
dy = (ymax - ymin) / Ny

# Boundaries
boundary_x = "BC_PERIODIC"
boundary_y = "BC_PERIODIC"

# Run
tend = 0.6
# Update - Hydro
CFL = 0.5
reconstruction = "PCM"
time_stepping = "euler"
riemann_solver = "fivewaves"
# Values
epsilon = 1e-6
smallr = 1e-10
# Output
log_frequency = 100
save_freq = 0.01
