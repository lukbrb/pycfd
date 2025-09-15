"""Paramètres de la simulation."""
from itertools import product
# from numpy import ndindex

# Physics
problem_name = "sod_x"
MHD = True
gamma = 5/3
Nfields = 9 if MHD else 5

# Mesh 
Nx = 16
Ny = 16
Nghosts = 2
Ntx = Nx + 2*Nghosts
Nty = Ny + 2*Nghosts
ibeg = Nghosts
iend = Nghosts+Nx
jbeg = Nghosts
jend = Nghosts+Ny

ParallelRange = product # eventually change it for the numpy ndindex
range_tot = ParallelRange(range(0, Ntx), range(0, Nty))
range_dom = ParallelRange(range(ibeg, iend), range(jbeg,jend))
range_xbound = ParallelRange(range(0, Nghosts), range(jbeg, jend))
range_ybound = ParallelRange(range(0, Ntx), range(0, Nghosts))
range_slopes = ParallelRange(range(ibeg-1, ibeg+1), range(jbeg-1, jbeg+1))

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
tend = 2.0
# Update - Hydro
CFL = 0.5
reconstruction = 'PCM'
time_stepping = 'euler'
riemann_solver = "HLL"
# Values
epsilon = 1e-6
smallr = 1e-10
# Output
log_frequency = 1
save_freq = 10