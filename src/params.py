"""Paramètres de la simulation."""
# Physics
problem_name = "test_ini"
MHD = False
gamma = 5/3
Nfields = 9 if MHD else 5

# Mesh 
Nx = 16
Ny = 16
Nghosts = 2
Ntx = Nx + Nghosts
Nty = Ny + Nghosts
ibeg = Nghosts;
iend = Nghosts+Nx;
jbeg = Nghosts;
jend = Nghosts+Ny;

xmax = 1.0
xmin = 0.0
ymax = 1.0
ymin = 0.0
dx = (xmax - xmin) / Nx
dy = (ymax - ymin) / Ny

# Run
tend = 2.0
# Update - Hydro
cfl = 0.5
reconstruction = 'PCM'
timestep = 'euler'
epsilon = 1e-6

# Output
log_frequency = 10