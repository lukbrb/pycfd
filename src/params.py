"""Paramètres de la simulation."""
problem_name = "test_ini"
MHD = False
Nx = 16
Ny = 16
Nghosts = 2
Ntx = Nx + Nghosts
Nty = Ny + Nghosts
Nfields = 9 if MHD else 5
gamma = 5/3
cfl = 0.5
xmax = 1.0
xmin = 0.0
ymax = 1.0
ymin = 0.0
dx = (xmax - xmin) / Nx
dy = (ymax - ymin) / Ny
tmax = 2.0

reconstruction = 'PCM'
timestep = 'euler'
