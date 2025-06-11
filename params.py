"""Paramètres de la simulation."""

Nx = 128
Ny = 128
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