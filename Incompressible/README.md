# Navier-Stokes spectral solver for 2D-incompressible flows

## Requirements :

- Python 3.6 (usage of fstrings)
- Numpy
- Matplotlib
- Scipy
- Vtk
- tkdm


## Short description

This program is a solver for the Navier-Stokes equations in 2D incompressible flows. It relies on spectral decomposition to compute derivatives using FFT. A first-order discretization scheme is applied for time integration.

The code is intended for training purposes and serves as a first step toward developing more comprehensive solvers.


## Limitations 

A dealiasing filter is applied when solving the non-linear term. As a result, small-scale phenomena are not fully resolved.

The use of spectral methods necessitates periodic boundary conditions at the domain's external boundaries. To mitigate the undesired effects of this model, several buffers are implemented near these boundaries.


