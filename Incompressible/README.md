# Navier-Stokes spectral solver for 2D-incompressible flows

## Requirements :

- Python 3.6 (usage of fstrings)
- Numpy
- Matplotlib
- Scipy
- Vtk
- tkdm


## Short description

This program is a solver for Navier-Stokes equations in incompressible flows. This solver relies on spectral decompositions to solve derivatives.(usage of fft)

## Limitations 

A dealiasing filter is applied when solving the non-linear term. Therefore, in this specific case, small scales phenomenon are not solved.

