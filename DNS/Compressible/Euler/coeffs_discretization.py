import numpy as np

# ------------------------------------------------------------------------------
#  AUTHOR
#  ------
#  Luca Sciacovelli (luca.sciacovelli@ensam.eu)
#
#  DESCRIPTION
#  -----------
#  List of coefficients for explicit derivatives
#  N.B. for centered schemes only half of the coefficients is listed because
#  they are symmetric. e.g. for 4th order a(-2) = - a(2), a(-1) = -a(1), a(0) = 0
#
#  Ref. for DRP centered schemes:
#  Bogey, Bailly: A family of low dispersive and low dissipative explicit schemes
#  for flow and noise computations, JCP 2004
# ------------------------------------------------------------------------------

# -------------------------------
# 3 POINTS
# -------------------------------
# Schéma numérique : df/dx ≈ (f(x+dx) - f(x-dx)) / (2*dx)
a02c_std = np.array([0.5])

a02d_std = np.array([-3.0, 4.0, -1.0]) / 2.0  # df/dx ≈ (-3*f0 + 4*f1 - f2) / (2*dx)

# -------------------------------
# 5 POINTS
# -------------------------------
# Schéma numérique : df/dx ≈ (-8*f(-1) + 8*f(1)) / (12*dx)
a04c_std = np.array([8.0, -1.0]) / 12.0

# df/dx ≈ (-3*f0 - 10*f1 + 18*f2 - 6*f3 + f4) / (12*dx)
a13d_std = np.array([-3.0, -10.0, 18.0, -6.0, 1.0]) / 12.0

# df/dx ≈ (-25*f0 + 48*f1 - 36*f2 + 16*f3 - 3*f4) / (12*dx)
a04d_std = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / 12.0

# -------------------------------
# 7 POINTS
# -------------------------------
# Schéma numérique : df/dx ≈ (45*f(-1) - 9*f(-2) + f(-3) - 45*f(1) + 9*f(2) - f(3)) / (60*dx)
a06c_std = np.array([45.0, -9.0, 1.0]) / 60.0


# -------------------------------
# 9 POINTS
# -------------------------------
# Schéma numérique : df/dx ≈ (672*f(-1) - 168*f(-2) + 32*f(-3) - 3*f(-4) - 672*f(1) + 168*f(2) - 32*f(3) + 3*f(4)) / (840*dx)
a08c_std = np.array([672.0, -168.0, 32.0, -3.0]) / 840.0


# -------------------------------
# 11 POINTS
# -------------------------------
# Schéma numérique : df/dx ≈ (2100*f(-1) - 600*f(-2) + 150*f(-3) - 25*f(-4) + 2*f(-5) - (symétrique)) / (2520*dx)
a10c_std = np.array([2100.0, -600.0, 150.0, -25.0, 2.0]) / 2520.0

# -------------------------------
# 13 POINTS
# -------------------------------
# Schéma numérique : df/dx ≈ (23760*f(-1) - 7425*f(-2) + 2200*f(-3) - 495*f(-4) + 72*f(-5) - 5*f(-6) - (symétrique)) / (27720*dx)
a12c_std = np.array([23760.0, -7425.0, 2200.0, -495.0, 72.0, -5.0]) / 27720.0



