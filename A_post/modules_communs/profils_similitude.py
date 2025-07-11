import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constantes as cte

# -----------------------------------
# Vitesses
# -----------------------------------

def u_m(z):
    return cte.b_u * cte.U_0 * cte.d_0 / (z - cte.z_0)

def f_u(eta):
    return 1/(cte.d*(eta)**2 + 1)**2

def uz_model(z, eta):
    return u_m(z) * f_u(eta)

# -----------------------------------
# Energie cinétique turbulente
# -----------------------------------

def k_m(z):

    return cte.a_k / z**cte.a_2

def g(eta): # G*(eta)

    return - (cte.a_u/2 * (2 / (cte.d* eta**2 + 1) + np.log(cte.d* eta**2 + 1))) / (2 * cte.d)

def f_k(eta):

    return cte.b_k1 * (eta**2 + 1) * np.exp(-eta**2 + cte.b_k2 - np.log(cte.b_k1*np.exp(cte.b_k2)))

def k_model(z, eta):
    return k_m(z) * f_k(eta)