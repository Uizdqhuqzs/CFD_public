import numpy as np
import matplotlib.pyplot as plt
from modules_communs import lect_fichiers as lf
from scipy.optimize import least_squares

def au_z0(U_0, d_0):

    d = np.sqrt(2) - 1

    # -----------------------------------
    # 1. Extraction des données
    # -----------------------------------

    data = lf.read_file("VELO_field.txt")

    tol = np.min(np.abs(data["x_0"].values))

    data = data[data["x_0"].values == tol]

    z_full = data["x_1"].values
    u_full = data["u_z"].values

    # -----------------------------------
    # 2. Modèles analytiques
    # -----------------------------------

    def u_m(z, a_u, z_0, U_0, d_0):

        b_u = (3 * d)**0.5 / (2 * a_u)

        return b_u * U_0 * d_0 / (z - z_0)


    def residuals(params, z, u_CALIF3S, U_0, d_0):
        a_u, z_0 = params
        return u_m(z, a_u, z_0, U_0, d_0) - u_CALIF3S

    # -----------------------------------
    # 3. Calibration
    # -----------------------------------

    # zone de calibration


    data_cal = data[data["x_1"] >= 40*d_0]

    z_cal = data_cal["x_1"].values
    u_cal = data_cal["u_z"].values


    # calibration

    initial_guess = [0.12, 0.01] # a_u, z_0
    result = least_squares(residuals, initial_guess, args=(z_cal, u_cal, U_0, d_0), method='lm')

    a_u, z_0 = result.x

    print(f"\nParamètres calibrés : a_u = {a_u}, z_0 = {z_0} \n")

    # -----------------------------------
    # 4. Affichage résultats
    # -----------------------------------

    u_fit = u_m(z_full, a_u, z_0, U_0, d_0)

    plt.plot(z_full, u_full, 'o-', label='u_z CALIF3S')
    plt.plot(z_full, u_fit, 's--', label='u_z modèle')

    plt.title("Uz le long de l'axe ($U_m$)")
    plt.ylim(0, U_0)
    plt.xlabel("$\eta$")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()

    plt.show()

    return a_u, z_0

