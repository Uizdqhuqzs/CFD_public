import numpy as np
import matplotlib.pyplot as plt
from modules_communs import lect_fichiers as lf
from scipy.optimize import least_squares

def ak_a2(U_0, d_0, z_0):

    # -----------------------------------
    # 1. Extraction des données
    # -----------------------------------

    data = lf.read_file("K_field.txt")

    tol = np.min(np.abs(data["x_0"].values))

    data = data[data["x_0"].values == tol]   # Extraction de k le long de l'axe

    z_full = data["x_1"].values
    k_full = data["k"].values


    # -----------------------------------
    # 2. Modèles analytiques
    # -----------------------------------

    def k_m(z_adim, a_k, a_2):

        return a_k / z_adim**a_2


    def residuals(params, z_adim, k_axe_adim):
        a_k, a_2 = params
        return k_m(z_adim, a_k, a_2) - k_axe_adim

    # -----------------------------------
    # 3. Calibration
    # -----------------------------------

    # zone de calibration


    data_cal = data[data["x_1"] >= 40*d_0]

    z_cal = data_cal["x_1"].values
    k_cal = data_cal["k"].values

    # adimensionnement

    k_adim = k_cal / (U_0)**2

    z_adim = (z_cal - z_0) / d_0

    z_adim_full = (z_full - z_0) / d_0

    # calibration

    initial_guess = [0.12, 0.01] # a_k, a2
    result = least_squares(residuals, initial_guess, args=(z_adim, k_adim), method='lm')

    a_k, a_2 = result.x

    print(f"\nParamètres calibrés : a_k = {a_k}, a_2 = {a_2} \n")

    # -----------------------------------
    # 4. Affichage résultats
    # -----------------------------------

    k_fit = k_m(z_adim_full, a_k, a_2) * U_0**2

    plt.plot(z_full, k_full, 'o-', label='u_z CALIF3S')
    plt.plot(z_full, k_fit, 's--', label='u_z modèle')

    plt.title("Uz le long de l'axe ($U_m$)")
    plt.ylim(0, np.max(k_full))
    plt.xlabel("$\eta$")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()

    plt.show()

    return a_k, a_2


