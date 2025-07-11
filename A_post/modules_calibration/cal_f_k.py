import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import least_squares
from modules_communs import lect_fichiers as rf

def bk1_bk2(U_0, d_0, a_u, z_0, a_k, a_2):

    # -----------------------------------
    # 1. Lecture du fichier k_field.txt
    # -----------------------------------

    data = rf.read_file("k_field.txt")

    r_full = data["x_0"].values
    z_full = data["x_1"].values
    k_full = data["k"].values

    eta_full = r_full / (a_u * (z_full - z_0))


    # zone de calibration

    data_cal = data[data["x_0"] / (data["x_1"] - z_0) <= 0.11]   # eta selon Hussein 1994
    data_cal = data_cal[data_cal["x_1"] >= 40*d_0]

    r_cal = data_cal["x_0"].values
    z_cal = data_cal["x_1"].values
    k_cal = data_cal["k"].values

    eta_cal = r_cal / (a_u * (z_cal - z_0))



    # -----------------------------------
    # 2. Mise en forme des données
    # -----------------------------------

    # adimensionnement 

    k_adim = k_cal / (U_0)**2

    z_adim = (z_cal - z_0) / d_0

    z_adim_full = (z_full - z_0) / d_0

    # -----------------------------------
    # 2. Modèles analytiques
    # -----------------------------------

    def k_m(z_adim, a_k, a_2):
        return a_k / z_adim**a_2

    def f_k(eta, bk1, bk2):

        return bk1 * (eta**2 + 1) * np.exp(-eta**2 + bk2 - np.log(bk1*np.exp(bk2)))

    def k_model(params, z_adim, eta_cal):
        bk1, bk2 = params
        return k_m(z_adim, a_k, a_2) * f_k(eta_cal, bk1, bk2)


    def residuals(params, z_adim, eta_cal, k_axe_adim):
        return k_model(params, z_adim, eta_cal) - k_axe_adim

    # -----------------------------------
    # 4. Calibration
    # -----------------------------------
    initial_guess = [1, 1] 
    result = least_squares(residuals, initial_guess, args=(z_adim, eta_cal, k_adim), method='lm')

    #result.x = [1, 0, 1, 0]

    b_k1, b_k2 = result.x

    print(f"\nParamètres calibrés : b_k1 = {b_k1}, b_k2 = {b_k2} \n")


    # -----------------------------------
    # 5. Affichage résultats
    # -----------------------------------

    # 1ère figure : k CALIF3S vs k modélisé et différence
    k_fit = k_model(result.x, z_adim, eta_cal)
    diff = np.abs(k_fit - k_adim)
    diff[diff>10] = 10

    plt.figure(figsize=(15, 4))

    # Champ CALIF3S
    plt.subplot(1, 3, 1)
    plt.tricontourf(eta_cal, z_adim, k_adim, levels=20)
    plt.title("k_CALIF3S")
    plt.xlabel("η")
    plt.ylabel("z_adim")
    plt.colorbar()

    # Champ modélisé
    plt.subplot(1, 3, 2)
    plt.tricontourf(eta_cal, z_adim, k_fit, levels=20)
    plt.title("k_modele")
    plt.xlabel("η")
    plt.ylabel("z_adim")
    plt.colorbar()

    # Différence
    plt.subplot(1, 3, 3)
    contours = plt.tricontourf(eta_cal, z_adim, diff, levels=20, cmap='bwr')  # Remplissage coloré
    plt.title("erreur")
    plt.xlabel("η")
    plt.ylabel("z_adim")
    plt.colorbar(contours)

    plt.tight_layout()
    plt.show()

    # 2ème figure : repère r-z

    k_fit_full = k_model(result.x, z_adim_full, eta_full) * (U_0)**2

    plt.figure(figsize=(15, 4))

    # Champ CALIF3S
    plt.subplot(1, 3, 1)
    plt.tricontourf(z_full, r_full, k_full, levels=20, cmap='viridis')
    plt.title("Champ k_CALIF3S")
    plt.xlabel("z")
    plt.ylabel("r")
    plt.colorbar(label="k")

    # Champ modélisé
    max_CALIF3S = np.max(k_full)
    plt.subplot(1, 3, 2)
    k_fit_full[k_fit_full > max_CALIF3S] = max_CALIF3S
    k_fit_full = np.nan_to_num(k_fit_full, nan=max_CALIF3S, posinf=max_CALIF3S, neginf=0)
    plt.tricontourf(z_full, r_full, k_fit_full, levels=20, cmap='viridis')
    plt.title("Champ k_modele")
    plt.xlabel("z")
    plt.ylabel("r")
    plt.colorbar(label="k")

    # Erreur
    err_full = np.abs((k_fit_full - k_full))
    err_full[err_full > max_CALIF3S*0.1] = max_CALIF3S*0.1
    plt.subplot(1, 3, 3)
    contour = plt.tricontourf(z_full, r_full, err_full, levels=20, cmap='bwr')
    plt.title("Champ d'erreur")
    plt.xlabel("z")
    plt.ylabel("r")

    #np.savetxt("erreur.txt", np.column_stack((err_full)), fmt="%.6f", delimiter=" ", header="# eta valeur", comments="")



    plt.colorbar(contour, label="Erreur ", extend='both')

    plt.tight_layout()
    plt.show()


    #print(f"Taille de r : {np.size(r)}, Taille de z : {np.size(z)}, Taille de eta : {np.size(eta)}, Taille de k_CALIF3S : {np.size(k_CALIF3S)}, Taille de k_fit : {np.size(k_fit)} \n")


    # 3e figure : coupe

    # Slider setup
    z_min = 20 * d_0
    z_max = np.max(z_full)
    z_init = 60 * d_0
    tol = 1e-6

    # Préparation du plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    line1, = ax.plot([], [], 'o-', label='k CALIF3S')
    line2, = ax.plot([], [], 's--', label='k modèle')
    title = ax.set_title("")
    ax.set_xlabel("η (Hussein, 1994)")
    ax.set_ylabel("k")
    ax.grid(True)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1.3)
    ax.legend()

    # Slider axes
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider_z = Slider(ax_slider, "z_coupe_target", valmin=z_min, valmax=z_max, valinit=z_init, valstep=0.001)

    # Fonction de mise à jour
    def update(val):
        z_coupe_target = slider_z.val
        idx_nearest = np.argmin(np.abs(z_full - z_coupe_target))
        z_coupe = z_full[idx_nearest]
        mask = np.isclose(z_full, z_coupe, atol=tol)

        if not np.any(mask):
            line1.set_data([], [])
            line2.set_data([], [])
            title.set_text(f"Aucun point trouvé pour z ≈ {z_coupe:.4f}")
        else:
            eta_slice = eta_full[mask] * a_u     # *a_u pour correspondre au energy balance de la littérature
            k_CALIF3S_slice = (k_full/k_m(z_adim_full, a_k, a_2)/U_0**2)[mask]
            k_fit_slice = f_k(eta_full,b_k1,b_k2)[mask]

            sort_idx = np.argsort(eta_slice)
            eta_sorted = eta_slice[sort_idx]
            k_CALIF3S_sorted = k_CALIF3S_slice[sort_idx]
            k_fit_sorted = k_fit_slice[sort_idx]

            line1.set_data(eta_sorted, k_CALIF3S_sorted)
            line2.set_data(eta_sorted, k_fit_sorted)
            title.set_text(f'f_kz vs k_CALIF3S adimentionné à z = {z_coupe:.4f}')
            '''ax.set_xlim(np.min(eta_sorted), np.max(eta_sorted))'''
            '''ax.set_ylim(
                min(np.min(k_CALIF3S_sorted), np.min(k_fit_sorted)) - 0.1,
                max(np.max(k_CALIF3S_sorted), np.max(k_fit_sorted)) + 0.1
            )'''

        fig.canvas.draw_idle()

    # Lier le slider
    slider_z.on_changed(update)

    # Initial update
    update(z_init)

    plt.show()

    return b_k1, b_k2