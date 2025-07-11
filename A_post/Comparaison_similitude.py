import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from modules_communs import lect_fichiers as lf
from modules_communs import constantes as cte

# -----------------------------------
# 1. Parsing
# -----------------------------------

def projection_axe(x_vals, y_vals, A_vals, y0):

    atol=np.min(np.abs(y_vals)) + 1e-6

    masu_y0 = np.isclose(y_vals, y0, atol=atol)
    x_unique = x_vals[masu_y0]
    A_y0 = A_vals[masu_y0]
    
    # Dictionnaire x -> A(x, y0)
    x_to_Ay0 = dict(zip(x_unique, A_y0))

    # Construction de A' avec substitution
    A_prime = np.array([x_to_Ay0.get(x, np.nan) for x in x_vals])
    
    return A_prime

data_0 = lf.read_file("k_calc.txt")

r_full_0 = data_0["x_0"].values
z_full_0 = data_0["x_1"].values
k_full_0 = data_0["k"].values
k_m_0 = projection_axe(z_full_0, r_full_0, k_full_0, 0)

print(k_m_0)

data_1 = lf.read_file("k_mod.txt")

r_full_1 = data_1["x_0"].values
z_full_1 = data_1["x_1"].values
k_full_1 = data_1["k"].values
k_m_1 = projection_axe(z_full_1, r_full_1, k_full_1, 0)

# -----------------------------------
# 2. Mise en forme des données
# -----------------------------------

# adimensionnement 

z_adim_full_0 = (z_full_0 - cte.z_0) / cte.d_0
eta_full_0 = r_full_0 / (cte.a_u * (z_full_0 - cte.z_0))
k_adim_0 = k_full_0 / cte.U_0**2 / k_m_0

z_adim_full_1 = (z_full_1 - cte.z_0) / cte.d_0
eta_full_1 = r_full_1 / (cte.a_u * (z_full_1 - cte.z_0))
k_adim_1 = k_full_1 / cte.U_0**2 / k_m_1

# Traitement des NaN

k_adim_0 = np.nan_to_num(k_adim_0, nan=1, posinf=1, neginf=0)

max_CALIF3S = 10 #np.max(k_adim_0)
k_adim_1[k_adim_1 > max_CALIF3S] = max_CALIF3S
k_adim_1 = np.nan_to_num(k_adim_1, nan=max_CALIF3S, posinf=max_CALIF3S, neginf=0)



# -----------------------------------
# 3. Comparaison
# -----------------------------------


# 1ère figure : k CALIF3S vs k modélisé et différence

diff = np.abs(k_adim_0 - k_adim_1)
diff[diff>10] = 10

plt.figure(figsize=(15, 4))

# Champ CALIF3S
plt.subplot(1, 3, 1)
plt.tricontourf(eta_full_0, z_adim_full_0, k_adim_0, levels=20)
plt.title("k_CALIF3S")
plt.xlabel("η")
plt.ylabel("z_adim")
plt.colorbar()

# Champ modélisé
plt.subplot(1, 3, 2)
plt.tricontourf(eta_full_0, z_adim_full_0, k_adim_1, levels=20)
plt.title("k_modele")
plt.xlabel("η")
plt.ylabel("z_adim")
plt.colorbar()

# Différence
plt.subplot(1, 3, 3)
contours = plt.tricontourf(eta_full_0, z_adim_full_0, diff, levels=20, cmap='bwr')  # Remplissage coloré
plt.title("erreur")
plt.xlabel("η")
plt.ylabel("z_adim")
plt.colorbar(contours)

plt.tight_layout()
plt.show()

# 2ème figure : repère r-z

plt.figure(figsize=(15, 4))

# Champ CALIF3S
plt.subplot(1, 3, 1)
plt.tricontourf(z_full_0, r_full_0, k_full_0, levels=20, cmap='viridis')
plt.title("Champ k_CALIF3S")
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar(label="k")

# Champ modélisé
plt.subplot(1, 3, 2)
plt.tricontourf(z_full_0, r_full_0, k_full_1, levels=20, cmap='viridis')
plt.title("Champ k_modele")
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar(label="k")

# Erreur
err_full = np.abs((k_full_0 - k_full_1))
err_full[err_full > max_CALIF3S*0.1] = max_CALIF3S*0.1
plt.subplot(1, 3, 3)
contour = plt.tricontourf(z_full_0, r_full_0, err_full, levels=20, cmap='bwr')
plt.title("Champ d'erreur")
plt.xlabel("z")
plt.ylabel("r")

#np.savetxt("erreur.txt", np.column_stack((err_full)), fmt="%.6f", delimiter=" ", header="# eta valeur", comments="")



plt.colorbar(contour, label="Erreur ", extend='both')

plt.tight_layout()
plt.show()


#print(f"Taille de r : {np.size(r)}, Taille de z : {np.size(z)}, Taille de eta : {np.size(eta)}, Taille de k_CALIF3S : {np.size(k_CALIF3S)}, Taille de k_fit : {np.size(k_fit)} \n")


# 3e figure : coupe

print(np.max(k_adim_0), np.max(k_adim_1))

# Slider setup
z_min = 60 * cte.d_0
z_max = np.max(z_full_0)
z_init = 60 * cte.d_0
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
#ax.set_xlim(-1, 10)
ax.set_ylim(0, np.max(k_adim_0)*cte.U_0**2)
ax.legend()

# Slider axes
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_z = Slider(ax_slider, "z_coupe_target", valmin=z_min, valmax=z_max, valinit=z_init, valstep=0.001)

# Fonction de mise à jour
def update(val):
    z_coupe_target = slider_z.val
    idx_nearest = np.argmin(np.abs(z_full_0 - z_coupe_target))
    z_coupe = z_full_0[idx_nearest]
    mask = np.isclose(z_full_0, z_coupe, atol=tol)

    if not np.any(mask):
        line1.set_data([], [])
        line2.set_data([], [])
        title.set_text(f"Aucun point trouvé pour z ≈ {z_coupe:.4f}")
    else:
        eta_slice = eta_full_0[mask] * cte.a_u    
        k_CALIF3S_slice = (k_adim_0)[mask]*cte.U_0**2
        k_fit_slice = (k_adim_1)[mask]*cte.U_0**2

        sort_idx = np.argsort(eta_slice)
        eta_sorted = eta_slice[sort_idx]
        k_CALIF3S_sorted = k_CALIF3S_slice[sort_idx]
        k_fit_sorted = k_fit_slice[sort_idx]

        line1.set_data(eta_sorted, k_CALIF3S_sorted)
        line2.set_data(eta_sorted, k_fit_sorted)
        title.set_text(f'f_kz vs k_CALIF3S adimentionné à z = {z_coupe:.4f}')
        ax.set_xlim(np.min(eta_sorted), np.max(eta_sorted))
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

contour = plt.tricontourf(z_full_0, r_full_0, k_m_0 - k_m_1, levels=20, cmap='viridis')
plt.colorbar(contour, label="Erreur ", extend='both')
plt.xlabel("z")
plt.ylabel('r')
plt.legend()
plt.title(f"Évolution de k_m")
plt.grid()
plt.show()