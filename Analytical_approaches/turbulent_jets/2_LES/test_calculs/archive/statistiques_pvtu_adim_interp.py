import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata

# --- Paramètres ---
pvtu_folder = "pvtu"
filename = os.path.join(pvtu_folder, "save.pvtu")
dpi = 600

d0 = 0.00612  # Longueur de référence pour z
U_0 = 72.5    # Vitesse de référence

# Grille manuelle
Nx, Ny = 1000, 1000  # Nombre de points en x et y

# Liste des champs à traiter
champsTraites = ["VELO"]

# Coefficients d'adimensionnement
field_scaling = {
    "VELO": U_0,
}

# --- Fonctions ---
def delta_z(z_val):
    """Demi-rayon du jet en fonction de z"""
    a_u = 0.12451280741522663
    z_0 = 0#.012854006829950816
    return a_u * (z_val - z_0) #* 1.126899696**-1


# --- Lecture du maillage ---
mesh = pv.read(filename)

print("Champs disponibles dans point_data :", list(mesh.point_data.keys()))
print("Champs disponibles dans cell_data  :", list(mesh.cell_data.keys()))

# Sélection de la source de données et des coordonnées
if len(mesh.point_data) > 0:
    data_source = "point_data"
    coords_raw = mesh.points
    source_data = mesh.point_data
elif len(mesh.cell_data) > 0:
    data_source = "cell_data"
    coords_raw = mesh.cell_centers().points
    source_data = mesh.cell_data
else:
    raise RuntimeError("Aucun champ détecté dans point_data ni cell_data.")

# --- Filtrage des points ---
x_raw, y_raw, z_raw = coords_raw[:, 0], coords_raw[:, 1], coords_raw[:, 2]

z0 = 0.012854006829950816
mask_valid = z_raw >= z0

coords = coords_raw[mask_valid]
x_raw = x_raw[mask_valid]
y_raw = y_raw[mask_valid]
z_raw = z_raw[mask_valid]

# Récupération et filtrage des champs
data_dict = {}
for key in source_data:
    data = source_data[key]
    if data.ndim == 1:
        data_dict[key] = data[mask_valid]
    elif data.ndim == 2:
        data_dict[key] = data[mask_valid, :]

print(f"Extraction des données depuis {data_source} avec {coords.shape[0]} points valides.")


delta = delta_z(z_raw)
z_adim = z_raw / d0
x_adim = x_raw / delta
y_adim = y_raw / delta

print(f"x max = {np.max(x_adim)}")

# Domaine 2D
x_min, x_max = np.min(x_adim), np.max(x_adim)
y_min, y_max = np.min(y_adim), np.max(y_adim)

xi = np.linspace(x_min, x_max, Nx)
yi = np.linspace(y_min, y_max, Ny)
XI, YI = np.meshgrid(xi, yi)

# --- Traitement ---
output_dir = "figures_pvtu_adim"
os.makedirs(output_dir, exist_ok=True)

z_unique = np.unique(z_adim)
z_unique.sort()

for name in champsTraites:
    
    if name not in data_dict:
        print(f"[AVERTISSEMENT] Champ '{name}' non présent dans les données. Ignoré.")
        continue

    data = data_dict[name]

    # Coefficient d'adimensionnement
    if name in field_scaling:
        coeff = field_scaling[name]
    else:
        coeff = 1.0
        print(f"[AVERTISSEMENT] Aucun coefficient pour '{name}'. Utilisation de 1.0.")

    data_adim = data / coeff

    # Champs vectoriels : traitement par composante
    if data_adim.ndim == 2 and data_adim.shape[1] == 3:
        
        for i, comp in enumerate(['x', 'y', 'z']):
            comp_name = f"{name}_{comp}"
            print(f"Traitement de la composante : {comp_name}")

            if comp_name != "VELO_z":
                continue

            sum_interp = np.zeros_like(XI, dtype=float)
            count = np.zeros_like(XI, dtype=float)

            for z_val in z_unique:
                mask = np.abs(z_adim - z_val) < 1e-6  # Plan z

                if np.sum(mask) < 3:
                    continue  # Pas assez de points pour interpoler

                x_plane = x_adim[mask]
                y_plane = y_adim[mask]
                values_plane = data_adim[mask, i]

                interp = griddata((x_plane, y_plane), values_plane, (XI, YI), method='linear')
                mask_interp = ~np.isnan(interp)
                sum_interp[mask_interp] += interp[mask_interp]
                count[mask_interp] += 1

            # Moyenne finale
            avg_field = np.full_like(sum_interp, np.nan)
            valid_mask = count > 0
            avg_field[valid_mask] = sum_interp[valid_mask] / count[valid_mask]

            # Sauvegarde
            plt.figure()
            plt.contourf(XI, YI, avg_field, levels=50, cmap='viridis')
            plt.colorbar()
            plt.title(f"{comp_name} (moyenne adim. selon z)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.xlim(-5, 5) 
            plt.ylim(-5, 5)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{comp_name}_avgz_adim.png", dpi=dpi)
            plt.close()

            # Sauvegarde de la coupe selon x = 0
            idx_x0 = np.argmin(np.abs(xi - 0))  # Trouve l'indice le plus proche de x=0
            coupe_y = yi  # Axe y adimensionné
            coupe_valeurs = avg_field[:, idx_x0]  # Valeurs de VELO_z le long de y à x=0

            # Combine en tableau (2 colonnes : y_adim, valeur)
            coupe_tableau = np.column_stack((coupe_y, coupe_valeurs))

            # Sauvegarde en .npy
            np.save(f"{output_dir}/{comp_name}_coupe_x0.npy", coupe_tableau)

            print(f"Coupe x=0 sauvegardée dans {output_dir}/{comp_name}_coupe_x0.npy")


    # Champs scalaires
    elif data_adim.ndim == 1:
        print(f"Traitement du champ scalaire : {name}")

        sum_interp = np.zeros_like(XI, dtype=float)
        count = np.zeros_like(XI, dtype=float)

        for z_val in z_unique:
            mask = np.abs(z_adim - z_val) < 1e-6

            if np.sum(mask) < 3:
                continue

            x_plane = x_adim[mask]
            y_plane = y_adim[mask]
            values_plane = data_adim[mask]

            interp = griddata((x_plane, y_plane), values_plane, (XI, YI), method='linear')
            mask_interp = ~np.isnan(interp)
            sum_interp[mask_interp] += interp[mask_interp]
            count[mask_interp] += 1

        avg_field = np.full_like(sum_interp, np.nan)
        valid_mask = count > 0
        avg_field[valid_mask] = sum_interp[valid_mask] / count[valid_mask]

        plt.figure()
        plt.contourf(XI, YI, avg_field, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(f"{name} (moyenne adim. selon z)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.xlim(-5, 5) 
        plt.ylim(-5, 5) 
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_avgz_adim.png", dpi=dpi)
        plt.close()

    else:
        print(f"[AVERTISSEMENT] Structure inattendue pour le champ '{name}'. Ignoré.")


print("Traitement terminé.")
