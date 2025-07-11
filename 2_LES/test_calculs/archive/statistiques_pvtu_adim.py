import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# --- Chemin vers le fichier .pvtu ---
pvtu_folder = "pvtu"
filename = os.path.join(pvtu_folder, "save.pvtu")  # adapte le nom
dpi = 1000

# --- Paramètres d'adimensionnement ---
d0 = 0.00612  # longueur de référence pour z (ex: hauteur caractéristique)
U_0 = 72.5
field_scaling = {
    "VELO_x": U_0,  # remplacer par les noms de champs et leurs coefficients
    "VELO_y": U_0,
    "VELO_z": U_0,
    "VELO": U_0,
    # ...
}

def delta_z(z_val):
    """demi-rayon du jet"""
    a_u = 0.12451280741522663
    z_0 = 0#0.012854006829950816
    return a_u*(z_val - z_0)

# --- Lecture du maillage ---
mesh = pv.read(filename)

print("Champs disponibles dans point_data :", list(mesh.point_data.keys()))
print("Champs disponibles dans cell_data  :", list(mesh.cell_data.keys()))
print("Champs disponibles dans field_data :", list(mesh.field_data.keys()))

if len(mesh.point_data) > 0:
    data_source = "point_data"
elif len(mesh.cell_data) > 0:
    data_source = "cell_data"
else:
    raise RuntimeError("Aucun champ détecté dans point_data ni cell_data.")

if data_source == "point_data":
    coords = mesh.points
    data_dict = mesh.point_data
elif data_source == "cell_data":
    coords = mesh.cell_centers().points
    data_dict = mesh.cell_data

# --- Extraction et adimensionnement des coordonnées ---
x_raw, y_raw, z_raw = coords[:, 0], coords[:, 1], coords[:, 2]
r_raw = np.sqrt(x_raw**2 + y_raw**2)

delta = delta_z(z_raw)
z = z_raw / d0
x = x_raw / delta
y = y_raw / delta

print(f"Extraction des données depuis {data_source} avec {coords.shape[0]} points.")

# --- Fonctions auxiliaires ---
tol = 1e-5

def hash_xy(x, y, tol):
    return (np.round(x / tol).astype(int), np.round(y / tol).astype(int))

xy_hash = hash_xy(x, y, tol)
xy_index = defaultdict(list)
for idx, key in enumerate(zip(*xy_hash)):
    xy_index[key].append(idx)

print(f"Nombre de points (x,y) uniques ~ {len(xy_index)}")

def average_along_z(values, z_vals, indices_group):
    avg = []
    x_out = []
    y_out = []

    for key, indices in indices_group.items():
        z_group = z_vals[indices]
        val_group = values[indices]

        if len(z_group) < 2:
            continue

        sort_idx = np.argsort(z_group)
        z_sorted = z_group[sort_idx]
        val_sorted = val_group[sort_idx]

        dz = np.diff(z_sorted)
        dz = np.concatenate([[dz[0]], dz])

        weighted = val_sorted * dz
        avg_val = np.sum(weighted) / np.sum(dz)

        x_out.append(x[indices[0]])
        y_out.append(y[indices[0]])
        avg.append(avg_val)

    return np.array(x_out), np.array(y_out), np.array(avg)

# --- Calcul et sauvegarde des moyennes pour chaque champ ---
output_dir = "figures_pvtu_adim"
os.makedirs(output_dir, exist_ok=True)

for name in data_dict:
    data = data_dict[name]

    # Adimensionnement du champ
    if name in field_scaling:
        coeff = field_scaling[name]
        print(f"Traitement de '{name}'. Utilisation de coeff = {coeff}.")
    else:
        coeff = 1.0
        print(f"[AVERTISSEMENT] Aucun coefficient d'adimensionnement spécifié pour '{name}'. Utilisation de coeff = 1.0.")

    data_adim = data / coeff
 
    if data_adim.ndim == 1:
        print(f"Champ scalaire : {name}")
        x2d, y2d, avg = average_along_z(data_adim, z, xy_index)

        plt.figure()
        plt.tricontourf(x2d, y2d, avg, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(f"{name} (moyenne adim. selon z)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_avgz_adim.png", dpi=dpi)
        plt.close()

    elif data_adim.ndim == 2 and data_adim.shape[1] == 3:
        for i, comp in enumerate(['x', 'y', 'z']):
            comp_name = f"{name}_{comp}"
            print(f"Champ vectoriel : {comp_name}")
            x2d, y2d, avg = average_along_z(data_adim[:, i], z, xy_index)

            plt.figure()
            plt.tricontourf(x2d, y2d, avg, levels=50, cmap='viridis')
            plt.colorbar()
            plt.title(f"{comp_name} (moyenne adim. selon z)")
            plt.xlabel("x"); plt.ylabel("y")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{comp_name}_avgz_adim.png", dpi=dpi)
            plt.close()
