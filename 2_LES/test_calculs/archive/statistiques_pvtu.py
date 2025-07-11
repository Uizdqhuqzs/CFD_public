import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# --- Chemin vers le fichier .pvtu ---
pvtu_folder = "pvtu"
filename = os.path.join(pvtu_folder, "save.pvtu")  # adapte le nom
dpi = 1000

mesh = pv.read(filename)

# --- Afficher les champs disponibles ---
print("Champs disponibles dans point_data :", list(mesh.point_data.keys()))
print("Champs disponibles dans cell_data  :", list(mesh.cell_data.keys()))
print("Champs disponibles dans field_data :", list(mesh.field_data.keys()))

# Choisir d’où extraire les champs : point_data ou cell_data (ou adapter)
if len(mesh.point_data) > 0:
    data_source = "point_data"
elif len(mesh.cell_data) > 0:
    data_source = "cell_data"
else:
    raise RuntimeError("Aucun champ détecté dans point_data ni cell_data.")

# --- Récupération des coordonnées et champs ---
if data_source == "point_data":
    coords = mesh.points
    data_dict = mesh.point_data
elif data_source == "cell_data":
    coords = mesh.cell_centers().points
    data_dict = mesh.cell_data

x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

print(f"Extraction des données depuis {data_source} avec {coords.shape[0]} points.")

# --- Reste du code inchangé ---
# ... (ici tu remets ta fonction hash_xy, average_along_z, etc.)

# Je te remets la fonction hash_xy et average_along_z pour cohérence

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
output_dir = "figures_pvtu"
os.makedirs(output_dir, exist_ok=True)

for name in data_dict:
    data = data_dict[name]

    if data.ndim == 1:
        print(f"Champ scalaire : {name}")
        x2d, y2d, avg = average_along_z(data, z, xy_index)

        plt.figure()
        plt.tricontourf(x2d, y2d, avg, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(f"{name} (moyenne selon z)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_avgz.png",dpi=dpi)
        plt.close()

    elif data.ndim == 2 and data.shape[1] == 3:
        for i, comp in enumerate(['x', 'y', 'z']):
            print(f"Champ vectoriel : {name}_{comp}")
            x2d, y2d, avg = average_along_z(data[:, i], z, xy_index)

            plt.figure()
            plt.tricontourf(x2d, y2d, avg, levels=50, cmap='viridis')
            plt.colorbar()
            plt.title(f"{name}_{comp} (moyenne selon z)")
            plt.xlabel("x"); plt.ylabel("y")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{name}_{comp}_avgz.png", dpi=dpi)
            plt.close()
