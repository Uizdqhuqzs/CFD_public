import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os

# === Parametres ===
vti_file = "hairpin_ring_field_longer_light.vti"
output_dir = "z_averaged_plots"
dpi = 200  # qualite des images

# === Chargement du fichier VTI ===
grid = pv.read(vti_file)
Nx, Ny, Nz = grid.dimensions
dx, dy, dz = grid.spacing
ox, oy, oz = grid.origin

x = np.linspace(ox, ox + dx * (Nx - 1), Nx)
y = np.linspace(oy, oy + dy * (Ny - 1), Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# === Fonctions ===
def average_along_z(arr_3d):
    return np.mean(arr_3d, axis=2)

def plot_scalar_field(data2d, title, filename, cmap="viridis"):
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, data2d.T, shading='auto', cmap=cmap)
    plt.colorbar(label=title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


# === Creation dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)

# === Traitement des champs ===
for name in grid.array_names:
    data = grid[name]

    if data.ndim == 1:
        arr = data.reshape((Nx, Ny, Nz), order='F')
        avg = average_along_z(arr)
        plot_scalar_field(avg, f"{name} (moyenne en z)", f"{output_dir}/{name}_zmoy.png")

        # Coupe x = 0
        idx_x0 = np.argmin(np.abs(x - 0))
        coupe_y = y
        coupe_valeurs = avg[idx_x0, :]

        coupe_tableau = np.column_stack((coupe_y, coupe_valeurs))
        np.save(f"{output_dir}/{name}_coupe_x0.npy", coupe_tableau)
        print(f"Coupe x=0 sauvegardée : {output_dir}/{name}_coupe_x0.npy")


    elif data.ndim == 2 and data.shape[1] == 3:
        vec = data.reshape((Nx, Ny, Nz, 3), order='F')
        for i, comp in enumerate('xyz'):
            avg = average_along_z(vec[..., i])
            plot_scalar_field(avg, f"{name}.{comp} (moyenne en z)", f"{output_dir}/{name}_{comp}_zmoy.png")

            # Coupe x = 0
            idx_x0 = np.argmin(np.abs(x - 0))
            coupe_y = y
            coupe_valeurs = avg[idx_x0, :]

            coupe_tableau = np.column_stack((coupe_y, coupe_valeurs))
            np.save(f"{output_dir}/{name}_{comp}_coupe_x0.npy", coupe_tableau)
            print(f"Coupe x=0 sauvegardée : {output_dir}/{name}_{comp}_coupe_x0.npy")


    else:
        print(f"[!] Ignore : {name} (shape {data.shape})")

print("\n Tous les champs moyennes selon z et sauvegardes dans :", output_dir)
