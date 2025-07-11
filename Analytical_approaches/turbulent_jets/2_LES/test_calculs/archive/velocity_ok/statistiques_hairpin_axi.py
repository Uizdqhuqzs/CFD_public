import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

# === Parametres ===
vti_file = "grfhrh.vti"
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

def cartesian_to_polar(X, Y):
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    return R, Theta

def create_polar_grid(r_max, Nr, Ntheta):
    r = np.linspace(0, r_max, Nr)
    theta = np.linspace(-np.pi, np.pi, Ntheta)
    R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')
    Xp = R_grid * np.cos(Theta_grid)
    Yp = R_grid * np.sin(Theta_grid)
    return Xp, Yp, r, theta

def interpolate_on_polar_grid(avg_field, x, y, Xp, Yp):
    interp = RegularGridInterpolator((x, y), avg_field, bounds_error=False, fill_value=np.nan)
    pts = np.stack((Xp.ravel(), Yp.ravel()), axis=-1)
    data_polar = interp(pts).reshape(Xp.shape)
    return data_polar

def azimuthal_average(data_polar):
    return np.nanmean(data_polar, axis=1)

# --- Fonctions ---

# === Creation dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)

# === Traitement des champs ===
for name in grid.array_names:
    data = grid[name]

    if data.ndim == 1:
        arr = data.reshape((Nx, Ny, Nz), order='F')
        avg = average_along_z(arr)
        plot_scalar_field(avg, f"{name} (moyenne en z)", f"{output_dir}/{name}_zmoy.png")

        np.save(f"{output_dir}/{name}_zmoy.npy", avg)

        '''idx_x0 = np.argmin(np.abs(x - 0))
        coupe_y = y
        coupe_valeurs = avg[idx_x0, :]

        coupe_tableau = np.column_stack((coupe_y, coupe_valeurs))
        np.save(f"{output_dir}/{name}_coupe_x0.npy", coupe_tableau)'''

    elif data.ndim == 2 and data.shape[1] == 3:
        vec = data.reshape((Nx, Ny, Nz, 3), order='F')
        for i, comp in enumerate('xyz'):
            avg = average_along_z(vec[..., i])
            plot_scalar_field(avg, f"{name}.{comp} (moyenne en z)", f"{output_dir}/{name}_{comp}_zmoy.png")

            np.save(f"{output_dir}/{name}_{comp}_zmoy.npy", avg)

            '''idx_x0 = np.argmin(np.abs(x - 0))
            coupe_y = y
            coupe_valeurs = avg[idx_x0, :]

            coupe_tableau = np.column_stack((coupe_y, coupe_valeurs))
            np.save(f"{output_dir}/{name}_{comp}_coupe_x0.npy", coupe_tableau)'''
    else:
        print(f"[!] Ignore : {name} (shape {data.shape})")

print("\nTous les champs moyennés selon z sauvegardés dans :", output_dir)

# === Azimutal average ===
r_max = np.sqrt(np.max(x)**2 + np.max(y)**2)
Nr = 300
Ntheta = 360
Xp, Yp, r, theta = create_polar_grid(r_max, Nr, Ntheta)
polar_dir = os.path.join(output_dir, "azim_averaged")
os.makedirs(polar_dir, exist_ok=True)

for file in os.listdir(output_dir):
    if file.endswith("_zmoy.npy") and "coupe" not in file:
        champ_path = os.path.join(output_dir, file)
        champ_name = file.replace("_zmoy.npy", "")

        print(f"\n[Azimutal] Traitement : {champ_name}")

        champ_2d = np.load(champ_path)
        data_polar = interpolate_on_polar_grid(champ_2d, x, y, Xp, Yp)
        azim_avg = azimuthal_average(data_polar)

        np.save(os.path.join(polar_dir, f"{champ_name}_azim_avg.npy"), np.column_stack((r, azim_avg)))

        plt.figure()
        plt.plot(r, azim_avg)
        plt.xlabel("r")
        plt.ylabel(f"{champ_name} (moyenne azimutale)")
        plt.title(f"{champ_name} - Moyenne azimutale en fonction de r")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(polar_dir, f"{champ_name}_azim_avg.png"), dpi=dpi)
        plt.close()

print("\nTous les profils azimutaux sauvegardés dans :", polar_dir)
