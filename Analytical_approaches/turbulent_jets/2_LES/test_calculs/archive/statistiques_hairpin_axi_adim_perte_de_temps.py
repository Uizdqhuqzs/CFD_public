import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

# === Parametres ===
vti_file = "grfhrh.vti"
output_dir = "z_averaged_plots_adim"
dpi = 200  # qualite des images
z0 = 0.000001#.012854006829950816
d0 = 0.00612

# === Chargement du fichier VTI ===
grid = pv.read(vti_file)
Nx, Ny, Nz = grid.dimensions
dx, dy, dz = grid.spacing
ox, oy, oz = grid.origin

x = np.linspace(ox, ox + dx * (Nx - 1), Nx)
y = np.linspace(oy, oy + dy * (Ny - 1), Ny)
z = np.linspace(oz, oz + dz * (Nz - 1), Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')



# parsing données en shape du meshgrid

data_grid = {}

# On récupère les valeurs Z sur l'axe k=0 (car homogène dans XY pour ImageData)
Z_vals = Z[0, 0, :]  # Shape (Nz,)

# Recherche de l'indice k0 correspondant à z0
k0 = np.argmax(Z_vals >= z0)  # Premier k tel que Z >= z0

# Tronquage des grilles
Z = Z[:, :, k0:]
X = X[:, :, k0:]
Y = Y[:, :, k0:]

# Adaptation des champs
for name in grid.array_names:
    data = grid[name]

    if data.ndim == 1:
        arr = data.reshape((Nx, Ny, Nz), order='F')[:, :, k0:]
        data_grid[name] = arr

    elif data.ndim == 2 and data.shape[1] == 3:
        arr = data.reshape((Nx, Ny, Nz, 3), order='F')[:, :, k0:, :]
        data_grid[f"{name}_x"] = arr[..., 0]
        data_grid[f"{name}_y"] = arr[..., 1]
        data_grid[f"{name}_z"] = arr[..., 2]

    else:
        print(f"Ignoré : {name} (shape {data.shape})")





# === Fonctions ===
def moyenne_Z(Z_adim, DATA):
    Nx, Ny, Nz = DATA.shape

    # Calcul des intervalles en Z (taille Nz-1) le long de l'axe z=2
    dz = np.diff(Z_adim, axis=2)  # shape (Nx, Ny, Nz-1)

    avg_2D = np.zeros((Nx, Ny))

    for k in range(Nz - 1):
        # Pondération par dz[:,:,k] (shape Nx,Ny)
        avg_2D += 0.5 * (DATA[:, :, k] + DATA[:, :, k + 1]) * dz[:, :, k]

    # Calcul de la "longueur" totale selon Z, moyenne spatiale
    Lz = np.mean(Z_adim[:, :, -1] - Z_adim[:, :, 0])

    #print(f"Z_adim = {Z_adim.shape}")

    avg_2D /= Lz

    return avg_2D


def plot_scalar_field(X, Y, data2d, title, filename, cmap="viridis", dpi=200):

    #print(f"data2d {data2d}")

    #print(f"shape X {X.shape}, shape Y {Y.shape}, shape data {data2d.shape}")

    plt.figure(figsize=(6, 5))
    cs = plt.contourf(X, Y, data2d, levels=50, cmap=cmap)  # 50 niveaux par défaut, ajustable
    plt.colorbar(cs, label=title)
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
def delta(z_val):
    """Evasement de Nickels"""
    a_u = 0.12451280741522663
    d0 = np.sqrt(2) - 1
    z_0 = 0#.012854006829950816
    return a_u/np.sqrt(d0) * (z_val - z_0)

# === Creation dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)

# === Adimensionnement ===

ref_values = delta(Z)
ref_values[ref_values <= 0] = np.nan
#print(f"Shape delta {ref_values.shape}, shape X {X.shape}, shape Y {Y.shape}, shape Z {Z.shape}")

X_adim = X / ref_values#[np.newaxis, np.newaxis, :]
Y_adim = Y / ref_values#[np.newaxis, np.newaxis, :]
Z_adim = Z / d0 #ref_values

#plot_scalar_field(np.linspace(1,100,599), Y[0,:,0],Z_adim[0, :, :], "Z_adim", "Z_adim")

# === Moyenne en z ===

# Nouveau dictionnaire pour stocker les champs 2D moyennés
data_grid_zmoy = {}

for name in grid.array_names:
    
    # Cas scalaire
    if name in data_grid:
        arr = data_grid[name]
        avg = moyenne_Z(Z_adim, arr)  
        data_grid_zmoy[name] = avg

        plot_scalar_field(X_adim[:, 0, 0], Y_adim[0, :, 0], avg, f"{name} (moyenne en z)", f"{output_dir}/{name}_zmoy.png")

    # Cas vectoriel (3 composantes)
    elif all(f"{name}_{comp}" in data_grid for comp in ['x', 'y', 'z']):
        for comp in ['x', 'y', 'z']:
            arr = data_grid[f"{name}_{comp}"]
            avg = moyenne_Z(Z_adim, arr)
            data_grid_zmoy[f"{name}_{comp}"] = avg

            plot_scalar_field(X_adim[:, 0, 0], Y_adim[0, :, 0], avg, f"{name}.{comp} (moyenne en z)", f"{output_dir}/{name}_{comp}_zmoy.png")

    else:
        print(f"[!] Ignore : {name} (pas trouvé dans data_grid)")

print("\nTous les champs moyennés selon z sont prêts.")


# === Moyenne azimutale ===

r_max = np.sqrt(np.max(X)**2 + np.max(Y)**2)
Nr = 300
Ntheta = 360
Xp, Yp, r, theta = create_polar_grid(r_max, Nr, Ntheta)
polar_dir = os.path.join(output_dir, "azim_averaged")
os.makedirs(polar_dir, exist_ok=True)

# Grille X, Y pour l'interpolation 
X_grid = X_adim[:, 0, 0]
Y_grid = Y_adim[0, :, 0]


for name, champ_2d in data_grid_zmoy.items():
    
    print(f"\n[Azimutal] Traitement : {name}")

    data_polar = interpolate_on_polar_grid(champ_2d, X_grid, Y_grid, Xp, Yp)
    azim_avg = azimuthal_average(data_polar)

    np.save(os.path.join(polar_dir, f"{name}_theta_avg.npy"), np.column_stack((r, azim_avg)))

    plt.figure()
    plt.plot(r, azim_avg)
    plt.xlabel("r")
    plt.ylabel(f"{name} (moyenne azimutale)")
    plt.title(f"{name} - Moyenne azimutale en fonction de r")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(polar_dir, f"{name}_theta_avg.png"), dpi=dpi)
    plt.close()

print("\nTous les profils azimutaux sauvegardés dans :", polar_dir)
