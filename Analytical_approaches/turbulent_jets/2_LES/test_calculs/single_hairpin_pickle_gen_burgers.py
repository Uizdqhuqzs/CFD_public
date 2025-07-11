import time

start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

# --- Hairpin et Ring ---

def plot_parametrized_curve(curve):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='navy')

    x_min, y_min, z_min = curve.min(axis=0)
    x_max, y_max, z_max = curve.max(axis=0)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Courbe paramétrée unique - Hairpin Ring")
    plt.tight_layout()
    plt.show()

def hairpin_curve(zeta, a, b, h, phi):
    x = a * np.cos(zeta) + a
    y = h * np.exp(-zeta**2 / 2)
    z = np.zeros_like(zeta)
    mask1 = zeta < -np.pi/4
    mask2 = (zeta >= -np.pi/4) & (zeta <= np.pi/4)
    mask3 = zeta > np.pi/4
    z[mask1] = b * (zeta[mask1] + np.pi/4) * np.tan(phi) - np.sqrt(2)/2 * b
    z[mask2] = b * np.sin(zeta[mask2])
    z[mask3] = b * (zeta[mask3] - np.pi/4) * np.tan(phi) + np.sqrt(2)/2 * b
    return np.stack((y, z, x), axis=-1) # IL FAUT INVERSER Z

def compute_radial_spacing(b, N, phi):
    if N == 0 :
        return 0
    else :
        return b / (np.tan(np.pi/N))
    

# --- Burgers ---
def burgers_velocity_field(X, Y, Z, line_pts, Gamma=1.0, nu=15.6e-6, rc=0.1):
    """
    Champ de vitesse 3D d’un tube de vortex de Burgers par morceaux,
    avec paramétrage par le rayon de cœur rc.
    """
    shape = X.shape
    velocity = np.zeros(shape + (3,))
    points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

    alpha = 2 * nu / rc**2  # lien avec rayon de cœur

    for i in range(len(line_pts) - 1):
        p0 = line_pts[i]
        p1 = line_pts[i + 1]
        dp = p1 - p0
        L = np.linalg.norm(dp)
        if L < 1e-10:
            continue

        z_hat = dp / L
        rel_vec = points - p0
        s = np.dot(rel_vec, z_hat)
        proj = np.outer(s, z_hat)
        radial_vec = rel_vec - proj
        r = np.linalg.norm(radial_vec, axis=1)
        r_safe = np.where(r == 0, 1e-16, r)

        e_r = radial_vec / r_safe[:, None]
        e_theta = np.cross(z_hat, e_r)
        e_theta /= np.linalg.norm(e_theta, axis=1, keepdims=True)
        e_z = z_hat

        # Composantes du vortex de Burgers
        v_r = -alpha * r * L
        v_theta = (Gamma*L / (2 * np.pi * r_safe)) * (1 - np.exp(-r**2 / rc**2))
        v_z = alpha * s * L

        # Champ de vitesse local (cylindrique → cartésien)
        v_local = (v_r[:, None] * e_r +
                   v_theta[:, None] * e_theta +
                   v_z[:, None] * e_z)

        velocity += v_local.reshape(shape + (3,))

    return velocity

# --- MAIN ---

# Mesh
Nx, Ny, Nz = 200, 200, 1000
Lx, Ly, Lz = 20, 20, 800
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

# Paramètres de base
h = 1.5
rc = 0.1
Gamma = 1 # à calibrer
st = 0.15
U0 = 72
#MixLayThickness = delta_u * 0.06
#f = st*U0/(2*np.pi*MixLayThickness)
f = st  # couronnes par unité de longueur
a = 0.5
b = 0.3
nb_couronnes = int(Lz * f) 

# Hairpin de base (pour interpolation)
zeta = np.linspace(-np.pi, np.pi, 100)
base_curve = hairpin_curve(zeta, a, b, h, np.pi/6)
#plot_parametrized_curve(base_curve)

pickle_filename = "hairpin_rc_0p1_burgers.pkl"

if os.path.exists(pickle_filename):
    print(f" Le pickle {pickle_filename} est déjà généré.")

else:
    print(f"Aucun fichier {pickle_filename} trouvé. Calcul des interpolations...")
    # Calcul du champ de base (unique)
    v_base = burgers_velocity_field(X, Y, Z, base_curve, Gamma=1.0, nu=15.6e-6, rc=rc)

    interp_vx = RegularGridInterpolator((x, y, z), v_base[..., 0], bounds_error=False, fill_value=0)
    interp_vy = RegularGridInterpolator((x, y, z), v_base[..., 1], bounds_error=False, fill_value=0)
    interp_vz = RegularGridInterpolator((x, y, z), v_base[..., 2], bounds_error=False, fill_value=0)

    with open(pickle_filename, 'wb') as f:
        pickle.dump((interp_vx, interp_vy, interp_vz), f)
    print(f"Interpolations sauvegardées dans {pickle_filename}.")
