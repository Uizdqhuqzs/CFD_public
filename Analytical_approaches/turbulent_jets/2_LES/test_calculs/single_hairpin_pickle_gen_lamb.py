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
    

# --- Lamb–Oseen ---
def lamb_oseen_line_velocity_field(X, Y, Z, line_pts, Gamma=1.0, rc=0.1):
    shape = X.shape
    velocity = np.zeros(shape + (3,))
    points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    for i in range(len(line_pts) - 1):
        p0 = line_pts[i]
        p1 = line_pts[i + 1]
        dp = p1 - p0
        L = np.linalg.norm(dp)
        if L < 1e-10:
            continue
        t_hat = dp / L
        rel_vec = points - p0
        s = np.dot(rel_vec, t_hat)
        proj = np.outer(s, t_hat)
        radial_vec = rel_vec - proj
        r = np.linalg.norm(radial_vec, axis=1)
        r_safe = np.where(r == 0, 1e-16, r)
        u_theta = (Gamma * L) / (2 * np.pi * r_safe) * (1 - np.exp(-r**2 / rc**2))
        azim_dir = np.cross(t_hat, radial_vec)
        azim_norm = np.linalg.norm(azim_dir, axis=1, keepdims=True)
        azim_unit = np.divide(azim_dir, azim_norm, out=np.zeros_like(azim_dir), where=azim_norm!=0)
        v_contrib = u_theta[:, None] * azim_unit
        velocity += v_contrib.reshape(shape + (3,))
    return velocity

# --- MAIN ---

# Mesh
Nx, Ny, Nz = 100, 100, 800   #Nx, Ny, Nz = 250, 250, 1000
Lx, Ly, Lz = 40, 40, 2000
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

# Paramètres de base
h = 1.5
rc = 0.05
Gamma = 1 # à calibrer
a = 0.5
b = 0.5

# Hairpin de base (pour interpolation)
zeta = np.linspace(-np.pi, np.pi, 100)
base_curve = hairpin_curve(zeta, a, b, h, np.pi/6)
#plot_parametrized_curve(base_curve)

pickle_filename = "hairpin_rc_0p05.pkl"

if os.path.exists(pickle_filename):
    print(f"{pickle_filename} déjà généré !")
    with open(pickle_filename, 'rb') as f:
        interp_vx, interp_vy, interp_vz = pickle.load(f)
else:
    print(f"Génération de {pickle_filename}.")
    # Calcul du champ de base (unique)
    v_base = lamb_oseen_line_velocity_field(X, Y, Z, base_curve, Gamma=Gamma, rc=rc)

    print("Calcul des interpolations...")

    interp_vx = RegularGridInterpolator((x, y, z), v_base[..., 0], bounds_error=False, fill_value=0)
    interp_vy = RegularGridInterpolator((x, y, z), v_base[..., 1], bounds_error=False, fill_value=0)
    interp_vz = RegularGridInterpolator((x, y, z), v_base[..., 2], bounds_error=False, fill_value=0)

    

    with open(pickle_filename, 'wb') as f:
        pickle.dump((interp_vx, interp_vy, interp_vz), f)
    print(f"Interpolations sauvegardées dans {pickle_filename}.")
