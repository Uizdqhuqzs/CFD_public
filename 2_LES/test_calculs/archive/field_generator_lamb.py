import time

start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm


def sample_N():
    values = np.arange(2, 11)
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #np.array([0.15, 0.14, 0.16, 0.18, 0.19, 0.13, 0.05, 0.0, 0.0])  # somme = 1
    return np.random.choice(values, p=weights)

# --- Nombre de pétales (Glauser) ---

'''def sample_N():

    poids_2=0.05
    poids_11=0.05
    mu=6
    sigma=2

    # Ensemble des entiers possibles
    values = np.arange(2, 12)
    
    # Loi gaussienne discrétisée
    probs = norm.pdf(values, mu, sigma)
    
    # Ajout du surplus
    probs[values == 2] += poids_2
    probs[values == 11] += poids_11
    
    # Normalisation
    probs /= np.sum(probs)

    # Affichage de la loi de probabilité
    plt.bar(values, probs, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Loi de probabilité discrète avec surplus à 2 et 11")
    plt.xlabel("Valeurs possibles")
    plt.ylabel("Probabilité")
    plt.show()
    
    # Tirage aléatoire selon cette loi discrète
    samples = np.random.choice(values, size=1, p=probs)
    
    return int(samples)'''


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

# --- Opérateurs différentiels ---
def lap(f, dx, dy, dz):
    return (
        np.gradient(np.gradient(f, dx, axis=0), dx, axis=0) +
        np.gradient(np.gradient(f, dy, axis=1), dy, axis=1) +
        np.gradient(np.gradient(f, dz, axis=2), dz, axis=2)
    )

def div(F, dx, dy, dz):
    return (
        np.gradient(F[...,0], dx, axis=0) +
        np.gradient(F[...,1], dy, axis=1) +
        np.gradient(F[...,2], dz, axis=2)
    )

def grad(f, dx, dy, dz):
    return np.stack((
        np.gradient(f, dx, axis=0),
        np.gradient(f, dy, axis=1),
        np.gradient(f, dz, axis=2)
    ), axis=-1)

def curl(F, dx, dy, dz):
    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
    return np.stack((
        np.gradient(Fz, dy, axis=1) - np.gradient(Fy, dz, axis=2),
        np.gradient(Fx, dz, axis=2) - np.gradient(Fz, dx, axis=0),
        np.gradient(Fy, dx, axis=0) - np.gradient(Fx, dy, axis=1)
    ), axis=-1)

# --- Sauvegarde Paraview ---
class SaveParaview:
    def __init__(self, Nx, Ny, Nz, Lx, Ly, Lz):
        dx = (2 * Lx) / (Nx - 1)
        dy = (2 * Ly) / (Ny - 1)
        dz = (Lz) / (Nz - 1)
        self.grid = pv.ImageData(dimensions=(Nx, Ny, Nz))
        self.grid.spacing = (dx, dy, dz)
        self.grid.origin = (-Lx, -Ly, 0)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

    def add_scalar_field(self, data, name):
        self.grid[name] = data.flatten(order='F')
    def add_vector_field(self, data, name):
        self.grid[name] = data.reshape(-1, 3, order='F')
    def save(self, filename):
        self.grid.save(filename)

# ======================================== MAIN ========================================

# Mesh
Nx, Ny, Nz = 150, 150, 500
Lx, Ly, Lz = 15, 15, 100
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

# Paramètres de base
h = 1.5
rc = 0.1
Gamma = 1 # à calibrer
st = 0.15  # Michalke
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

pickle_filename = "hairpin_rc_0p1_general_finer.pkl"

# Interpolateurs

if os.path.exists(pickle_filename):
    print(f"Chargement des interpolations depuis {pickle_filename}...")
    with open(pickle_filename, 'rb') as f:
        interp_vx, interp_vy, interp_vz = pickle.load(f)
else:
    print(f"Aucun fichier {pickle_filename} trouvé. Calcul des interpolations...")
    # Calcul du champ de base (unique)
    v_base = lamb_oseen_line_velocity_field(X, Y, Z, base_curve, 1.0, rc)

    interp_vx = RegularGridInterpolator((x, y, z), v_base[..., 0], bounds_error=False, fill_value=0)
    interp_vy = RegularGridInterpolator((x, y, z), v_base[..., 1], bounds_error=False, fill_value=0)
    interp_vz = RegularGridInterpolator((x, y, z), v_base[..., 2], bounds_error=False, fill_value=0)

    with open(pickle_filename, 'wb') as f:
        pickle.dump((interp_vx, interp_vy, interp_vz), f)
    print(f"Interpolations sauvegardées dans {pickle_filename}.")

# Champ total initialisé

dom = np.stack((X, Y, Z), axis=-1)
v_total = np.zeros_like(dom)

# Générer les couronnes
z_centers = np.linspace(0, Lz, nb_couronnes)

for z_center in z_centers:
    N = sample_N()
    phi = np.pi / 6
    R = compute_radial_spacing(b, N, phi)
    print(f"Couronne à z = {z_center:.2f} avec N = {N} (R = {R:.2f})")

    # Rotation aléatoire éventuelle
    theta_offset = np.random.uniform(0, 2 * np.pi)

    for k in range(N):
        theta = 2 * np.pi * k / N + theta_offset
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        # 1. Rotation des points autour de l'origine
        points_rot = points @ rot

        # 2. Translation dans le repère global

        sigma = 0.527  # 0.26 donné par Nickels
        shift_x = np.random.normal(0, sigma)
        shift_y = np.random.normal(0, sigma)
        shift_z = np.random.normal(0, sigma)

        points_rot[:, 0] -= R + shift_x   # placement radial
        points_rot[:, 1] -= shift_y
        points_rot[:, 2] -= z_center + shift_z  # placement axial

        # 3. Interpolation du champ de vitesse dans le repère tourné et translaté
        v_x = interp_vx(points_rot)
        v_y = interp_vy(points_rot)
        v_z = interp_vz(points_rot)
        v_interp = np.stack((v_x, v_y, v_z), axis=-1)

        # 4. Retour à l'orientation globale (rotation inverse)
        v_rotated = v_interp @ rot.T

        v_rotated *= Gamma/N

        # 5. Ajout au champ total
        v_total += v_rotated.reshape(X.shape + (3,))


print("Calcul des champs de vorticité et de divergence\n")

# Post
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
vort = curl(v_total, dx, dy, dz)
div_field = div(v_total, dx, dy, dz)

print("Sauvegarde en .vti\n")

# Sauvegarde Paraview
saver = SaveParaview(Nx, Ny, Nz, Lx, Ly, Lz)
saver.add_vector_field(v_total, "velocity")
saver.add_vector_field(vort, "vorticity")
saver.add_scalar_field(div_field, "divergence")
saver.save("grfhrh.vti")

print(f"Terminé en {time.time() - start_time:.2f} s")