import time

start_time = time.time()

import numpy as np
import pyvista as pv


# --- Hairpin et Ring ---
def hairpin_curve(zeta, h=1.0, phi=np.pi/6):
    a = b = h / 2
    x = a * np.cos(zeta) + a
    y = h * np.exp(-zeta**2 / 2)
    z = np.zeros_like(zeta)
    mask1 = zeta < -np.pi/4
    mask2 = (zeta >= -np.pi/4) & (zeta <= np.pi/4)
    mask3 = zeta > np.pi/4
    z[mask1] = b * (zeta[mask1] + np.pi/4) * np.tan(phi) - np.sqrt(2)/2 * b
    z[mask2] = b * np.sin(zeta[mask2])
    z[mask3] = b * (zeta[mask3] - np.pi/4) * np.tan(phi) + np.sqrt(2)/2 * b
    return np.stack((x, y, z), axis=-1)

def compute_phi(N):
    return np.arctan(np.pi / (np.sqrt(2) * N))

def compute_radial_spacing(h, N, phi):
    b = h / 2
    lateral_spacing = b * (1.5 * np.pi * np.tan(phi) + np.sqrt(2))
    R = (lateral_spacing * N) / (2 * np.pi)
    return R

'''def generate_parametrized_ring(N=6, h=1.0, phi=np.pi/6, R=None, total_points=2000):
    if R is None:
        R = compute_radial_spacing(h, N, phi)
    points_per_hairpin = total_points // N
    zeta = np.linspace(-np.pi, np.pi, points_per_hairpin)
    base_curve = hairpin_curve(zeta, h=h, phi=phi)
    full_curve = []
    for k in range(N):
        theta = 2 * np.pi * k / N
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        radial_shift = np.array([R, 0, 0])
        translated = base_curve + radial_shift
        rotated = translated @ rot.T
        full_curve.append(rotated)
    return np.concatenate(full_curve, axis=0)'''

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
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        dz = Lz / (Nz - 1)
        self.grid = pv.ImageData(dimensions=(Nx, Ny, Nz))
        self.grid.spacing = (dx, dy, dz)
        self.grid.origin = (0, 0, 0)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
    def add_scalar_field(self, data, name):
        self.grid[name] = data.flatten(order='F')
    def add_vector_field(self, data, name):
        self.grid[name] = data.reshape(-1, 3, order='F')
    def save(self, filename):
        self.grid.save(filename)

# --- MAIN ---


# Mesh
Nx, Ny, Nz = 300, 300, 300
Lx, Ly, Lz = 20.0, 20.0, 20.0
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
z = np.linspace(-Lz, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Paramètres 
N = 10
h = 2.5
phi = compute_phi(N)
R = compute_radial_spacing(h, N, phi)

# Générer un seul hairpin
zeta = np.linspace(-np.pi, np.pi, 500)
base_curve = hairpin_curve(zeta, h=h, phi=phi)

# Calcul du champ de vitesse total par superposition
Gamma = 5.0
rc = 0.25
v_total = np.zeros(X.shape + (3,))
for k in range(N):
    theta = 2 * np.pi * k / N

    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    radial_shift = np.array([0, R, 0])# np.array([R, 0, 0])
    transformed = base_curve + radial_shift
    rotated = transformed @ rot.T
    v_k = lamb_oseen_line_velocity_field(X, Y, Z, rotated, Gamma=Gamma, rc=rc)
    v_total += v_k


# Post
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
vort = curl(v_total, dx, dy, dz)
div_field = div(v_total, dx, dy, dz)

# Sauvegarde Paraview
saver = SaveParaview(Nx, Ny, Nz, Lx, Ly, Lz)
saver.add_vector_field(v_total, "velocity")
saver.add_vector_field(vort, "vorticity")
saver.add_scalar_field(div_field, "divergence")
saver.save("hairpin_ring_field.vti")

print(f"Terminé en {time.time() - start_time:.2f} s")
