import numpy as np
import pyvista as pv
import time

# Classe SaveParaview
class SaveParaview:
    def __init__(self, Nx, Ny, Nz, Lx, Ly, Lz):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        dx = 2 * Lx / (Nx - 1)
        dy = 2 * Ly / (Ny - 1)
        dz = Lz / (Nz - 1)

        self.grid = pv.ImageData()
        self.grid.dimensions = (Nx, Ny, Nz)
        self.grid.spacing = (dx, dy, dz)
        self.grid.origin = (-Lx, -Ly, 0)

    def add_scalar_field(self, data, name):
        if data.size != self.Nx * self.Ny * self.Nz:
            raise ValueError(f"Le champ '{name}' n'a pas la bonne taille.")
        self.grid[name] = data.flatten(order='F')

    def add_vector_field(self, data, name):
        if data.shape != (self.Nx, self.Ny, self.Nz, 3):
            raise ValueError(f"Le champ vectoriel '{name}' doit être de taille (Nx, Ny, Nz, 3).")
        flat_data = data.reshape(-1, 3, order='F')
        self.grid[name] = flat_data

    def save(self, filename):
        self.grid.save(filename)



def lap(champs_scal, dx, dy, dz):
    """
    Calcule le laplacien d’un champ scalaire 3D sur une grille régulière.

    champs_scal : np.ndarray de forme (Nx, Ny, Nz)
    dx, dy, dz : pas selon x, y, z

    Retour : np.ndarray de forme (Nx, Ny, Nz) — champ scalaire
    """
    d2f_dx2 = np.gradient(np.gradient(champs_scal, dx, axis=0), dx, axis=0)
    d2f_dy2 = np.gradient(np.gradient(champs_scal, dy, axis=1), dy, axis=1)
    d2f_dz2 = np.gradient(np.gradient(champs_scal, dz, axis=2), dz, axis=2)
    return d2f_dx2 + d2f_dy2 + d2f_dz2



def div(champs_vect, dx, dy, dz):
    """
    Calcule la divergence d’un champ vectoriel 3D sur une grille régulière.

    champs_vect : np.ndarray de forme (Nx, Ny, Nz, 3)
    dx, dy, dz : pas selon x, y, z

    Retour : np.ndarray de forme (Nx, Ny, Nz) — champ scalaire
    """
    Fx = champs_vect[..., 0]
    Fy = champs_vect[..., 1]
    Fz = champs_vect[..., 2]

    dFx_dx = np.gradient(Fx, dx, axis=0)
    dFy_dy = np.gradient(Fy, dy, axis=1)
    dFz_dz = np.gradient(Fz, dz, axis=2)

    return dFx_dx + dFy_dy + dFz_dz

def grad(champs_scal, dx, dy, dz):
    """
    Calcule le gradient d’un champ scalaire 3D sur une grille régulière.

    champs_scal : np.ndarray de forme (Nx, Ny, Nz)
    dx, dy, dz : pas selon x, y, z

    Retour : np.ndarray de forme (Nx, Ny, Nz, 3) — champ vectoriel
    """
    dS_dx = np.gradient(champs_scal, dx, axis=0)
    dS_dy = np.gradient(champs_scal, dy, axis=1)
    dS_dz = np.gradient(champs_scal, dz, axis=2)

    return np.stack((dS_dx, dS_dy, dS_dz), axis=-1)



def rot(champs_vect, dx, dy, dz):
    """
    Calcule le rotationnel d’un champ vectoriel 3D sur une grille régulière.

    champs_vect : np.ndarray de forme (Nx, Ny, Nz, 3)
        Le champ vectoriel (composantes x, y, z en dernière dimension)
    dx, dy, dz : float
        Pas d’espace selon x, y, z

    Retour :
        np.ndarray de forme (Nx, Ny, Nz, 3) — le rotationnel
    """
    Fx = champs_vect[..., 0]
    Fy = champs_vect[..., 1]
    Fz = champs_vect[..., 2]

    # Dérivées partielles centrées
    dFz_dy = np.gradient(Fz, dy, axis=1)
    dFy_dz = np.gradient(Fy, dz, axis=2)

    dFx_dz = np.gradient(Fx, dz, axis=2)
    dFz_dx = np.gradient(Fz, dx, axis=0)

    dFy_dx = np.gradient(Fy, dx, axis=0)
    dFx_dy = np.gradient(Fx, dy, axis=1)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    rot_field = np.stack((rot_x, rot_y, rot_z), axis=-1)
    return rot_field



# Hairpin vortex curve
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

# Champ de vitesse Lamb-Oseen par morceaux le long d'une ligne

def lamb_oseen_line_velocity_field(X, Y, Z, line_pts, Gamma=1.0, rc=0.1):
    """
    Calcule le champ de vitesse induit par une ligne de vortex,
    modélisée comme une suite de vortex de Lamb-Oseen 2D (plans orthogonaux à la ligne),
    chacun infini dans son propre plan.

    Paramètres :
    - X, Y, Z : grilles 3D (même forme)
    - line_pts : (N, 3) points de la ligne centrale du tube
    - Gamma : circulation de chaque élément
    - rc : rayon de cœur du vortex de Lamb-Oseen (2*np.sqrt(nu*t))

    Retour :
    - velocity : champ vectoriel 3D (shape X.shape + (3,))
    """
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

        t_hat = dp / L  # tangente unitaire à la ligne

        # vecteur position relatif
        rel_vec = points - p0

        # projection radiale perpendiculaire au plan du vortex (repère t_hat, radial_vec/r, azim_unit)
        s = np.dot(rel_vec, t_hat)  
        proj = np.outer(s, t_hat)  # composante dans la direction de t_hat
        radial_vec = rel_vec - proj  # vecteur radial dans le plan orthogonal
        r = np.linalg.norm(radial_vec, axis=1)
        r_safe = np.where(r == 0, 1e-16, r)

        # vitesse tangente (formule classique Lamb–Oseen)
        u_theta = (Gamma * L) / (2 * np.pi * r_safe) * (1 - np.exp(-r**2 / rc**2))

        # direction azimutale dans le plan orthogonal à t_hat
        azim_dir = np.cross(t_hat, radial_vec)
        azim_norm = np.linalg.norm(azim_dir, axis=1, keepdims=True)
        azim_unit = np.divide(azim_dir, azim_norm, out=np.zeros_like(azim_dir), where=azim_norm != 0)

        v_contrib = u_theta[:, None] * azim_unit
        velocity += v_contrib.reshape(shape + (3,))

    return velocity



start_time = time.time()


# Paramètres du domaine
Nx, Ny, Nz = 100, 100, 200
Lx, Ly, Lz = 8.0, 8.0, 8.0

N_ligne_vort = 50

x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
z = np.linspace(-Lz, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Courbe du vortex hairpin
zeta = np.linspace(-np.pi, np.pi, N_ligne_vort)

# Segment droit de longueur 1 dans la direction z
p0 = np.array([0.0, 0.0, 0.5])
p1 = np.array([0.0, 0.0, 1.5])

line_pts = np.array([p0, p1])  # shape (2, 3)

hairpin_pts = line_pts # hairpin_curve(zeta, h=2.0, phi=np.pi/6)

# Champ de vitesse
v_lamb = lamb_oseen_line_velocity_field(X, Y, Z, hairpin_pts, Gamma=5.0, rc=0.2)
vort_lamb = rot(v_lamb, Lx/Nx, Lx/Nx, Lz/Nz) # np.copy(v_lamb)
divergence = div(v_lamb, 2*Lx/Nx, 2*Lx/Nx, 2*Lz/Nz)

# Sauvegarde avec SaveParaview
saver = SaveParaview(Nx, Ny, Nz, Lx, Ly, Lz)
saver.add_vector_field(v_lamb, "velocity")
saver.add_vector_field(vort_lamb, "vort")
saver.add_scalar_field(divergence, "div")
saver.save(f"singerie_{N_ligne_vort}.vti")

end_time = time.time()
print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")
