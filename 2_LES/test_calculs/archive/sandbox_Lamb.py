import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

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

        # Création de la grille régulière
        self.grid = pv.ImageData()
        self.grid.dimensions = (Nx, Ny, Nz)
        self.grid.spacing = (dx, dy, dz)
        self.grid.origin = (-Lx, -Ly, 0)

    def add_scalar_field(self, data, name):
        if data.size != self.Nx * self.Ny * self.Nz:
            raise ValueError(f"Le champ '{name}' n'a pas la bonne taille.")
        self.grid[name] = data.flatten(order='F')  # ordre Fortran pour correspondance VTK

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


def lamb_oseen_velocity_field(X, Y, Z, Gamma=1.0, rc=0.1, x0=0.0, y0=0.0):
    """
    Calcule le champ de vitesse 3D d'un tourbillon de Lamb-Oseen d'axe z.
    
    Entrées:
      X, Y, Z : grilles 3D des coordonnées (mêmes dimensions)
      Gamma : circulation (intensité du tourbillon)
      rc : rayon caractéristique du vortex
      x0, y0 : centre du vortex dans le plan xy
    
    Sortie:
      v : champ vectoriel 4D (Nx, Ny, Nz, 3) avec composantes (vx, vy, vz)
    """
    # Coordonnées relatives au centre
    x_rel = X - x0
    y_rel = Y - y0
    
    # Distance radiale au centre
    r = np.sqrt(x_rel**2 + y_rel**2)
    
    # Pour éviter division par zéro au centre
    r_safe = np.where(r == 0, 1e-16, r)
    
    # Vitesse azimutale u_theta(r)
    u_theta = (Gamma / (2 * np.pi * r_safe)) * (1 - np.exp(-r**2 / rc**2))
    
    # Composantes du vecteur vitesse (champ circulaire)
    vx = -u_theta * y_rel / r_safe
    vy = u_theta * x_rel / r_safe
    vz = np.zeros_like(vx)
    
    # Champ vectoriel 4D
    v = np.stack((vx, vy, vz), axis=-1)
    
    return v




# Initialisation

# Dimensions du domaine
Nx = 100
Ny = Nx
Nz = 200

Lx = 1
Ly = Lx
Lz = 4

# Axes
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
z = np.linspace(0, Lz, Nz)

# Grille 3D (indexing='ij' pour correspondre aux dimensions (Nx, Ny, Nz))
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

saver = SaveParaview(Nx, Ny, Nz, Lx, Ly, Lz)

'''# Champ scalaire constant
champ_constant = np.ones((100, 100, 200))
saver.add_scalar_field(champ_constant, "champ_constant")

# Champ scalaire variable (ex : variation linéaire en z)
z = np.linspace(0, 4, 200)
Z = np.ones((100, 100, 200)) * z[np.newaxis, np.newaxis, :]
saver.add_scalar_field(Z, "champ_lineaire_z")

# Champ vectoriel (ex : vent constant vers z)
vecteur = np.zeros((100, 100, 200, 3))
vecteur[..., 2] = 1.0  # direction z
saver.add_vector_field(vecteur, "vecteur_z")'''



R = 0.5  # rayon du cylindre

# Champ vectoriel nul dans cylindre
v = np.zeros((Nx, Ny, Nz, 3))
mask_cylindre = (X**2 + Y**2 <= 2*Lx**2) & (Z >= 1) & (Z <= 2)
v[...][mask_cylindre] = 1.0
saver.add_vector_field(v, "vecteur_cylindre")

# Champ de vitesse Lamb-Oseen
v_out = lamb_oseen_velocity_field(X, Y, Z, Gamma=5.0, rc=0.3, x0=0.0, y0=0.0)
v_out_tronque = v_out * v
saver.add_vector_field(v_out, "velocity_lamb_oseen")

vort_lamb = rot(v_out, Lx/Nx, Lx/Nx, Lz/Nz)
saver.add_vector_field(vort_lamb, "vort_lamb_oseen")

vort_lamb_tronque = rot(v_out_tronque, Lx/Nx, Lx/Nx, Lz/Nz)
saver.add_vector_field(vort_lamb_tronque, "vort_lamb_oseen_tronque")

vort_comp = vort_lamb - vort_lamb_tronque
saver.add_vector_field(vort_comp, "comparaison_vort")

# Sauvegarde
saver.save("champ_global.vti")

