import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres du vortex
h = 1.0
a = h / 2
b = h / 2
phi = np.pi / 6

# Paramètre ζ
zeta = np.linspace(-np.pi, np.pi, 500)

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


def compute_radial_spacing(h, N, phi):
    b = h / 2
    lateral_spacing = b * (1.5 * np.pi * np.tan(phi) + np.sqrt(2))
    R = b / (np.tan(np.pi/N)) #(lateral_spacing * N) / (2 * np.pi)
    return R



def generate_hairpin_ring(N=6, h=1.0, phi=np.pi/6, R=None):
    zeta = np.linspace(-np.pi, np.pi, 300)
    base_curve = hairpin_curve(zeta, h=h, phi=phi)

    # Si R n’est pas donné, le calculer automatiquement
    if R is None:
        R = compute_radial_spacing(h, N, phi)


    curves = []
    for k in range(N):
        theta = 2 * np.pi * k / N  # Angle de rotation autour de x
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

        # Translation radiale vers l’extérieur
        radial_shift = np.array([0, R, 0])
        translated_curve = base_curve + radial_shift

        # Appliquer la rotation autour de x
        rotated_curve = translated_curve @ rotation_matrix.T
        curves.append(rotated_curve)

    return curves

def plot_hairpin_ring(curves):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Tracer les courbes
    for curve in curves:
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])

    # Empiler toutes les courbes pour trouver les min/max globaux
    all_points = np.concatenate(curves, axis=0)
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    # Centre et demi-longueur max
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    # Appliquer les mêmes bornes centrées
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    ax.set_box_aspect([1, 1, 1])  # même aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hairpin Vortex Ring Structure')
    plt.tight_layout()
    plt.show()


def compute_phi(N):
    return np.arctan(np.pi / (np.sqrt(2) * N))

def generate_parametrized_ring(N=6, h=1.0, phi=np.pi/6, R=None, total_points=2000):
    """
    Génère une seule courbe 3D continue représentant la couronne complète.
    """
    if R is None:
        R = compute_radial_spacing(h, N, phi)

    # Nombre de points par hairpin
    points_per_hairpin = total_points // N
    zeta = np.linspace(-np.pi, np.pi, points_per_hairpin)
    base_curve = hairpin_curve(zeta, h=h, phi=phi)

    full_curve = []

    for k in range(N):
        theta = 2 * np.pi * k / N

        # Rotation autour de x
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

        # Translation radiale
        radial_shift = np.array([0, R, 0])
        transformed_curve = base_curve + radial_shift
        rotated_curve = transformed_curve @ rotation_matrix.T

        full_curve.append(rotated_curve)

    # Fusionner en une seule courbe
    return np.concatenate(full_curve, axis=0)

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
    ax.set_title("Courbe paramétrée unique – Hairpin Ring")
    plt.tight_layout()
    plt.show()




N = 20
h = 5.0
phi = compute_phi(N)
R = compute_radial_spacing(h, N, phi)

curve = generate_parametrized_ring(N=N, h=h, phi=phi, R=R)
plot_parametrized_curve(curve)

