import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres du vortex
h = 1.5
a = 2 # longueur caract axiale
b = 3 # longueur caract radiale
phi = np.pi/6

# Paramètre ζ
zeta = np.linspace(-np.pi, np.pi, 500)

'''# Équations paramétriques
x = a * np.cos(zeta) + a
y = h * np.exp(-zeta**2 / 2)

# Définition de z selon les cas
z = np.zeros_like(zeta)

# Cas 1 : ζ ∈ [−π, −π/4)
mask1 = zeta < -np.pi/4
z[mask1] = b * (zeta[mask1] + np.pi/4) * np.tan(phi) - np.sqrt(2)/2 * b

# Cas 2 : ζ ∈ [−π/4, π/4]
mask2 = (zeta >= -np.pi/4) & (zeta <= np.pi/4)
z[mask2] = b * np.sin(zeta[mask2])

# Cas 3 : ζ ∈ (π/4, π]
mask3 = zeta > np.pi/4
z[mask3] = b * (zeta[mask3] - np.pi/4) * np.tan(phi) + np.sqrt(2)/2 * b'''

# Équations paramétriques
x = a * np.cos(zeta) + a
y = h * np.exp(-zeta**2 / 2)

# Définition de z selon les cas
z = np.zeros_like(zeta)

# Cas 1 : ζ ∈ [−π, −π/4)
mask1 = zeta < -np.pi/4
z[mask1] = b * (zeta[mask1] + np.pi/4) * np.tan(phi) - np.sqrt(2)/2 * b

# Cas 2 : ζ ∈ [−π/4, π/4]
mask2 = (zeta >= -np.pi/4) & (zeta <= np.pi/4)
z[mask2] = b * np.sin(zeta[mask2])

# Cas 3 : ζ ∈ (π/4, π]
mask3 = zeta > np.pi/4
z[mask3] = b * (zeta[mask3] - np.pi/4) * np.tan(phi) + np.sqrt(2)/2 * b

# Affichage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Hairpin vortex centerline')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Hairpin Vortex Centerline")
ax.legend()
plt.tight_layout()
plt.show()
