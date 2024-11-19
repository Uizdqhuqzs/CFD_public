# Initialisation de la figure pour visualisation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from scipy.interpolate import interp1d
import vtk
from vtk.util import numpy_support
import numpy as np
import os

# Définir OMP_NUM_THREADS à la valeur souhaitée (par exemple, 4 threads)
os.environ["OMP_NUM_THREADS"] = "8"

# Fichier paraview
def save_vti(filename, u, v, x, y):
    # Création d'un objet vtkImageData
    grid = vtk.vtkImageData()
    
    # Définir la dimension de la grille en fonction des données
    ny, nx = u.shape
    grid.SetDimensions(nx, ny, 1)
    
    # Calcul des espacements à partir des coordonnées x et y
    dx = x[1] - x[0]  # Supposant un maillage uniforme
    dy = y[1] - y[0]
    grid.SetSpacing(dx, dy, 1)
    
    # Définir l'origine
    grid.SetOrigin(x[0], y[0], 0)
    
    # Conversion des tableaux numpy u et v en vtkArray
    u_flat = u.ravel()
    v_flat = v.ravel()
    w_flat = np.zeros_like(u_flat)  # Composante nulle pour le champ en 2D
    
    # Combiner les composantes u, v et w en un seul champ vectoriel
    velocity_vectors = np.column_stack((u_flat, v_flat, w_flat))
    velocity_vtk = numpy_support.numpy_to_vtk(velocity_vectors, deep=True, array_type=vtk.VTK_FLOAT)
    velocity_vtk.SetName("velocity")
    
    # Ajouter le champ vectoriel au point data de la grille
    grid.GetPointData().SetVectors(velocity_vtk)
    
    # Ecrire dans un fichier VTI
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

# Créer un masque pour l'obstacle NACA 0012

def naca0012(x, t):
    """
    Génère les coordonnées y des profils supérieur et inférieur NACA 0012.
    :param x: Coordonnées le long de la corde.
    :param c: Longueur de la corde.
    :param t: Épaisseur du profil.
    :return: Coordonnées y des profils supérieur et inférieur.
    """
    # Calcul des ordonnées pour le profil NACA 0012
    y = (t / 0.2) * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    return y

def obstacle_naca0012(x, y, Nx, Ny, X_pos, Y_pos, c, t):
    """
    Crée un masque pour un obstacle NACA 0012 centré à (X_pos, Y_pos).
    :param x: Grille des positions x.
    :param y: Grille des positions y.
    :param X_pos: Position x du centre de l'obstacle.
    :param Y_pos: Position y du centre de l'obstacle.
    :param c: Longueur de la corde du profil.
    :param t: Épaisseur du profil.
    :param threshold: Seuil de distance pour considérer un point à l'intérieur de l'obstacle.
    :return: Masque représentant l'obstacle.
    """
    # Discrétisation du profil le long de x
    n_points = 1000
    x_profil = np.linspace(0, 1, n_points)
    y_profil = naca0012(x_profil, t)
    
    # Mise à l'échelle et décalage pour centrer le profil à (X_pos, Y_pos)
    x_profil = X_pos + c * x_profil
    y_profil = Y_pos + y_profil

    # Interpolation des profils supérieur et inférieur
    y_interp = interp1d(x_profil, y_profil, kind='linear', fill_value="extrapolate")

    # Création d'un masque pour représenter l'obstacle
    mask = np.zeros((Ny, Nx))

    # Boucle sur chaque point du maillage de simulation
    for i in range(Nx):
        for j in range(Ny):
            if X_pos <= x[j, i] <= X_pos + c:  # Si x[i] est dans la plage de la corde
                
                # Si le point est proche de l'un des profils
                if np.abs(y[j, i]) < y_interp(x[j, i]):
                    mask[j, i] = 1

    """# Plot the cylindrical obstacle
    plt.contourf(x, y, mask)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Cylindrical Obstacle")
    plt.show()"""


    return mask



# Initialisation de la figure pour visualiser les champs de vitesse, norme, et pression
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_vorticity, ax_vorticity = plt.subplots(figsize=(6, 5))  # Fenêtre séparée pour la vorticité
plt.ion()  
Contour_number = 50

"---------------------------  physical and numerical parameters  --------------------------"
Re = 3000
T, dt = 50, 1e-4
Nx, X1, X2 = 300, -0.1, 10
Ny, Y1, Y2 = 100, -0.2, 0.2
Suivre = False

"------------------------- physical and spectral discretizations  -------------------------"
Nt = round(T/dt)+1
x  = np.linspace(0, Nx-1, Nx) / Nx * 2 * np.pi
kx = np.fft.fftfreq(Nx) * Nx
y  = np.linspace(0, Ny-1, Ny) / Ny * 2 * np.pi
ky = np.fft.fftfreq(Ny) * Ny

# mapping [0, 2 pi]x[0, 2 pi] -> [X1, X2]x[Y1, Y2]
x, kx = (X2 - X1) / (2 * np.pi) * x + X1, 2 * np.pi / (X2 - X1) * kx
y, ky = (Y2 - Y1) / (2 * np.pi) * y + Y1, 2 * np.pi / (Y2 - Y1) * ky
# grids
x, y = np.meshgrid(x, y)
kx, ky = np.meshgrid(kx, ky)

k2 = kx**2 + ky**2
dealias = np.zeros((Ny, Nx))
kmax = np.max(np.sqrt(k2))
dealias[np.sqrt(k2) < 2 * kmax / 3] = 1

"------------------------------------------ Conditions initiales ------------------------------------------------"
#CI Green-Taylor
"""u = np.sin(x) * np.cos(y)  # Condition initiale
v = -np.sin(y) * np.cos(x)
p = -0.25 * (np.cos(2 * x) + np.cos(2 * y))"""

#CI Vortex Merging

"""r1_2 = (x-0.5)**2 + y**2
r2_2 = (x+0.5)**2 + y**2
m_r1 = np.max(np.sqrt(r1_2))
m_r2 = np.max(np.sqrt(r2_2))

print(f"r1 = {m_r1:.5f}, r2 = {m_r2:.5f}")
u = (y/np.sqrt(r1_2))*np.exp(-r1_2) + (y/np.sqrt(r2_2))*np.exp(-r2_2)
v = -((x-0.5)/np.sqrt(r1_2))*np.exp(-r1_2) - ((x+0.5)/np.sqrt(r2_2))*np.exp(-r2_2)
p = u*0

m_u = np.max(np.abs(u))
m_v = np.max(np.abs(v))
max_u = np.max(np.sqrt(u**2 + v**2))"""
#print(f"vitesse initiale : u_max = {m_u:.5f}, v_max = {m_v:.5f}, norme_max = {max_u:.5f}\n\n")

# Ecoulement Laminaire

u = np.ones((Ny, Nx)) * 2
v = np.zeros((Ny, Nx)) + (np.random.rand(Ny, Nx) - 0.5)
p = np.zeros((Ny, Nx))

"------------------------------------------ Forces Extérieures ------------------------------------------------"
# Obstacle Cylindre
"""masque = np.zeros((Ny, Nx))
masque[np.sqrt(x**2 + y**2) <= 0.5] = 1
u_star = 0 * u
ampl_obstacle = 4"""

# NACA 0012
c = 1
t = 0.12/3
masque = obstacle_naca0012(x, y, Nx, Ny, 0, 0, c, t)
u_star = 0 * u
ampl_obstacle = 20


# Buffer
ampl_buffer = 20
buffer = ampl_buffer*np.exp(-(x - 0.9 * (X2 - X1))**2 / 4)


"------------------------------------------ Solver ------------------------------------------------"
#espace spectral
pf = np.fft.fftn(p)
uf = np.fft.fftn(u)
vf = np.fft.fftn(v)
K2P = (kx**2 + ky**2)
K2P[0, 0] = 1
energy = []
divergence = []
timeScale = []

# Calcul des espacements Δx et Δy
dx = (X2 - X1) / Nx
dy = (Y2 - Y1) / Ny

# Calcul des limites fixes pour les échelles de couleur
u_min, u_max = -1.5, 1.1 #np.min(u), np.max(u)
v_min, v_max = -1.5, 1.1 #np.min(v), np.max(v)
p_min, p_max = np.min(p), np.max(p)

# itérations temporelle
for n in tqdm(range(Nt), desc="Calcul de la simulation", unit="étape"):

    # Calcul du nombre de Courant
    max_u = np.max(np.sqrt(u**2 + v**2))  # Norme maximale de la vitesse
    Cx = max_u * dt / dx
    Cy = max_u * dt / dy
    C = max(Cx, Cy)  # Nombre de Courant maximal
    if (n%(Nt//10) == 0):
        print(f"Étape {n}/{Nt}")
        print("Nombre de courant max : ", C)
        print(f"max_u ={max_u:.5f}, dt = {dt:.5f}, dx = {dx:.5f}, dy = {dy:.5f}")

    # Interruption si C > 1
    if C > 1:
        print(f"Nombre de Courant dépassé : C = {C:.2f}. Simulation arrêtée.")
        break

    dux = np.fft.ifftn(1j * kx * uf)
    dvx = np.fft.ifftn(1j * kx * vf)
    duy = np.fft.ifftn(1j * ky * uf)
    dvy = np.fft.ifftn(1j * ky * vf)

    obstacle_u = ampl_obstacle*masque*(u_star-u)
    obstacle_v = ampl_obstacle*masque*(u_star-v)

    # Prédiction
    Tnlu = np.fft.fftn(u * dux + v * duy) * dealias
    Tnlv = np.fft.fftn(u * dvx + v * dvy) * dealias

    # Projection
    # Update spectral u and v with diffusion and buffer terms
    uft = (uf / dt + np.fft.fftn(obstacle_u) - np.fft.fftn(buffer * (u - 1)) - 1j * kx * pf - Tnlu) / (1 / dt + K2P / Re)
    vft = (vf / dt + np.fft.fftn(obstacle_v) - np.fft.fftn(buffer * (v))     - 1j * ky * pf - Tnlv) / (1 / dt + K2P / Re)
    ppf = pf - 1j*(kx * uft + ky * vft) / (dt * K2P)

    uf = uft - dt*1j*kx*(ppf-pf)
    vf = vft - dt*1j*ky*(ppf-pf)

    pf = ppf

    #retour espace physique
    u = np.real(np.fft.ifftn(uf))
    v = np.real(np.fft.ifftn(vf))
    p = np.real(np.fft.ifftn(pf))

    # Énergie cinétique
    energy.append(np.mean(u**2 + v**2) / 2)
    timeScale.append(n * dt)

    #divergence
    div = np.fft.ifftn(1j * kx * uf + 1j * ky * vf)
    divergence.append(np.max(np.abs(div)))

    if (n % 1000 == 0):
        save_vti(f"temps{1000*n * dt:.0f}.vti",u, v, x[0], y[:,0])

# Visualisation tous les 10 pas de temps
if (n % 10 == 0) and Suivre:
    for ax in axes:
         ax.cla()  # Effacement de l'axe pour réinitialiser le graphique
    ax_vorticity.cla()  # Effacement de la fenêtre de vorticité

    # Norme de u
    norme = np.sqrt(u**2 + v**2)
    c_norme = axes[0].contourf(x, y, norme, levels=Contour_number, cmap="RdBu")
    if n == 0:
            fig.colorbar(c_norme, ax=axes[0])
    axes[0].set_title("Norme $u$")

    # Champ de vitesse u
    c_u = axes[1].contourf(x, y, u, levels=Contour_number, cmap="RdBu", vmin=u_min, vmax=u_max, extend='both')
    if n == 0:
            fig.colorbar(c_u, ax=axes[1])
    axes[1].set_title("Champ de vitesse $u$")

    # Champ de vitesse v
    c_v = axes[2].contourf(x, y, v, levels=Contour_number, cmap="RdBu", vmin=v_min, vmax=v_max, extend='both')
    if n == 0:
        fig.colorbar(c_v, ax=axes[2])
    axes[2].set_title("Champ de vitesse $v$")

    # Vorticité de u (affichage dans la fenêtre séparée)
    vorticite = np.real(np.fft.ifftn(1j*kx*vf - 1j*ky*uf))
    c_vorticite = ax_vorticity.contourf(x, y, vorticite, levels=Contour_number, cmap="RdBu")
    if n == 0:
        fig_vorticity.colorbar(c_vorticite, ax=ax_vorticity)
    ax_vorticity.set_title("Vorticité $u$")

    # Mettre à jour les titres et affichages
    fig.suptitle(f"Temps : {n * dt:.3f} s")
    fig_vorticity.suptitle(f"Temps : {n * dt:.3f} s (Vorticité)")
    plt.pause(1e-20)

"------------------------------------------ Affichage final ------------------------------------------------"
# Affichage final si Suivre = False
if not Suivre:
    for ax in axes:
        ax.cla()  # Effacement de l'axe pour réinitialiser le graphique
    ax_vorticity.cla()  # Effacement de la fenêtre de vorticité

    # Norme de u (échelle dynamique)
    norme = np.sqrt(u**2 + v**2)
    c_norme = axes[0].contourf(x, y, norme, levels=Contour_number, cmap="RdBu")
    fig.colorbar(c_norme, ax=axes[0])
    axes[0].set_title("Norme $u$")
    axes[0].contour(x, y, masque, levels=[0.5], colors="black")  # Masque de l'obstacle

    # Champ de vitesse u (échelle fixe)
    c_u = axes[1].contourf(x, y, u, levels=Contour_number, cmap="RdBu", vmin=u_min, vmax=u_max, extend='both')
    fig.colorbar(c_u, ax=axes[1])
    axes[1].set_title("Champ de vitesse $u$")
    axes[1].contour(x, y, masque, levels=[0.5], colors="black")  # Masque de l'obstacle

    # Champ de vitesse v (échelle fixe)
    c_v = axes[2].contourf(x, y, v, levels=Contour_number, cmap="RdBu", vmin=v_min, vmax=v_max, extend='both')
    fig.colorbar(c_v, ax=axes[2])
    axes[2].set_title("Champ de vitesse $v$")
    axes[2].contour(x, y, masque, levels=[0.5], colors="black")  # Masque de l'obstacle

    # Vorticité de u (échelle dynamique)
    vorticite = np.real(np.fft.ifftn(1j*kx*vf - 1j*ky*uf))
    c_vorticite = ax_vorticity.contourf(x, y, vorticite, levels=Contour_number, cmap="RdBu", vmin=-60, vmax=40)
    fig_vorticity.colorbar(c_vorticite, ax=ax_vorticity)
    ax_vorticity.set_title("Vorticité $u$")
    ax_vorticity.contour(x, y, masque, levels=[0.5], colors="black")  # Masque de l'obstacle

    # Mise à jour des titres finaux
    fig.suptitle(f"Temps final : {n * dt:.3f} s")
    fig_vorticity.suptitle(f"Temps final : {n * dt:.3f} s (Vorticité)")
    plt.pause(1e-20)


    save_vti(f"temps_final_{1000*n * dt:.3f}s.vti",u, v, x[0], y[:,0])

plt.ioff()  # Désactiver le mode interactif à la fin
plt.show()


"------------------------------------------ Plot de controle ------------------------------------------------"
exact_energy = []

for i in range(len(timeScale)):
    exact_energy.append(energy[0]*np.exp(-2*timeScale[i]/Re))

# Tracer l'énergie cinétique en fonction du temps
fig_energy, ax_energy = plt.subplots(figsize=(8, 6))
ax_energy.plot(timeScale, energy, label="Énergie cinétique")
ax_energy.plot(timeScale, exact_energy, label="Énergie Exacte")
ax_energy.set_xlabel('Temps')
ax_energy.set_ylabel('Énergie cinétique')
ax_energy.set_title('Évolution de l\'énergie cinétique en fonction du temps')
ax_energy.legend()
plt.show()

#tracer la divergence
# Tracer l'énergie cinétique en fonction du temps
fig_divergence, ax_divergence = plt.subplots(figsize=(8, 6))
ax_divergence.plot(timeScale, divergence, label="divergence")
ax_divergence.set_xlabel('Temps')
ax_divergence.set_ylabel('Div u')
ax_divergence.set_title('Évolution de Div u')
ax_divergence.legend()
plt.show()

#print(energy,"\n\n",timeScale,"\n\n",exact_energy,"\n\n")