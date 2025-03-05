import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from scipy.interpolate import interp1d
import vtk
from vtk.util import numpy_support
import numpy as np
import os

# Set OMP_NUM_THREADS to the desired value (e.g., 4 threads)
os.environ["OMP_NUM_THREADS"] = "8"

# Paraview file
def save_vti(filename, u, v, x, y):
    grid = vtk.vtkImageData()
    ny, nx = u.shape
    grid.SetDimensions(nx, ny, 1)
    dx = x[1] - x[0] # Assuming uniform grid
    dy = y[1] - y[0]
    grid.SetSpacing(dx, dy, 1)
    grid.SetOrigin(x[0], y[0], 0)
    
    
    u_flat = u.ravel()
    v_flat = v.ravel()
    w_flat = np.zeros_like(u_flat)  # Zero component for 2D field
    
    velocity_vectors = np.column_stack((u_flat, v_flat, w_flat))
    velocity_vtk = numpy_support.numpy_to_vtk(velocity_vectors, deep=True, array_type=vtk.VTK_FLOAT)
    velocity_vtk.SetName("velocity")
    
    grid.GetPointData().SetVectors(velocity_vtk)
    
    # Write to a VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()


# Create a mask for the NACA 0012 obstacle

def naca0012(x, t):
    """
    Generates the y-coordinates of the upper and lower NACA 0012 profiles.
    :param x: Coordinates along the chord.
    :param c: Chord length.
    :param t: Profile thickness.
    :return: y-coordinates of upper and lower profiles.
    """
    y = (t / 0.2) * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    return y

def obstacle_naca0012(x, y, Nx, Ny, X_pos, Y_pos, c, t):
    """
    Creates a mask for a NACA 0012 obstacle centered at (X_pos, Y_pos).
    :param x: Grid of x positions.
    :param y: Grid of y positions.
    :param X_pos: x position of the obstacle center.
    :param Y_pos: y position of the obstacle center.
    :param c: Chord length of the profile.
    :param t: Profile thickness.
    :param threshold: Distance threshold to consider a point inside the obstacle.
    :return: Mask representing the obstacle.
    """
    # Discretization of the profile along x
    n_points = 1000
    x_profil = np.linspace(0, 1, n_points)
    y_profil = naca0012(x_profil, t)
    
    # Scaling and shifting to center the profile at (X_pos, Y_pos)
    x_profil = X_pos + c * x_profil
    y_profil = Y_pos + y_profil

    
    y_interp = interp1d(x_profil, y_profil, kind='linear', fill_value="extrapolate")

    
    mask = np.zeros((Ny, Nx))

    
    for i in range(Nx):
        for j in range(Ny):
            if X_pos <= x[j, i] <= X_pos + c:   # Check if x[i] is within the chord range
                
                # If the point is close to one of the profiles
                if np.abs(y[j, i]) < y_interp(x[j, i]):
                    mask[j, i] = 1

    return mask



fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_vorticity, ax_vorticity = plt.subplots(figsize=(6, 5)) 
Contour_number = 50

"---------------------------  Physical and Numerical Parameters  --------------------------"
Re = 3000
T, dt = 50, 1e-4
Nx, X1, X2 = 300, -0.1, 10
Ny, Y1, Y2 = 100, -0.2, 0.2
Suivre = False

"------------------------- Discretization of Physical and Spectral Domains  -------------------------"
Nt = round(T/dt)+1
x  = np.linspace(0, Nx-1, Nx) / Nx * 2 * np.pi
kx = np.fft.fftfreq(Nx) * Nx
y  = np.linspace(0, Ny-1, Ny) / Ny * 2 * np.pi
ky = np.fft.fftfreq(Ny) * Ny

# Mapping [0, 2 pi]x[0, 2 pi] -> [X1, X2]x[Y1, Y2]
x, kx = (X2 - X1) / (2 * np.pi) * x + X1, 2 * np.pi / (X2 - X1) * kx
y, ky = (Y2 - Y1) / (2 * np.pi) * y + Y1, 2 * np.pi / (Y2 - Y1) * ky
# Grids
x, y = np.meshgrid(x, y)
kx, ky = np.meshgrid(kx, ky)

k2 = kx**2 + ky**2
dealias = np.zeros((Ny, Nx))
kmax = np.max(np.sqrt(k2))
dealias[np.sqrt(k2) < 2 * kmax / 3] = 1

"------------------------------------------ Initial state ------------------------------------------------"

# CI Green-Taylor

"""u = np.sin(x) * np.cos(y)  # Condition initiale
v = -np.sin(y) * np.cos(x)
p = -0.25 * (np.cos(2 * x) + np.cos(2 * y))"""

# CI Vortex Merging

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

# Laminar flow

u = np.ones((Ny, Nx)) * 2
v = np.zeros((Ny, Nx)) + (np.random.rand(Ny, Nx) - 0.5)
p = np.zeros((Ny, Nx))

"------------------------------------------ Forces Extérieures ------------------------------------------------"
# Cylinder obstacle
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

# Forçage (W.I.P)


"------------------------------------------ Solver ------------------------------------------------"
# Spectral space
pf = np.fft.fftn(p)
uf = np.fft.fftn(u)
vf = np.fft.fftn(v)
K2P = (kx**2 + ky**2)
K2P[0, 0] = 1
energy = []
divergence = []
timeScale = []

dx = (X2 - X1) / Nx
dy = (Y2 - Y1) / Ny

# Color scale for plots
u_min, u_max = -1.5, 1.1 #np.min(u), np.max(u)
v_min, v_max = -1.5, 1.1 #np.min(v), np.max(v)
p_min, p_max = np.min(p), np.max(p)

# Time iteration
for n in tqdm(range(Nt), desc="Calcul de la simulation", unit="étape"):

    # Compute CFL number
    max_u = np.max(np.sqrt(u**2 + v**2))
    Cx = max_u * dt / dx
    Cy = max_u * dt / dy
    C = max(Cx, Cy)  # CFL max
    if (n%(Nt//10) == 0):
        print(f"Étape {n}/{Nt}")
        print("Nombre de courant max : ", C)
        print(f"max_u ={max_u:.5f}, dt = {dt:.5f}, dx = {dx:.5f}, dy = {dy:.5f}")

    # Interruption if C > 1
    if C > 1:
        print(f"Nombre de Courant dépassé : C = {C:.2f}. Simulation arrêtée.")
        break

    dux = np.fft.ifftn(1j * kx * uf)
    dvx = np.fft.ifftn(1j * kx * vf)
    duy = np.fft.ifftn(1j * ky * uf)
    dvy = np.fft.ifftn(1j * ky * vf)

    obstacle_u = ampl_obstacle*masque*(u_star-u)
    obstacle_v = ampl_obstacle*masque*(u_star-v)

    # Prediction (Temam-Chorin method)
    Tnlu = np.fft.fftn(u * dux + v * duy) * dealias
    Tnlv = np.fft.fftn(u * dvx + v * dvy) * dealias

    # Projection
    uft = (uf / dt + np.fft.fftn(obstacle_u) - np.fft.fftn(buffer * (u - 1)) - 1j * kx * pf - Tnlu) / (1 / dt + K2P / Re)
    vft = (vf / dt + np.fft.fftn(obstacle_v) - np.fft.fftn(buffer * (v))     - 1j * ky * pf - Tnlv) / (1 / dt + K2P / Re)
    ppf = pf - 1j*(kx * uft + ky * vft) / (dt * K2P)

    uf = uft - dt*1j*kx*(ppf-pf)
    vf = vft - dt*1j*ky*(ppf-pf)

    pf = ppf

    # Translation to physical space
    u = np.real(np.fft.ifftn(uf))
    v = np.real(np.fft.ifftn(vf))
    p = np.real(np.fft.ifftn(pf))

    # Kinetic energy
    energy.append(np.mean(u**2 + v**2) / 2)
    timeScale.append(n * dt)

    # Divergence
    div = np.fft.ifftn(1j * kx * uf + 1j * ky * vf)
    divergence.append(np.max(np.abs(div)))

    if (n % 1000 == 0):
        save_vti(f"temps{1000*n * dt:.0f}.vti",u, v, x[0], y[:,0])

# Visualisation every 10 time steps

if (n % 10 == 0) and Suivre:
    # Plot window cleaning
    for ax in axes:
         ax.cla()  
    ax_vorticity.cla()  

    # Norm u
    norme = np.sqrt(u**2 + v**2)
    c_norme = axes[0].contourf(x, y, norme, levels=Contour_number, cmap="RdBu")
    if n == 0:
            fig.colorbar(c_norme, ax=axes[0])
    axes[0].set_title("Norme $u$")

    # u field
    c_u = axes[1].contourf(x, y, u, levels=Contour_number, cmap="RdBu", vmin=u_min, vmax=u_max, extend='both')
    if n == 0:
            fig.colorbar(c_u, ax=axes[1])
    axes[1].set_title("Champ de vitesse $u$")

    # v field
    c_v = axes[2].contourf(x, y, v, levels=Contour_number, cmap="RdBu", vmin=v_min, vmax=v_max, extend='both')
    if n == 0:
        fig.colorbar(c_v, ax=axes[2])
    axes[2].set_title("Champ de vitesse $v$")

    # vorticity field
    vorticite = np.real(np.fft.ifftn(1j*kx*vf - 1j*ky*uf))
    c_vorticite = ax_vorticity.contourf(x, y, vorticite, levels=Contour_number, cmap="RdBu")
    if n == 0:
        fig_vorticity.colorbar(c_vorticite, ax=ax_vorticity)
    ax_vorticity.set_title("Vorticité $u$")

    # Plot update
    fig.suptitle(f"Temps : {n * dt:.3f} s")
    fig_vorticity.suptitle(f"Temps : {n * dt:.3f} s (Vorticité)")
    plt.pause(1e-20)

"------------------------------------------ Affichage final ------------------------------------------------"
# Final plot if Suivre = False
if not Suivre:
    for ax in axes:
        ax.cla() 
    ax_vorticity.cla()  


    norme = np.sqrt(u**2 + v**2)
    c_norme = axes[0].contourf(x, y, norme, levels=Contour_number, cmap="RdBu")
    fig.colorbar(c_norme, ax=axes[0])
    axes[0].set_title("Norme $u$")
    axes[0].contour(x, y, masque, levels=[0.5], colors="black")  


    c_u = axes[1].contourf(x, y, u, levels=Contour_number, cmap="RdBu", vmin=u_min, vmax=u_max, extend='both')
    fig.colorbar(c_u, ax=axes[1])
    axes[1].set_title("Champ de vitesse $u$")
    axes[1].contour(x, y, masque, levels=[0.5], colors="black")  


    c_v = axes[2].contourf(x, y, v, levels=Contour_number, cmap="RdBu", vmin=v_min, vmax=v_max, extend='both')
    fig.colorbar(c_v, ax=axes[2])
    axes[2].set_title("Champ de vitesse $v$")
    axes[2].contour(x, y, masque, levels=[0.5], colors="black")  


    vorticite = np.real(np.fft.ifftn(1j*kx*vf - 1j*ky*uf))
    c_vorticite = ax_vorticity.contourf(x, y, vorticite, levels=Contour_number, cmap="RdBu", vmin=-60, vmax=40)
    fig_vorticity.colorbar(c_vorticite, ax=ax_vorticity)
    ax_vorticity.set_title("Vorticité $u$")
    ax_vorticity.contour(x, y, masque, levels=[0.5], colors="black")  


    fig.suptitle(f"Temps final : {n * dt:.3f} s")
    fig_vorticity.suptitle(f"Temps final : {n * dt:.3f} s (Vorticité)")
    plt.pause(1e-20)


    save_vti(f"temps_final_{1000*n * dt:.3f}s.vti",u, v, x[0], y[:,0])

plt.ioff()  # Disable interactive mode
plt.show()


"------------------------------------------ Control plots ------------------------------------------------"
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

# Tracer la divergence
fig_divergence, ax_divergence = plt.subplots(figsize=(8, 6))
ax_divergence.plot(timeScale, divergence, label="divergence")
ax_divergence.set_xlabel('Temps')
ax_divergence.set_ylabel('Div u')
ax_divergence.set_title('Évolution de Div u')
ax_divergence.legend()
plt.show()

#print(energy,"\n\n",timeScale,"\n\n",exact_energy,"\n\n")