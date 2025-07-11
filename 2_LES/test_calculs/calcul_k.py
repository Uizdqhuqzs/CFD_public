import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, RegularGridInterpolator


def create_polar_grid(r_max, Nr, Ntheta):
    r = np.linspace(0, r_max, Nr)
    theta = np.linspace(-np.pi, np.pi, Ntheta)
    R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')
    Xp = R_grid * np.cos(Theta_grid)
    Yp = R_grid * np.sin(Theta_grid)
    return Xp, Yp, r, theta

def interpolate_on_polar_grid(field2d, x, y, Xp, Yp):
    interp = RegularGridInterpolator((x, y), field2d, bounds_error=False, fill_value=np.nan)
    pts = np.stack((Xp.ravel(), Yp.ravel()), axis=-1)
    data_polar = interp(pts).reshape(Xp.shape)
    return data_polar

def azimuthal_average(data_polar):
    return np.nanmean(data_polar, axis=1)

# === Paramètres ===
vti_file = "4.vti"
output_dir = "TKE_plots"
dpi = 200
correctif_gamma = 1 / 0.10948188224413068  # 0.10571 si nécessaire

# === Chargement du fichier VTI ===
grid = pv.read(vti_file)
Nx, Ny, Nz = grid.dimensions
dx, dy, dz = grid.spacing
ox, oy, oz = grid.origin

x = np.linspace(ox, ox + dx * (Nx - 1), Nx)
y = np.linspace(oy, oy + dy * (Ny - 1), Ny)
z = np.linspace(oz, oz + dz * (Nz - 1), Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# === Chargement des profils moyens 1D en r ===
U_r_data = np.load("avg_plots/azim_averaged/velocity_x_azim_avg.npy")
V_r_data = np.load("avg_plots/azim_averaged/velocity_y_azim_avg.npy")
W_r_data = np.load("avg_plots/azim_averaged/velocity_z_azim_avg.npy")

r_prof = U_r_data[:, 0]
U_r = U_r_data[:, 1] * correctif_gamma
V_r = V_r_data[:, 1] * correctif_gamma
W_r = W_r_data[:, 1] * correctif_gamma

r_max = np.max(r_prof)

# === Grille cylindrique pour interpolation ===
Nr = len(r_prof)
Ntheta = 200
Nz = grid.dimensions[2]

r_cyl = r_prof
theta_cyl = np.linspace(-np.pi, np.pi, Ntheta)
z_cyl = np.linspace(oz, oz + dz * (Nz - 1), Nz)

R_cyl, Theta_cyl, Z_cyl = np.meshgrid(r_cyl, theta_cyl, z_cyl, indexing='ij')

U_cyl = np.repeat(U_r[:, np.newaxis, np.newaxis], Ntheta, axis=1)
U_cyl = np.repeat(U_cyl, Nz, axis=2)
V_cyl = np.repeat(V_r[:, np.newaxis, np.newaxis], Ntheta, axis=1)
V_cyl = np.repeat(V_cyl, Nz, axis=2)
W_cyl = np.repeat(W_r[:, np.newaxis, np.newaxis], Ntheta, axis=1)
W_cyl = np.repeat(W_cyl, Nz, axis=2)

X_cyl = R_cyl * np.cos(Theta_cyl)
Y_cyl = R_cyl * np.sin(Theta_cyl)

U_cart_cyl = U_cyl * np.cos(Theta_cyl) - V_cyl * np.sin(Theta_cyl)
V_cart_cyl = U_cyl * np.sin(Theta_cyl) + V_cyl * np.cos(Theta_cyl)
W_cart_cyl = W_cyl

# === Interpolation sur la grille cartésienne ===
interp_U = RegularGridInterpolator((r_cyl, theta_cyl, z_cyl), U_cart_cyl, bounds_error=False, fill_value=0)
interp_V = RegularGridInterpolator((r_cyl, theta_cyl, z_cyl), V_cart_cyl, bounds_error=False, fill_value=0)
interp_W = RegularGridInterpolator((r_cyl, theta_cyl, z_cyl), W_cart_cyl, bounds_error=False, fill_value=0)

pts_cyl = np.stack((R.ravel(), Theta.ravel(), Z.ravel()), axis=-1)  # issu de la grille cartésienne

U_mean = interp_U(pts_cyl).reshape(X.shape)
V_mean = interp_V(pts_cyl).reshape(X.shape)
W_mean = interp_W(pts_cyl).reshape(X.shape)

# === Calcul de la TKE ===
os.makedirs(output_dir, exist_ok=True)


tke = None

for name in grid.array_names:
    data = grid[name]

    if data.ndim == 2 and data.shape[1] == 3:
        data *= correctif_gamma
        vec = data.reshape((Nx, Ny, Nz, 3), order='F')

        u = vec[..., 0]
        v = vec[..., 1]
        w = vec[..., 2]

        u_fluct = u - U_mean
        v_fluct = v - V_mean
        w_fluct = w - W_mean

        u_var_3d = u_fluct ** 2
        v_var_3d = v_fluct ** 2
        w_var_3d = w_fluct ** 2

        # === Moyenne azimutale après interpolation ===

        
        u_var = np.nanmean(u_var_3d, axis=2)
        v_var = np.nanmean(v_var_3d, axis=2)
        w_var = np.nanmean(w_var_3d, axis=2)

        # === Moyenne azimutale de chaque variance ===
        r_max_interp = np.sqrt(np.max(x)**2 + np.max(y)**2)
        Nr = 300
        Ntheta = 500
        Xp, Yp, r, theta = create_polar_grid(r_max_interp, Nr, Ntheta)

        u_interp = interpolate_on_polar_grid(u_var, x, y, Xp, Yp)
        v_interp = interpolate_on_polar_grid(v_var, x, y, Xp, Yp)
        w_interp = interpolate_on_polar_grid(w_var, x, y, Xp, Yp)

        u_azim_avg = azimuthal_average(u_interp)
        v_azim_avg = azimuthal_average(v_interp)
        w_azim_avg = azimuthal_average(w_interp)

        tke_azim_avg = 0.5 * (u_azim_avg + v_azim_avg + w_azim_avg)

        polar_dir = os.path.join(output_dir, "azim_averaged")
        os.makedirs(polar_dir, exist_ok=True)

        np.save(os.path.join(polar_dir, "TKE_azim_avg.npy"), np.column_stack((r, tke_azim_avg)))



        plt.figure()
        plt.plot(r, tke_azim_avg, label="TKE")
        plt.plot(r, u_azim_avg, label="uu")
        plt.plot(r, v_azim_avg, label="vv")
        plt.plot(r, w_azim_avg, label="ww")
        plt.xlabel("eta")
        plt.ylabel("TKE (moyenne azimutale)")
        plt.title("TKE - Moyenne azimutale en fonction de eta")
        plt.grid()
        plt.legend()
        plt.xlim(0,5)
        plt.tight_layout()
        plt.savefig(os.path.join(polar_dir, "TKE_azim_avg.png"), dpi=dpi)
        plt.close()

        print("\nProfil azimutal de la TKE sauvegardé dans :", polar_dir)
