import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

#vti_list = ["4.vti", "5.vti", "6.vti", "7.vti", "8.vti"]
vti_list = ["N5_poids11_0p05.vti"]

for vti_file in vti_list :

    print(f"Traitement de {vti_file}\n")

    # === Parametres === 
    output_dir = "avg_plots"
    dpi = 200  # qualite des images

    # === Chargement du fichier VTI ===
    grid = pv.read(vti_file)
    Nx, Ny, Nz = grid.dimensions
    dx, dy, dz = grid.spacing
    ox, oy, oz = grid.origin

    x = np.linspace(ox, ox + dx * (Nx - 1), Nx)
    y = np.linspace(oy, oy + dy * (Ny - 1), Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')



    # === Fonctions ===
    def average_along_z(arr_3d):
        return np.mean(arr_3d, axis=2)

    def plot_scalar_field(data2d, title, filename, cmap="viridis"):
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(X, Y, data2d.T, shading='auto', cmap=cmap)
        plt.colorbar(label=title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def cartesian_to_polar(X, Y):
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        return R, Theta

    def create_polar_grid(r_max, Nr, Ntheta):
        r = np.linspace(0, r_max, Nr)
        theta = np.linspace(-np.pi, np.pi, Ntheta)
        R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')
        Xp = R_grid * np.cos(Theta_grid)
        Yp = R_grid * np.sin(Theta_grid)
        return Xp, Yp, r, theta

    def interpolate_on_polar_grid(avg_field, x, y, Xp, Yp):
        interp = RegularGridInterpolator((x, y), avg_field, bounds_error=False, fill_value=np.nan)
        pts = np.stack((Xp.ravel(), Yp.ravel()), axis=-1)
        data_polar = interp(pts).reshape(Xp.shape)
        return data_polar

    def azimuthal_average(data_polar):
        return np.nanmean(data_polar, axis=1)

    # --- Fonctions ---

    # === Creation dossier de sortie ===
    os.makedirs(output_dir, exist_ok=True)

    # === Traitement des champs ===
    for name in grid.array_names:
        data = grid[name]

        if data.ndim == 1:
            arr = data.reshape((Nx, Ny, Nz), order='F')
            avg = average_along_z(arr)
            plot_scalar_field(avg, f"{name} (moyenne en z)", f"{output_dir}/{name}_zmoy_2D_{vti_file}.png")

            np.save(f"{output_dir}/{name}_zmoy.npy", avg)


        elif data.ndim == 2 and data.shape[1] == 3:
            vec = data.reshape((Nx, Ny, Nz, 3), order='F')
            for i, comp in enumerate('xyz'):
                avg = average_along_z(vec[..., i])
                plot_scalar_field(avg, f"{name}.{comp} (moyenne en z)", f"{output_dir}/{name}_{comp}_zmoy_2D_{vti_file}.png")

                np.save(f"{output_dir}/{name}_{comp}_zmoy.npy", avg)

        else:
            print(f"[!] Ignore : {name} (shape {data.shape})")

    print("\nTous les champs moyennés selon z sauvegardés dans :", output_dir)

    # === Azimutal average ===
    r_max = np.sqrt(np.max(x)**2 + np.max(y)**2)
    Nr = 300
    Ntheta = 600
    Xp, Yp, r, theta = create_polar_grid(r_max, Nr, Ntheta)
    polar_dir = os.path.join(output_dir, "curves")
    os.makedirs(polar_dir, exist_ok=True)

    for file in os.listdir(output_dir):
        if file.endswith("_zmoy.npy") and "coupe" not in file:
            champ_path = os.path.join(output_dir, file)
            champ_name = file.replace("_zmoy.npy", "")

            print(f"\n[Azimutal] Traitement : {champ_name}")

            champ_2d = np.load(champ_path)
            data_polar = interpolate_on_polar_grid(champ_2d, x, y, Xp, Yp)
            azim_avg = azimuthal_average(data_polar)

            azim_avg = np.nan_to_num(azim_avg, False, 0, 100000, -1000)

            np.save(os.path.join(polar_dir, f"{champ_name}_azim_avg.npy"), np.column_stack((r, azim_avg)))

            plt.figure()
            plt.plot(r, azim_avg)
            plt.xlabel("r")
            plt.ylabel(f"{champ_name} (moyenne azimutale)")
            plt.title(f"{champ_name} - Moyenne azimutale en fonction de r")
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(polar_dir, f"{champ_name}_1D_{vti_file}.png"), dpi=dpi)
            plt.close()

    print("\nTous les profils de vitesse moyenne sauvegardés dans :", polar_dir)



    #=================================== Compare Moyenne ===================================

    # === Fichier ===
    fichier = "avg_plots/curves/velocity_z_azim_avg.npy"


    # === Chargement des données ===
    data = np.load(fichier)

    y_vti, val_vti = data[:, 0], data[:, 1]

    # Application du facteur au vti
    val_vti_corrige = val_vti/val_vti[0]

    correctif_gamma = 1/val_vti[0]

    #print(f"Nan dans la vitesse ? {np.isnan(val_vti_corrige)}")

    # === Expresssion analytique ===

    lim_r = 10
    eta = np.linspace(-lim_r, lim_r, 1000)
    d = np.sqrt(2) - 1
    val_expr = 1/((d*eta)**2 + 1)**2 # Changement de repère : r/delta = eta * np.sqrt(d)  

    # === Affichage ===
    plt.figure()
    plt.plot(eta, val_expr, label="expression", color='blue')
    plt.plot(y_vti, val_vti_corrige, label=f"hairpins", color='red', linestyle='--')
    plt.title("Vitesse axiale en fonction du rayon (moyenne axiale + azimutale)")
    plt.xlabel("eta")
    plt.ylabel("VELO_z")
    plt.grid(True)
    plt.xlim(0, lim_r)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(polar_dir, f"velo_compare_{vti_file}.png"), dpi=dpi)
    plt.close()

    #=================================== Calcul k ===================================

    # === Paramètres ===

    output_dir = "TKE_plots"

    # === Chargement du fichier VTI ===
    '''grid = pv.read(vti_file)
    Nx, Ny, Nz = grid.dimensions
    dx, dy, dz = grid.spacing
    ox, oy, oz = grid.origin'''

    x = np.linspace(ox, ox + dx * (Nx - 1), Nx)
    y = np.linspace(oy, oy + dy * (Ny - 1), Ny)
    z = np.linspace(oz, oz + dz * (Nz - 1), Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # === Chargement des profils moyens 1D en r ===
    U_r_data = np.load(f"avg_plots/curves/velocity_x_azim_avg.npy")
    V_r_data = np.load(f"avg_plots/curves/velocity_y_azim_avg.npy")
    W_r_data = np.load(f"avg_plots/curves/velocity_z_azim_avg.npy")

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

    #correctif_gamma = 1

    tke = None

    for name in grid.array_names:
        if name == "velocity" :
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

                print(f"max u' = {np.max(u_fluct)}, max v' = {np.max(v_fluct)}, max w' = {np.max(w_fluct)}")

                u_var_3d = u_fluct ** 2
                v_var_3d = v_fluct ** 2
                w_var_3d = w_fluct ** 2

                # === Moyenne azimutale après interpolation ===

                
                u_var = np.nanmean(u_var_3d, axis=2)
                v_var = np.nanmean(v_var_3d, axis=2)
                w_var = np.nanmean(w_var_3d, axis=2)

                '''plot_scalar_field(u_var,"u'","u'.png")
                plot_scalar_field(v_var,"v'","v'.png")
                plot_scalar_field(w_var,"w'","w'.png")'''

                # === Moyenne azimutale de chaque variance ===
                r_max_interp = np.sqrt(np.max(x)**2 + np.max(y)**2)
                Nr = 300
                Ntheta = 500
                Xp, Yp, r, theta = create_polar_grid(r_max_interp, Nr, Ntheta)

                u_interp = interpolate_on_polar_grid(u_var, x, y, Xp, Yp)
                v_interp = interpolate_on_polar_grid(v_var, x, y, Xp, Yp)
                w_interp = interpolate_on_polar_grid(w_var, x, y, Xp, Yp)

                d = np.sqrt(2) - 1
                a_u = 0.12451280741522663
                b_u = (0.5/a_u) * (3*d)**0.5
                U_0 = 72.5
                d_0 = 0.00612

                a_k = 1.8166635470030046

                a_z = (b_u * U_0 * d_0)**2 / a_k

                u_azim_avg = azimuthal_average(u_interp) * a_z
                v_azim_avg = azimuthal_average(v_interp) * a_z
                w_azim_avg = azimuthal_average(w_interp) * a_z

                tke_azim_avg = 0.5 * (np.abs(u_azim_avg) + np.abs(v_azim_avg) + np.abs(w_azim_avg))

                np.save(os.path.join(output_dir, f"TKE_azim_avg.npy"), np.column_stack((r, tke_azim_avg)))



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
                plt.savefig(os.path.join(output_dir, f"Reynolds_stresses_{vti_file}.png"), dpi=dpi)
                plt.close()

                print("\nProfil azimutal de la TKE sauvegardé dans :", output_dir)


    #=================================== Compare k ===================================

    # === Fichier ===
    fichier = "TKE_plots/TKE_azim_avg.npy"


    # === Chargement des données ===
    data = np.load(fichier)

    y_vti, val_vti = data[:, 0], data[:, 1]

    # Application du facteur au vti
    val_vti_corrige = val_vti #/ val_vti[0] 

    #print(val_vti[0])

    # === Expresssion analytique ===

    # Données du problème 

    U_0 = 72.5
    d_0 = 0.00612
    z_demi = 0.0625
    d = np.sqrt(2) - 1

    b_k1 = 1.0
    b_k2 = 1.0

    lim_r = 10
    eta = np.linspace(-lim_r, lim_r, 1000)

    AAAAA = 1*np.sqrt(d)

    val_expr = b_k1 * ((eta*AAAAA)**2 + 1) * np.exp(-(eta*AAAAA)**2 + b_k2 - np.log(b_k1*np.exp(b_k2)))

    # === Affichage ===
    plt.figure()
    plt.plot(eta, val_expr, label="expression", color='blue')
    plt.plot(y_vti, val_vti_corrige, label=f"hairpins", color='red', linestyle='--')
    plt.title("énergie cinétique turbulente")
    plt.xlabel("eta")
    plt.ylabel("k")
    plt.grid(True)
    plt.xlim(0, lim_r)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"k_compare_{vti_file}.png"), dpi=dpi)
    plt.close()


    # Suppression des .npy
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".npy"):
                full_path = os.path.join(root, file)
                print(f"Suppression : {full_path}")
                os.remove(full_path)
