import numpy as np
import matplotlib.pyplot as plt

# === Fichier ===
fichier = "avg_plots/azim_averaged/velocity_z_azim_avg.npy"


# === Chargement des données ===
data = np.load(fichier)

y_vti, val_vti = data[:, 0], data[:, 1]

# Application du facteur au vti
val_vti_corrige = val_vti/val_vti[0]

print(f"facteur correctif gamma {val_vti[0]}")

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
plt.show()
