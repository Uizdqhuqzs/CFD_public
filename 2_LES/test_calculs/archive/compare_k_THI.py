import numpy as np
import matplotlib.pyplot as plt

# === Fichier ===
fichier = "velocity_TKE_azim_avg.npy"


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
plt.show()
