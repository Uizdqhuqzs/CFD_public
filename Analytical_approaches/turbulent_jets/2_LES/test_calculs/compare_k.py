import numpy as np
import matplotlib.pyplot as plt

# === Fichier ===
fichier = "TKE_plots/azim_averaged/TKE_azim_avg_8.vti.npy"


# === Chargement des données ===
data = np.load(fichier)

y_vti, val_vti = data[:, 0], data[:, 1]

d = np.sqrt(2) - 1
a_u = 0.12451280741522663
b_u = (0.5/a_u) * (3*d)**0.5
U_0 = 72.5
d_0 = 0.00612

a_k = 1.8166635470030046

a_z = b_u * U_0 * d_0

# Application du facteur au vti
val_vti_corrige = val_vti #* (a_z**2 /a_k) #/ val_vti[0] # 

print(f"changement adim = {(a_z**2 /a_k)}")

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
