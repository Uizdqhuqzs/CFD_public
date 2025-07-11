import os
import numpy as np
import matplotlib.pyplot as plt
from modules_communs import lect_fichiers as lf

# Préfixe de fichier
prefix = "Wm_"

# Plage d’indices à tester
k, l = 2, 23

# Liste des résultats
resultats = []

prev_data = lf.read_file(f"{prefix}{k}.txt")

for i in range(k, l):
    filename = f"{prefix}{i}.txt"
    if os.path.exists(filename):
        try:
            data = lf.read_file(filename)
            resultats.append(np.linalg.norm(np.abs(data-prev_data)))
            prev_data = np.copy(data)
        except Exception as e:
            print(f"❌ Erreur lors de la lecture de {filename} : {e}")
    else:
        print(f"⚠️ Fichier non trouvé : {filename}")

plt.plot(list(range(k,l)), resultats, label="erreur")
plt.xlabel("itération")
plt.ylabel('L2 erreur')
plt.legend()
plt.title(f"Etude de convergence")
plt.grid()
plt.show()