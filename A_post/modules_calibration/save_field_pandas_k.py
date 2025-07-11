import pyvista as pv
import numpy as np
import pandas as pd

# Charger le fichier VTU
mesh = pv.read("calcul.vtu")

# Récupérer le temps / cycle si disponibles
time = mesh.field_data.get("TimeValue", [0.0])[0]
cycle = mesh.field_data.get("TimeIndex", [0])[0]

# Déterminer où se trouve le champ "_K"
k_field = None
coords = None

if "_K" in mesh.point_data:
    print("Champ '_K' trouvé dans les points.")
    k_field = mesh.point_data["_K"]
    coords = mesh.points[:, :2]  # x, y
elif "_K" in mesh.cell_data:
    print("Champ '_K' trouvé dans les cellules.")
    k_field = mesh.cell_data["_K"]
    # Calculer les centroïdes des cellules
    coords = mesh.cell_centers().points[:, :2]
else:
    raise ValueError("Champ '_K' non trouvé dans le fichier.")

# Créer un DataFrame propre
df = pd.DataFrame({
    "x_0": coords[:, 0],
    "x_1": coords[:, 1],
    "k": k_field
})

# Écrire dans un fichier texte lisible par pandas
with open("K_calc.txt", "w") as f:
    f.write("# \n")
    f.write("# Champs k\n")
    f.write("# \n")
    f.write('#     Domain: "fluid"\n')
    f.write("# \n")
    f.write(f"# Time = {time}   Cycle = {cycle}\n")
    f.write("# \n")
    f.write("#         x_0          x_1            k\n")
    df.to_csv(f, sep=" ", index=False, header=False, float_format="%.5E")

print("✅ Fichier 'K_field.txt' sauvegardé avec succès.")
