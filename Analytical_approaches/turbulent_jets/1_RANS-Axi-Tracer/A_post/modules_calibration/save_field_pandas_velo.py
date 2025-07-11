import pyvista as pv
import numpy as np
import pandas as pd

# Charger le fichier VTU
mesh = pv.read("saveT21.vtu")

# Récupérer le temps / cycle si disponibles
time = mesh.field_data.get("TimeValue", [0.0])[0]
cycle = mesh.field_data.get("TimeIndex", [0])[0]

# Déterminer où se trouve le champ "VELO"
k_field = None
coords = None

if "VELO" in mesh.point_data:
    print("Champ 'VELO' trouvé dans les points.")
    field = mesh.point_data["VELO"][:,1]
    coords = mesh.points[:, :2]  # x, y
elif "VELO" in mesh.cell_data:
    print("Champ 'VELO' trouvé dans les cellules.")
    field = mesh.cell_data["VELO"][:,1]
    # Calculer les centroïdes des cellules
    coords = mesh.cell_centers().points[:, :2]
else:
    raise ValueError("Champ 'VELO' non trouvé dans le fichier.")

# Créer un DataFrame propre
df = pd.DataFrame({
    "x_0": coords[:, 0],
    "x_1": coords[:, 1],
    "U_z": field
})

# Écrire dans un fichier texte lisible par pandas
with open("VELO_field.txt", "w") as f:
    f.write("# \n")
    f.write("# Champs k\n")
    f.write("# \n")
    f.write('#     Domain: "fluid"\n')
    f.write("# \n")
    f.write(f"# Time = {time}   Cycle = {cycle}\n")
    f.write("# \n")
    f.write("#         x_0          x_1            u_z\n")
    df.to_csv(f, sep=" ", index=False, header=False, float_format="%.5E")

print("✅ Fichier 'VELO_field.txt' sauvegardé avec succès.")
