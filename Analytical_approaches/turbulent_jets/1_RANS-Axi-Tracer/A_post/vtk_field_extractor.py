import pyvista as pv
import numpy as np
import pandas as pd

# Paramètre : nom du champ à extraire
nom_champ_extrait = "Wm"

# Charger le fichier VTU/PVTU
mesh = pv.read("save.pvtu")

# Récupérer le temps / cycle si disponibles
time = mesh.field_data.get("TimeValue", [0.0])[0]
cycle = mesh.field_data.get("TimeIndex", [0])[0]

# Déterminer où se trouve le champ
if nom_champ_extrait in mesh.point_data:
    print(f"✅ Champ '{nom_champ_extrait}' trouvé dans les points.")
    field = mesh.point_data[nom_champ_extrait]
    coords = mesh.points
elif nom_champ_extrait in mesh.cell_data:
    print(f"✅ Champ '{nom_champ_extrait}' trouvé dans les cellules.")
    field = mesh.cell_data[nom_champ_extrait]
    coords = mesh.cell_centers().points
else:
    print("🔍 Champs disponibles :")
    print(" - Points :", list(mesh.point_data.keys()))
    print(" - Cellules :", list(mesh.cell_data.keys()))
    raise ValueError(f"❌ Champ '{nom_champ_extrait}' non trouvé dans le fichier.")

# Récupérer les coordonnées x_0, x_1, (x_2 si 3D)
data = {}
for i in range(coords.shape[1]):
    data[f"x_{i}"] = coords[:, i]

# Gérer les champs scalaires, vectoriels ou tensoriels
if field.ndim == 1:
    data[nom_champ_extrait] = field
elif field.ndim == 2:
    for i in range(field.shape[1]):
        data[f"{nom_champ_extrait}_{i}"] = field[:, i]
else:
    raise ValueError(f"❌ Le champ '{nom_champ_extrait}' a une forme inattendue : {field.shape}")

df = pd.DataFrame(data)

# Écriture dans le fichier de sortie
filename = f"{nom_champ_extrait}_far.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write("# \n")
    f.write(f"# Champ extrait : {nom_champ_extrait}\n")
    f.write("# \n")
    f.write('#     Domain: "fluid"\n')
    f.write("# \n")
    f.write(f"# Time = {time}   Cycle = {cycle}\n")
    f.write("# \n")
    f.write("# " + " ".join(f"{col:>12}" for col in df.columns) + "\n")
    df.to_csv(f, sep=" ", index=False, header=False, float_format="%.5E")

print(f"✅ Fichier '{filename}' sauvegardé avec succès.")


