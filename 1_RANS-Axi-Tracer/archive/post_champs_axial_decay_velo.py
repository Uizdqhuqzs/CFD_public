import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Demander à l'utilisateur le nom du fichier à analyser
file_name = "cutline_1_00051.txt" #input("Entrez le nom du fichier à analyser : ")

if not os.path.exists(file_name):
    raise FileNotFoundError(f"Le fichier {file_name} n'existe pas.")

# Lecture des noms de colonnes à partir du fichier
with open(file_name, 'r') as f:
    lines = f.readlines()

# Identifier la ligne contenant les noms des colonnes (dernière ligne commençant par "#")
column_names = []
for line in reversed(lines):
    if line.startswith("#"):
        column_names = line.strip("# ").split()
        break

if not column_names:
    raise ValueError("Impossible d'identifier les noms de colonnes.")

# Vérification des deux premières colonnes
expected_columns = ["x_0", "x_1"]
if column_names[:2] != expected_columns:
    raise ValueError(f"Les deux premières colonnes doivent être {expected_columns}, mais sont {column_names[:2]}.")

# Chargement des données avec les noms de colonnes pré-définis
df = pd.read_csv(file_name, delim_whitespace=True, comment="#", skiprows=4, header=None)
df.columns = column_names

# Structuration des données
data = {"x0": df["x_0"].to_numpy(), "x1": df["x_1"].to_numpy(), "champs": {}}
for col in column_names[2:]:
    data["champs"][col] = df[col].to_numpy()

# Définition de l'expression de aff en fonction des champs disponibles

#w = 0.5*72.5*(1 - np.tanh((0.00612*0.5 - data["x0"])/(0.00612/16)))

x_scale = (1/(0.00612*0.5))
y_scale = np.ones_like(data["x1"])/72.5


aff = 1/(data["champs"]["V_1"] * y_scale)
x = data["x1"]*x_scale

print(x)

# Affichage du résultat
plt.figure(figsize=(10, 6))
plt.plot(x, aff, label='v')
plt.xlabel("$z/r_j$")
plt.ylabel("$w_j/w$")
plt.legend()
plt.title(f"Évolution de {"v"}")
plt.grid()
plt.show()
