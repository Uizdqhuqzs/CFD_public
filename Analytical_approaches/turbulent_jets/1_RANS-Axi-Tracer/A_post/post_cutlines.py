import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#__________________________________________________________________Création de la structure de données__________________________________________________________________

num = 45

# Nom du fichier de simulation
file_name = f"axis_z_000{num}.txt"  # input("Entrez le nom du fichier à analyser : ")

if not os.path.exists(file_name):
    raise FileNotFoundError(f"Le fichier {file_name} n'existe pas.")

# Lecture des noms de colonnes
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

# Chargement des données avec les noms de colonnes
df = pd.read_csv(file_name, delim_whitespace=True, comment="#", skiprows=4, header=None)
df.columns = column_names

# Vérification de la présence des colonnes x_0 et x_2
if "x_0" not in df.columns or "x_1" not in df.columns or "x_2" not in df.columns:
    raise ValueError("Les colonnes 'x_0', 'x_1' et 'x_2' sont requises mais manquent dans le fichier de données.")

# Structuration des données
data = {"x_0": df["x_0"].to_numpy(), "x_1": df["x_1"].to_numpy(), "x_2": df["x_2"].to_numpy(), "champs": {}}
for col in column_names:
    if col not in ["x_0", "x_1", "x_2"]:
        data["champs"][col] = df[col].to_numpy()

#__________________________________________________________________Adimensionnement__________________________________________________________________

'''# Constantes
a_u = 0.09334657661978871 
z_0 = 0.008961378021260326
U_0 = 72.5
d_0 = 0.00612
d = np.sqrt(2) - 1
b_u = (0.5/a_u) * (3*d)**0.5
sigma_k = 1

print(f"a_u = {a_u} et b_u = {b_u}")

delta_u = a_u * (data["x_2"] - z_0)
u_axe = U_0 * b_u * d_0 /(data["x_2"] - z_0)
eta = data["x_0"] / (delta_u) #*a_u'''

#__________________________________________________________________Définition des champs à tracer__________________________________________________________________


x = data["x_2"]
aff = data["champs"]["M(w)_0"]
aff_label = "M(w)_0"

#print("b_u =", data["champs"][champs_trace][0] / u_axe[0])

#__________________________________________________________________Sauvegarde des données simulées__________________________________________________________________

sauvegarder = True

if sauvegarder:
    simu_file = f"M(w)_0_{num}.txt"
    try:
        np.savetxt(simu_file, np.column_stack((x, aff)), fmt="%.6f", delimiter=" ", header="# eta valeur", comments="")
        print(f"Données de simulation enregistrées dans {simu_file}.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de {simu_file} : {e}")

#__________________________________________________________________Parsing des données supplémentaires__________________________________________________________________

# Liste des fichiers à parser
exp_files = [
    "M(w)_0_50.txt",
    "M(w)_0_49.txt",
    "M(w)_0_48.txt",
    "M(w)_0_47.txt",
    "M(w)_0_46.txt",
    "M(w)_0_45.txt",
    "M(w)_0_.txt",
]

# Liste pour stocker les données supplémentaires
exp_data_list = []

# Charger chaque fichier expérimental s'il existe
for exp_file in exp_files:
    if os.path.exists(exp_file):
        try:
            exp_data = np.loadtxt(exp_file)
            if exp_data.shape[1] >= 2:  # Vérifier s'il y a au moins deux colonnes
                exp_data_list.append((exp_data[:, 0], exp_data[:, 1], exp_file))
                print(f"✔ Données chargées depuis {exp_file}")
            else:
                print(f"⚠ Le fichier {exp_file} ne contient pas suffisamment de colonnes.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {exp_file} : {e}")
    else:
        print(f"⚠ Le fichier {exp_file} n'existe pas. Il ne sera pas affiché.")

#__________________________________________________________________Affichage__________________________________________________________________

plt.figure(figsize=(10, 6))

# Tracé de la simulation 
plt.plot(x, aff, label=f"Simu - {file_name}", linestyle='-', marker=',', color='blue')

# Ajouter les données expérimentales disponibles avec des couleurs et markers distincts
colors = ["red", "green", "purple", "orange", "brown", "cyan", "black"]
markers = [",", ",", ",", "*", "^", "v", "o"]

for i, (exp_x, exp_y, exp_file) in enumerate(exp_data_list):
    label_prefix = "Simu -" if ("simu" or "Simu") in exp_file.lower() else "Exp -"
    plt.plot(exp_x, exp_y, label=f"{label_prefix} {exp_file}", linestyle='-', marker=markers[i % len(markers)], color=colors[i % len(colors)])

plt.xlabel("z")
plt.ylabel(aff_label)
plt.legend()
plt.title(f"Évolution de {aff_label}")
plt.grid()
plt.show()
