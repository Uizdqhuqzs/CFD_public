import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#__________________________________________________________________Création de la tructure de données__________________________________________________________________

# Demander à l'utilisateur le nom du fichier à analyser
file_name = "radial_60dj_00021.txt" #input("Entrez le nom du fichier à analyser : ")

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

# Chargement des données avec les noms de colonnes pré-définis
df = pd.read_csv(file_name, delim_whitespace=True, comment="#", skiprows=4, header=None)
df.columns = column_names

# Vérification de la présence des colonnes x_0 et x_1
if "x_0" not in df.columns or "x_1" not in df.columns:
    raise ValueError("Les colonnes 'x_0' et 'x_1' sont requises mais manquent dans le fichier de données.")

# Structuration des données
data = {"x_0": df["x_0"].to_numpy(), "x_1": df["x_1"].to_numpy(), "champs": {}}
for col in column_names:
    if col not in ["x_0", "x_1"]:
        data["champs"][col] = df[col].to_numpy()


#__________________________________________________________________Définition des champs à traçer__________________________________________________________________

#Données du problème
U_0 = 72.5
d_0 = 0.00612

#calibration pour z = 60dj
real_delta = 4.0886e-2
real_z = 0.3672
real_z0 = 0.09


z0 = real_z0
a_u = real_delta/(real_z - real_z0) #0.123821926
b_u = 5.59799E+00 / (U_0*d_0/(real_z - real_z0))


delta_u = a_u*(data["x_1"] - z0)
u_axe = U_0*b_u/((data["x_1"] - z0)/d_0)


x_scale = 1/(delta_u)
y_scale = 1/u_axe

champs_trace = "v"

x = data["x_0"]*x_scale #eta

aff = data["champs"][champs_trace]*y_scale
aff_label = champs_trace

#print(data["champs"][champs_trace])
print("b_u = ",data["champs"][champs_trace][0]/u_axe[0])

#__________________________________________________________________Parsing données xp__________________________________________________________________

# Initialisation des données expérimentales
exp_x, exp_y = None, None  # Valeurs par défaut

# Charger les données expérimentales si le fichier existe
exp_file = "w-affinity-Exp.txt"

if os.path.exists(exp_file):
    try:
        exp_data = np.loadtxt(exp_file)
        exp_x = exp_data[:, 0]  # Première colonne
        exp_y = exp_data[:, 1]  # Deuxième colonne
    except Exception as e:
        print(f"Erreur lors du chargement de {exp_file} : {e}")
else:
    print(f"Le fichier {exp_file} n'existe pas. Aucune donnée expérimentale ne sera affichée.")



#__________________________________________________________________Affichage__________________________________________________________________

# Affichage du résultat avec ou sans données expérimentales
plt.figure(figsize=(10, 6))
plt.plot(x, aff, label="Simulation", linestyle='-', marker=',')

# Ajouter les données expérimentales seulement si elles existent
if exp_x is not None and exp_y is not None:
    plt.plot(exp_x, exp_y, label="Expérimental", linestyle='--', marker='o', color='red')

plt.xlabel("$\eta$")
plt.ylabel(aff_label)
plt.legend()
plt.title(f"Évolution de {aff_label}")
plt.grid()
plt.show()


'''# Affichage du résultat
plt.figure(figsize=(10, 6))
plt.plot(x, aff, label=aff_label)
plt.xlabel("$\eta$")
plt.ylabel(aff_label)
plt.legend()
plt.title(f"Évolution de {aff_label}")
plt.grid()
plt.show()'''
