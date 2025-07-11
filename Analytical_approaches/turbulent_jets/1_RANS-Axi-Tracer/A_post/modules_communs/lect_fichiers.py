import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# renvoie un dataframe pandas avec les données du fichier filenames (nécessite au moins des coordonnées 2D nommées "x_0" et "x_1")

def read_file(file_name):

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

    # Vérification de la présence des colonnes x_0 et x_1
    if "x_0" not in df.columns or "x_1" not in df.columns:
        raise ValueError("Les colonnes 'x_0' et 'x_1' sont requises mais manquent dans le fichier de données.")

    '''# Structuration des données
    data = {}
    for col in column_names:
        data[col] = df[col].to_numpy()'''

    return df


