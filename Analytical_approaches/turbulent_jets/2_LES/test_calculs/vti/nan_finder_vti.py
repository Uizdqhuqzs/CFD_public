import pyvista as pv
import numpy as np

# Lire le fichier .vti
input_filename = "N5_poids11_0p05.vti"  
output_filename = "isnan.vti"

grid = pv.read(input_filename)

# Fonction pour créer des masques NaN
def add_nan_masks(data_container, container_name):
    new_arrays = {}
    for name in data_container:
        array = np.asarray(data_container[name])

        # Champ vectoriel
        if array.ndim == 2 and array.shape[1] > 1:
            for i in range(array.shape[1]):
                mask = np.isnan(array[:, i]).astype(np.uint8)
                new_name = f"{name}_nan_{container_name}_component_{i}"
                new_arrays[new_name] = mask
        else:
            # Champ scalaire
            mask = np.isnan(array).astype(np.uint8)
            new_name = f"{name}_nan_{container_name}_mask"
            new_arrays[new_name] = mask

    # Ajout des nouveaux masques au container
    for new_name, mask in new_arrays.items():
        data_container[new_name] = mask

# Gérer point_data
add_nan_masks(grid.point_data, "point")

# Gérer cell_data
add_nan_masks(grid.cell_data, "cell")

# Sauvegarder le nouveau fichier .vti
grid.save(output_filename)

print(f"Fichier sauvegardé : {output_filename}")
