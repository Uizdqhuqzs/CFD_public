import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Données du problème 

U_0 = 72.5
d_0 = 0.00612
z_demi = 0.0625
d = np.sqrt(2) - 1
sigma_k = 1

a_u = 0.12451280741522663
z_0 = 0.012854006829950816
b_u = (0.5/a_u) * (3*d)**0.5

a_k = 1.8166635470030046
a_2 = 1.9611604032637748

b_k1 = 1.0
b_k2 = 1.0