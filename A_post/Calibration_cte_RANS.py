from modules_calibration import cal_U_m, cal_k_m, cal_f_k

# Données du problème

U_0 = 72.5
d_0 = 0.00612

a_u, z_0 = cal_U_m.au_z0(U_0, d_0)

a_k, a_2 = cal_k_m.ak_a2(U_0, d_0, z_0)

bk1, bk2 = cal_f_k.bk1_bk2(U_0, d_0, a_u, z_0, a_k, a_2) 