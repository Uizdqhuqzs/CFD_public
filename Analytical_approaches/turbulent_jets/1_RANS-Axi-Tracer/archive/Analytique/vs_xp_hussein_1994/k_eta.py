import numpy as np
import matplotlib.pyplot as plt

# Données du problème
U_0 = 72.5
d_0 = 0.00612

# Calibration pour z = 60dj
real_delta = 4.0886e-2
real_z = 0.3672
real_z0 = 0.012
real_U_axe = 5.59799E+00

z0 = real_z0
a_u = real_delta / (real_z - real_z0)
b_u = real_U_axe / (U_0 * d_0 / (real_z - real_z0))

print(f"a_u ={a_u}, b_u = {b_u}, z0 = {z0}")

r0 = 0

r = np.linspace(r0, 2, 1000)
z = np.linspace(z0, 2, 1000)

#r,z = np.meshgrid(r,z)

eta = np.linspace(0, 0.2, 10000)
#print(eta)


delta_u = a_u * (z - z0)

u_axe = np.minimum((np.zeros_like(z) + 1)*U_0, np.abs(U_0 * b_u * ((z - z0) / d_0) ** (-1)))  #u_m




# Définition des constantes
A = a_u/2  
B = np.sqrt(2) -1  # d

# Définition de la fonction

a_k = a_u**(-2)
k_m = 1 #a_k * (z - z0)

sigma_k = 1  
C = -7.9936227447946955 #-delta_u * u_axe * sigma_k
lamb = 0.22469644198031863
f_k = (lamb*np.exp(C * ( (A * ((2 / (B * (eta*a_k)**2 + 1)) + np.log(B * (eta*a_k)**2 + 1))) / (2 * B)))) #+1.032 +0.082
y = k_m * f_k

eta_0 = 0
#print(f"f = {(lamb*np.exp(C * ( (A * ((2 / (B * (eta_0*a_k)**2 + 1)) + np.log(B * (eta_0*a_k)**2 + 1))) / (2 * B))))}")

# Affichage du résultat
plt.plot(eta, f_k, label=r'$f(\eta)$', marker='o',linestyle='none')
#plt.plot(z, u_axe, label=r'$u_m$', marker=',',linestyle='none')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$f(\eta)$')
plt.legend()
plt.grid()

#plt.xlim(0, 10) #(0, 0.025) 
#plt.ylim(0, 10)

plt.show()
