import numpy as np
import matplotlib.pyplot as plt

# Données du problème
U_0 = 72.5
d_0 = 0.00612

# Calibration z0, a_u, b_u pour z = 60dj
real_delta = 4.0886e-2
real_z = 0.3672
real_z0 = 0.012
real_U_axe = 5.59799E+00

z0 = real_z0
a_u = real_delta / (real_z - real_z0)
b_u = real_U_axe / (U_0 * d_0 / (real_z - real_z0))

print(f"a_u ={a_u}, b_u = {b_u}, z0 = {z0}")

eta = np.linspace(0, 0.5, 10000)

'''r0 = 0

r = np.linspace(r0, 2, 1000)
z = np.linspace(z0, 2, 1000)

#r,z = np.meshgrid(r,z)

#print(eta)


delta_u = a_u * (z - z0)

u_axe = np.minimum((np.zeros_like(z) + 1)*U_0, np.abs(U_0 * b_u * ((z - z0) / d_0) ** (-1)))  #u_m'''

# Calibration lamb et C pour z = 60dj

k_m_60dj = 3.4783036913627714
fk0 = 0.6/k_m_60dj

fk_eta = 0.1 #0.3
fk_im = 0.48097/k_m_60dj #0.00936

echelle_eta = 1#/a_u**2

d = np.sqrt(2) - 1
a_u =0.11510698198198198
D = d*(fk_eta*echelle_eta)**2 +1

def g(a_u, d, eta): # G*(eta)

    eta =  eta*echelle_eta

    return - (a_u/2 * (2 / (d * eta**2 + 1) + np.log(d * eta**2 + 1))) / (2 * d)

print(g(a_u,d,0))

C = np.log(fk_im/fk0) / (g(a_u,d,fk_eta) - g(a_u,d,0))
lamb = fk0/np.exp(C*g(a_u,d,0))

print(f"Calibration : Ck_exp = {C} et lambda = {lamb}")

'''C = 0.46998082
lamb = 1.8876389'''

# Définition des constantes de fk
A = a_u/2  
B = np.sqrt(2) -1  # d

# Calcul de fk
k_m = k_m_60dj #a_k * (z - z0)

sigma_k = 1  
'''C = -4.920694180233765 #-delta_u * u_axe * sigma_k
lamb = 0.34175624814061345'''
f_k = (lamb*np.exp(C * -( (A * ((2 / (B * (eta*echelle_eta)**2 + 1)) + np.log(B * (eta*echelle_eta)**2 + 1))) / (2 * B)))) #+1.032 +0.082
y = k_m * f_k

eta_0 = 0
#print(f"f = {(lamb*np.exp(C * ( (A * ((2 / (B * (eta_0*a_k)**2 + 1)) + np.log(B * (eta_0*a_k)**2 + 1))) / (2 * B))))}")

# Affichage du résultat
plt.plot(eta, y, label=r'$k(\eta)$', marker='o',linestyle='none',color='green')
#plt.plot(z, u_axe, label=r'$u_m$', marker=',',linestyle='none')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$k$')
plt.legend()
plt.grid()

plt.title("k analytique cal @k0p5")
#plt.xlim(0, 0.35) #(0, 0.025) 
#plt.ylim(0, 0.8)

plt.show()
