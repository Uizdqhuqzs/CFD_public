import numpy as np
import matplotlib.pyplot as plt

z0 = 0.012
a_u =0.11510698198198198
dj = 0.00612
d = np.sqrt(2) - 1
echelle_eta = 1 #/a_u**1

A = 0.48097 # k(60dj, 0.1)
zA = 60*dj
B = 0.5938 #0.8922 # k(40dj, 0)
zB = 60*dj

a = (1/B - 1/A) / (-((zA - z0)/(d*(0.1*echelle_eta)**2 + 1))**2 +(zB - z0)**2)
b = 1/A - a*((zA - z0)/(d*(0.1*echelle_eta)**2 + 1))**2


'''a = -380.3277735741214
b = 49.668898131640475 '''


print(f"a = {a} et b = {b} \n ")

eta = np.linspace(0, 5.3, 10000)

z = 60*dj
k = (a*((z-z0)/(d*(eta)**2 + 1))**2 + b)**(-1) #+ 2*0.5938

plt.plot(eta, k, label=r'$f(\eta)$', marker='o',linestyle='none',color='red')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$f(\eta)$')
plt.legend()
plt.grid()

plt.xlim(0, 0.35) #(0, 0.025) 
plt.ylim(0, 0.8)

plt.show()
