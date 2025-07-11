import numpy as np
import matplotlib.pyplot as plt

k_m_60dj = 3.4783036913627714
fk0 = 0.6/k_m_60dj

fk_eta = 0.15
fk_im = fk0/2

d = np.sqrt(2) - 1
a_u =0.11510698198198198
D = d*(fk_eta/a_u**2)**2 +1

def g(A, B, eta): # G*(eta)

    eta =  eta/A**2

    return - (A/2 * (2 / (B * eta**2 + 1) + np.log(B * eta**2 + 1))) / (2 * B)

print(g(a_u,d,0))

Ck_exp = np.log(fk_im/fk0) / (g(a_u,d,fk_eta) - g(a_u,d,0))
lamb = fk0/np.exp(Ck_exp*g(a_u,d,0))

print(f"Calibration : Ck_exp = {Ck_exp} et lambda = {lamb}")

'''grand_C = ((4*d/a_u)*np.log(0.024/lamb))/((2/D + np.log(D)))

print(grand_C)


#  0.024 = 0.074/exp(x*0.139)  *  exp(x*(a_u/2) * (2/D + ln(D)) /(2*d))


0.024 = 0.074/exp(x*0.139)  *  exp(x*(0.11510698198198198/2) * (2/54.088464470082904 + ln(54.088464470082904)) /(2*0.41421356237309515))'''

eta_arr = np.linspace(0,0.5,1000)

plt.plot(eta_arr, g(a_u,d,eta_arr), label=r'$f(\eta)$', marker='o',linestyle='none')
#plt.plot(z, u_axe, label=r'$u_m$', marker=',',linestyle='none')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$f(\eta)$')
plt.legend()
plt.grid()

#plt.xlim(0, 0.35) #(0, 0.025) 
#plt.ylim(0, 1)

plt.show()