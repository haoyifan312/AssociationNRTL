import matplotlib.pyplot as plt
import numpy as np

from AssociationNRTL import AssociationNRTL

# This file illustrates the functionality of AssociationNRTL class
# reference Hao and Chen, AIChE J. 2020

# Use chloroform-water binary case as an example
# 1-chloroform 2-water
ri = np.array([2.87, 0.76])    # r_I parameters as in Eqn 3, 10, 11

tauij = np.zeros((2, 2))  # tau_IJ as in Eqn 4, 5
tauij[0][1] = -1.21
tauij[1][0] = 2.85

alphaij = np.zeros((2, 2))  # alpha_IJ as in Eqn 5, 6
alphaij[0][1] = alphaij[1][0] = 0.2

nu = [[1], [2, -2]]     # nu_A,I as in Eqn 7, 10, 11; positive for HB donor sites and negative for acceptor sites
delta = [[0.145], [1.0, 1.0]]   # delta^A or delta^D as in Eqn 12

# Association NRTL class is constructed by model parameters
ChloroformWaterBinary = AssociationNRTL(r_i=ri,
                                        tau_ij=tauij,
                                        alpha_ij=alphaij,
                                        nu_i=nu,
                                        delta_i=delta)

# activity coefficient is calculated by compute function by pass in temperature and mole fraction
x = np.array([0.2, 0.8])
T = 300.0   # K
lnGamma = ChloroformWaterBinary.compute(x, T)
print(f'Calculated ln(gamma):\n{lnGamma}')

# additional information of calculated values can be retrieved by passing in an empty dictionary
info = {}
lnGamma = ChloroformWaterBinary.compute(x, T, info)
print(f'Additional calculated values:\n{info}')

# plot gamma and unbonded site fraction at different composition
x_CCL3 = np.linspace(0.0, 1.0, 51)
x_H2O = 1.0-x_CCL3
points = x_CCL3.shape[0]
gamma_CCL3 = np.zeros(points)
gamma_H2O = np.zeros(points)
gammaC_CCL3 = np.zeros(points)
gammaC_H2O = np.zeros(points)
gammaR_CCL3 = np.zeros(points)
gammaR_H2O = np.zeros(points)
gammaA_CCL3 = np.zeros(points)
gammaA_H2O = np.zeros(points)
xD_CCL3 = np.zeros(points)
xD_H2O = np.zeros(points)
xA_H2O = np.zeros(points)
T = 298.15
info = {}
for i in range(points):
    x = np.array([x_CCL3[i], 1.0-x_CCL3[i]])
    gamma = ChloroformWaterBinary.compute(x, T, info)
    gamma_CCL3[i], gamma_H2O[i] = gamma
    gammaC_CCL3[i], gammaC_H2O[i] = info['lnGammaC']
    gammaR_CCL3[i], gammaR_H2O[i] = info['lnGammaR']
    gammaA_CCL3[i], gammaA_H2O[i] = info['lnGammaA']
    xA_H2O[i], xD_CCL3[i], xD_H2O[i] = info['xA']

# example to use calculated info to plot Fig 3b
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(x_H2O, gamma_H2O, linestyle='-',
         color='black', label=r'$\gamma$')
ax1.plot(x_H2O, gamma_CCL3, linestyle='-', color='black')
ax1.plot(x_H2O, gammaC_H2O, linestyle=':',
         color='black', label=r'$\gamma^C$')
ax1.plot(x_H2O, gammaC_CCL3, linestyle=':', color='black')
ax1.plot(x_H2O, gammaR_H2O, linestyle='--',
         color='black', label=r'$\gamma^R$')
ax1.plot(x_H2O, gammaR_CCL3, linestyle='--', color='black')
ax1.plot(x_H2O, gammaA_H2O, linestyle='-.',
         color='black', label=r'$\gamma^A$')
ax1.plot(x_H2O, gammaA_CCL3, linestyle='-.', color='black')
ax1.legend()
ax1.set_title('activity coefficient contributions')
ax1.set_xlabel('Water (mole frac)')
ax1.set_xlim(0.0, 1.0)
ax1.set_ylabel(r'$ln \gamma$')

ax2.plot(x_H2O, xA_H2O, linestyle='--', color='black',
         label=r'$X^{acceptor}_{H2O}$')
ax2.plot(x_H2O, xD_H2O, linestyle='-', color='black',
         label=r'$X^{donor}_{H2O}$')
ax2.plot(x_H2O, xD_CCL3, linestyle='-.', color='black',
         label=r'$X^{donor}_{chloroform}$')
ax2.legend()
ax2.set_title('unbonded site fractions')
ax2.set_xlabel('Water (mole frac)')
ax2.set_xlim(0.0, 1.0)
ax2.set_ylabel('Free site fraction')
ax2.set_ylim(0.0, 1.0)

plt.tight_layout()
plt.show()
