__version__ = "2023.07.21.14"


# 2023.07.21.14: modified two_species_ss_tau,
#                added PoissonMixtureSS, 
#                and modified other files
# 2023.05.13.12: add gene idx and use params to store all parameters for update_theta_j 
# 2023.05.08.08: CCC --> R2
# 2023.05.07.08: change to update global tau after 10 epoch
# 2023.05.02.14: add Ub
# 2023.05.01.10: add standard option for AIC and BIC
# 2023.04.26.23: change AIC to AICc, add BIC
# 2023.04.24.22: change CC around from 2 to 3; add norm_Q option
# 2023.04.24.15: change default lambda_tau to 0
# 2023.04.19.16: change bnd_tau to 0.1 np.min(np.diff(tau))
