from .inference import *
from .mixtures import *
from .plotting import *
from .utils import *
from .simulation import *


__version__ = "2024.10"

# 2024.10: (1) change normlize_Q by normlizing per lineage; (2) add store_info option and harmonize elbos and remove Q_hist. Now elbos record the elbo along epoch and all_elbos record elbos of all initialization if store_info is true.
# 2023.09.01: add compute_gene_logL
# 2023.08.16: change init_Q by removing one added
# 2023.08.14: change initial beta and gamma to ratio and one
# 2023.07.31: corrected PMSS: added update weights
# 2023.07.26: change initial beta and gamma back to before
# 2023.07.25: change initial beta and gamma,
#             change default fit_global_tau to False with tau model,
#             return to old normalize_Q but include weights
# 2023.07.24: store all Q
# 2023.07.21: modified two_species_ss_tau,
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
