import numpy as np
from gamma_functions import flory_huggins, nrtl, association_gamma


class AssociationNRTL:
    def __init__(self,
                 r_i: np.array,
                 tau_ij,
                 alpha_ij: np.ndarray,
                 nu_i,
                 delta_i,
                 delta_ref: list = None,
                 solver_verbose: int = 0,
                 solver_method: str = 'dogbox'):
        """
        Association NRTL model, reference Hao and Chen, AIChE J. 2020
        :param r_i: normalized volume parameters of each component i as in Eqn 3, 7, 10, 11
        :param tau_ij: NRTL binary parameters between component i and j as in Eqn 4 and 5
        argument can be numpy ndarray of square matrix, which will be temperature independent
        or a temperature (K) dependent function tau(T) that returns numpy ndarray
        :param alpha_ij: non-randomness factor as in Eqn 5, numpy ndarray square matrix
        :param nu_i: number of different HB donor and acceptor sites for each component
        as nu_I^A in Eqn 7, 10, and 11
        here positive - HB donor sites, negative - HB acceptor sites
        e.g. [[], [1], [1, -2], [1, -1, -2]] represent a four-component system:
        1. no association site; 2. 1 donor site; 3. 1 donor site and 2 acceptor site;
        4. 1 donor site, two types of HB acceptor sites which has 1 and 2 sites on the molecule respectively
        :param delta_i: association strength parameter, delta^A or delta^D in Eqn 12
        e.g. [[], [0.33], [1.0, 1.0], [1.0, 0.5, 1.0]]
        :param delta_ref: reference association strength parameter kappa_ref^AD and epsilon_ref^AD in Eqn 13,
        default value 0.034 and 1960
        :param solver_verbose: optional, default 0,
        verbose option to print solver info, refer to numpy.optimize.least_square verbose argument
        :param solver_method: optional, default 'dogbox', solver method to solve unbonded site fraction xA,
        refer to numpy.optimize.least_square method argument
        """
        self.r = r_i
        self.size = r_i.shape[0]
        self.temp_dep_tau = False
        if callable(tau_ij):
            self.temp_dep_tau = True
            self.tau_fun = tau_ij
        else:
            self.tau = tau_ij
            if tau_ij.shape != (self.size, self.size):
                raise Exception(f'tau_ij size {tau_ij.shape} is inconsistent with r_i size {self.size}')
        self.alpha = alpha_ij
        if alpha_ij.shape != (self.size, self.size):
            raise Exception(f'alpha_ij size {alpha_ij.shape} is inconsistent with r_i size {self.size}')
        self.nu = nu_i
        self.delta = delta_i
        if len(nu_i) != self.size or len(delta_i) != self.size:
            raise Exception(f'nu_i size {len(nu_i)} or delta_i size {len(delta_i)} '
                            f'is inconsistent with r_i size {self.size}')
        self.kappa_ref = 0.034
        self.eps_ref = 1960.0
        if delta_ref is not None:
            self.kappa_ref, self.eps_ref = delta_ref
        self.delta_as, self.delta_ds = self.get_delta_arrays()
        self.delta_ad = np.zeros((len(self.delta_as), len(self.delta_ds)))
        self.solver_verbose = solver_verbose
        self.solver_method = solver_method

    def get_delta_arrays(self):
        """
        flattern component sites to two arrays of HB donors and acceptors
        :return: delta_as, delta_ds ->np.array, np.array
        """
        delta_as = []
        delta_ds = []
        for sites, deltas in zip(self.nu, self.delta):
            delta_as.extend([deltas[i] for i, nu_s in enumerate(sites) if nu_s < 0])
            delta_ds.extend([deltas[i] for i, nu_s in enumerate(sites) if nu_s > 0])
        return delta_as, delta_ds

    def delta_ad_ref(self, T: float):
        """
        compute Delta^AD as function of T from Eqn 13
        :param T: temperature (K)
        :return: Delta between any HB acceptor (A) and donor (D) sites
        """
        return self.kappa_ref * (np.exp(self.eps_ref / T) - 1.0)

    def update_delta_ad_matrix(self, temp: float):
        """
        build dense matrix of Delta^AD between all HB donor and acceptor sites at temperature T
        :param temp: temperature (K)
        :return: None, update self.delta_ad
        """
        delta_ref = self.delta_ad_ref(temp)
        for i, delta_a in enumerate(self.delta_as):
            for j, delta_d in enumerate(self.delta_ds):
                self.delta_ad[i][j] = delta_a*delta_d*delta_ref

    def compute(self, x: np.array, temp: float, info: dict = None):
        """
        compute activity coefficients from Association NRTL model
        :param x: mole fraction
        :param temp: temperature (K)
        :param info: optional, information dictionary which will hold gammaC, gammaR, gammaA, and solved xA
        :return: ln(gamma) -> np.array
        """
        if x.shape[0] != self.size:
            raise Exception(f'Size of x ({x.shape[0]}) is not consitant with system size {self.size}')

        # combinatorial term
        gammaC = flory_huggins(self.r, x, 2.0/3.0)

        # residual term
        if self.temp_dep_tau:
            tau_ij = self.tau_fun(temp)
            if tau_ij.shape != (self.size, self.size):
                raise Exception(f'Size of return 2-D array of tau_ij function {tau_ij.shape} '
                                f'is not consistent with system size {self.size}')
        else:
            tau_ij = self.tau
        gammaR = nrtl(x, tau_ij, self.alpha)

        # association term
        self.update_delta_ad_matrix(temp)   # -> self.delta_ad
        gammaA = association_gamma(x,
                                   self.r,
                                   self.nu,
                                   self.delta_ad,
                                   info,
                                   verbose=self.solver_verbose,
                                   opt_method=self.solver_method)
        if info is not None:
            info['gammaC'] = list(gammaC)
            info['gammaR'] = list(gammaR)
            info['gammaA'] = list(gammaA)

        return gammaC + gammaR + gammaA
