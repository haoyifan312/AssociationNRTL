import numpy as np
from math import log10
from scipy.optimize import least_squares


def flory_huggins(r: np.array, x: np.array, power_factor: float = 1.0) -> np.array:
    """
    calculate combinatorial activity coefficient from Flory-Huggins model due to size differences
    :param r: normalized size parameters
    :param x: mole fraction, sanity check of fraction number need to be prior to calling this function
    :param power_factor: power parameter, default 1.0 for original Flory-Huggins expression
    :return: ln(gamma)
    """
    r_i = np.power(r, power_factor) if power_factor != 1.0 else r
    # phiI/xI to avoid divide by 0
    phi_i_by_x_i = r_i / np.sum(x * r_i)
    return 1.0 - phi_i_by_x_i + np.log(phi_i_by_x_i)


def set_symmetrical_matrix(array: np.array, i: int, j: int, value):
    array[i, j] = value
    array[j, i] = value


def nrtl(x: np.array, tau_ij: np.array, alpha_ij: np.array) -> np.array:
    """
    calculate activity coefficients from NRTL
    :param x: numpy 1D array, mole fraction, sanity check of fraction number need to be prior to calling this function
    :param tau_ij: numpy 2D array, NRTL binary interaction parameters
    :param alpha_ij: numpy 2D array, NRTL binary nonrandomness factors
    :return: ln(gamma)
    """
    size = x.size
    if not size > 0:
        raise Exception("NRTL input size is 0")
    # compute Gij
    g_ij = np.exp(-alpha_ij * tau_ij)
    # sum_k(x_k*G_kj)
    sum_k_xk_gkj = np.dot(x, g_ij)
    # sum_j(x_j*tau_ji*G_ji)
    tauji_gji = tau_ij * g_ij
    sum_j_xj_tauji_gji = np.dot(x, tauji_gji)

    term1 = sum_j_xj_tauji_gji / sum_k_xk_gkj
    gamma_nrtl = np.zeros(size)
    for i in range(size):
        sumj = 0.0
        for j in range(size):
            outer = x[j] * g_ij[i, j] / sum_k_xk_gkj[j]
            inner = tau_ij[i, j] - sum_j_xj_tauji_gji[j] / sum_k_xk_gkj[j]
            sumj += outer * inner
        gamma_nrtl[i] = term1[i] + sumj

    return gamma_nrtl


def association_unbonded_site_fraction_residuals(xs: np.array, rho_a, rho_d, cap_delta_ad: np.ndarray):
    """
    compute function value of solving unbonded site fractions
    :param xs: current unbonded site fractions
    :param rho_a:
    :param rho_d:
    :param cap_delta_ad:
    :return:
    """
    size_a = rho_a.shape[0]
    size_d = rho_d.shape[0]
    ret = np.zeros(size_a + size_d)
    for i in range(size_a):
        sum_term = 0.0
        for j in range(size_d):
            sum_term += rho_d[j] * cap_delta_ad[i][j] * xs[size_a + j]
        ret[i] = log10(xs[i]) + log10(1.0 + sum_term)

    for i in range(size_d):
        sum_term = 0.0
        for j in range(size_a):
            sum_term += rho_a[j] * cap_delta_ad[j][i] * xs[j]
        ret[size_a + i] = log10(xs[size_a + i]) + log10(1.0 + sum_term)

    return ret


def association_gamma(x_i: np.array, r_i: np.array, nu_i: list, cap_delta_ad: np.ndarray, info_dict: dict = None, verbose=0,
                      opt_method='dogbox'):
    """
    calculate activity coefficient from association theory, reference Hao and Chen, AIChE J. 2020
    :param opt_method: scipy least square method name, default 'dogbox', refer to scipy.optimize.least_square
    :param verbose: option to print out scipy optimization details, refer to scipy.optimize.least_square
    :param info_dict: dictionary, solved unbonded site fraction X^A will be stored here
    :param x_i: mole fraction of component i, sanity check of fraction need to be outside
    :param r_i: normalized volume parameters of components
    :param nu_i: number of site type A or D of each component, e.g. [[], [2,-2], [1], [-1,-1]] positive - HB donor,
    negative - HB acceptor
    :param cap_delta_ad: Delta^AD between all different A and D sites pre-computed
    :return: ln(gamma) of association contribution
    """
    all_a = []
    all_d = []
    for i, sites in enumerate(nu_i):
        for site in sites:
            if site > 0:
                all_d.append((i, site))
            elif site < 0:
                all_a.append((i, -site))

    if cap_delta_ad.shape != (len(all_a), len(all_d)):
        raise Exception(f"Input cap_delta_ad shape {cap_delta_ad.shape} "
                        f"is not consistent with component sites {(len(all_a), len(all_d))}")

    # compute unbonded site fraction in mixture X^A
    sum_ri_xi = sum(x_i * r_i)  # denominator of Eqn 10
    # assume no same sites in different species
    rho_a = np.zeros(len(all_a))
    rho_d = np.zeros(len(all_d))
    # Eqn 10
    for i, pa in enumerate(all_a):
        comp_id, nu = pa
        rho_a[i] = nu * x_i[comp_id] / sum_ri_xi
    for i, pa in enumerate(all_d):
        comp_id, nu = pa
        rho_d[i] = nu * x_i[comp_id] / sum_ri_xi
    xA = np.ones((len(rho_a) + len(rho_d)))
    # solve Eqn 8
    result = least_squares(association_unbonded_site_fraction_residuals,
                           xA,
                           args=(rho_a, rho_d, cap_delta_ad),
                           bounds=(1.0e-10, 1.0),
                           method=opt_method,
                           verbose=verbose)

    if not result['success']:
        print('X^A did not solve')
        return None

    xA = result['x']  # X^A
    rho_all_sites = np.concatenate((rho_a, rho_d), axis=0)
    sec_sum = sum(rho_all_sites * (1.0 - xA) / 2.0)  # second summation in Eqn 7
    lnGammaA = np.full((len(x_i)), sec_sum)
    lnGammaA *= r_i  # second term in Eqn 7

    # compute pure component unbonded site fractions X_i^A and first summation term in lnGammaA
    id_a = 0
    id_d = 0
    for i, sites in enumerate(nu_i):
        if len(sites) > 0:
            xIA = np.ones((len(sites)))
            # Eqn 11
            rho_a_i = np.array([-nu / r_i[i] for nu in sites if nu < 0])
            rho_d_i = np.array([nu / r_i[i] for nu in sites if nu > 0])
            size_a = rho_a_i.shape[0]
            size_d = rho_d_i.shape[0]
            if max(sites) > 0 > min(sites):
                # only compute if component has both A and D sites, otherwise all 1
                delta = np.zeros((size_a, size_d))
                for a_index in range(size_a):
                    for d_index in range(size_d):
                        delta[a_index][d_index] = cap_delta_ad[id_a + a_index][id_d + d_index]
                # solve Eqn 9
                result = least_squares(association_unbonded_site_fraction_residuals,
                                       xIA,
                                       args=(rho_a_i, rho_d_i, delta),
                                       bounds=(1.0e-10, 1.0),
                                       method=opt_method,
                                       verbose=verbose)
                if result['success']:
                    xIA = result['x']
                else:
                    print('X_i^A did not solve')
                    return None

            # X^A unbonded site fraction of sites to component I
            xA_of_I = np.concatenate((xA[id_a:id_a + size_a], xA[len(all_a) + id_d:len(all_a) + id_d + size_d]), axis=0)

            nu_A_J = [-site for site in sites if site < 0.0]  # acceptor sites
            nu_A_J.extend([site for site in sites if site > 0.0])   # append donor sites to end
            nu_A_J = np.array(nu_A_J)
            lnGammaA[i] += sum(nu_A_J * (np.log(xA_of_I / xIA) + (xIA - 1.0) / 2.0))
            id_a += size_a
            id_d += size_d

    if info_dict is not None:
        info_dict['xA'] = list(xA)

    return lnGammaA


# methanol n-hexane
xi = np.array([0.1, 0.9])
ri = np.array([1.43, 4.75])
nui = [[1, -2],
       []]
delta_ad = np.zeros((1, 1))
delta_ad[0][0] = 29.15458358
info_dict = {}
print(association_gamma(xi, ri, nui, delta_ad, verbose=2, info_dict=info_dict))
print(info_dict)
