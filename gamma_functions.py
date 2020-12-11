import numpy as np


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
    if array.shape[0] == array.shape[1]:
        size = array.shape[0]
        if i < size and j < size:
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
    g_ij = np.exp(-alpha_ij*tau_ij)
    # sum_k(x_k*G_kj)
    sum_k_xk_gkj = np.dot(x, g_ij)
    # sum_j(x_j*tau_ji*G_ji)
    tauji_gji = tau_ij*g_ij
    sum_j_xj_tauji_gji = np.dot(x, tauji_gji)

    term1 = sum_j_xj_tauji_gji/sum_k_xk_gkj
    gamma_nrtl = np.zeros(size)
    for i in range(size):
        sumj = 0.0
        for j in range(size):
            outer = x[j]*g_ij[i, j]/sum_k_xk_gkj[j]
            inner = tau_ij[i, j] - sum_j_xj_tauji_gji[j]/sum_k_xk_gkj[j]
            sumj += outer*inner
        gamma_nrtl[i] = term1[i] + sumj

    return gamma_nrtl

