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
