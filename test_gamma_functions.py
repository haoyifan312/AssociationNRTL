import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from gamma_functions import flory_huggins, set_symmetrical_matrix, nrtl, association_gamma


class GammaFunctionTests(unittest.TestCase):
    """Tests for gamma_functions.py"""

    def test_flory_huggins(self):
        # binary power=1
        binary_r = np.array([2.0, 3.0])
        binary_x = np.array([0.2, 0.8])
        assert_array_almost_equal(flory_huggins(binary_r, binary_x), np.array([-0.05075795, -0.0024357]))
        # binary power=2/3
        assert_array_almost_equal(flory_huggins(binary_r, binary_x, 2.0/3.0), np.array([-0.02287156, -0.00119687]))
        # ternary power = 1
        ternary_r = np.array([2., 3., 4.])
        ternary_x = np.array([.1, .2, .7])
        assert_array_almost_equal(flory_huggins(ternary_r, ternary_x), np.array([-0.14334222, -0.01565489, -0.0057506]))
        # ternary power = 2/3
        assert_array_almost_equal(flory_huggins(ternary_r, ternary_x, 2./3.), np.array([-0.066243, -0.006603, -0.002855]))

    def test_nrtl(self):
        # binary
        tau = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        tau[0, 1] = 2.
        tau[1, 0] = 4.
        set_symmetrical_matrix(alpha, 0, 1, 0.3)
        x = np.array([.1, .9])
        assert_array_almost_equal(nrtl(x, tau, alpha), np.array([3.10967304,  0.09410169]))
        # ternary
        tau3 = np.zeros((3, 3))
        alpha3 = np.zeros((3, 3))
        tau3[0, 1] = 2.
        tau3[1, 0] = 4.
        tau3[0, 2] = -2.
        tau3[2, 0] = 3.
        tau3[1, 2] = 5.
        tau3[2, 1] = -3.
        set_symmetrical_matrix(alpha3, 0, 1, 0.3)
        set_symmetrical_matrix(alpha3, 0, 2, 0.3)
        set_symmetrical_matrix(alpha3, 1, 2, 0.3)
        x3 = np.array([.1, .2, .7])
        assert_array_almost_equal(nrtl(x3, tau3, alpha3), np.array([-0.38167668, -1.32633701, -0.09814396]))

    def test_association_gamma(self):
        # methanol n-hexane
        xi = np.array([0.1, 0.9])
        ri = np.array([1.43, 4.75])
        nui = [[1, -2],
               []]
        delta_ad = np.zeros((1, 1))
        delta_ad[0][0] = 29.15458358
        info_dict = {}
        lnGammaA = association_gamma(xi, ri, nui, delta_ad, info_dict=info_dict)
        assert_array_almost_equal(lnGammaA, np.array([2.2029566,  0.05352245]))
        xA_calculated = info_dict['xA']
        xA_save = [0.7510924235549102, 0.5021848466186324]
        for calc, save in zip(xA_calculated, xA_save):
            self.assertAlmostEqual(calc, save)


if __name__ == "__main__":
    unittest.main(verbosity=2)