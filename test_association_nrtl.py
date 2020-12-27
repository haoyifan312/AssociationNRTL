import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from AssociationNRTL import AssociationNRTL


class AssociationNRTLTests(unittest.TestCase):
    """Tests for AssociationNRTL.py"""

    def test_methanol_hexane_binary(self):
        # 1-methanol 2-hexane
        rs = np.array([1.43, 5.45])
        tau = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        tau[0][1] = -1.45  # methanol-hexane
        tau[1][0] = 2.188  # hexane-methanol
        alpha[0][1] = alpha[1][0] = 0.2
        nu = [[1, -2], []]
        delta = [[1.0, 1.0], []]
        MethanolHexane = AssociationNRTL(rs, tau, alpha, nu, delta)
        info = {}
        xs = np.array([0.1, 0.9])
        gamma = MethanolHexane.compute(xs, 290.0, info)
        assert_array_almost_equal(gamma, np.array([2.15093317, 0.05506928]))
        info_expected = {'lnGammaA': [2.2941267426117498, 0.05074053854893874],
                         'lnGammaC': [-0.26668843103865236, -0.0018881073985488744],
                         'lnGammaR': [0.1234948602480419, 0.006216847427475997],
                         'xA': [0.7650107902819817, 0.5300215805551459]}
        self.assertTrue(info.keys() == info_expected.keys())
        for key, values in info.items():
            values_expected = info_expected[key]
            for value, value_expected in zip(values, values_expected):
                self.assertAlmostEqual(value, value_expected)

    def test_water_egbe_binary(self):
        # 1-water 2-EGBE
        rs = np.array([0.76, 5.05])
        tau = np.zeros((2, 2))
        alpha = np.zeros((2, 2))
        tau[0][1] = -2.07880482  # water-EGBE
        tau[1][0] = 2.65715504  # EGBE-water
        alpha[0][1] = alpha[1][0] = 0.3
        nu = [[2, -2], [1, -2, -2]]
        delta = [[1.0, 1.0], [1.0, 1.0, 0.26]]
        WaterEGBE = AssociationNRTL(rs, tau, alpha, nu, delta)
        info = {}
        xs = np.array([0.8, 0.2])
        gamma = WaterEGBE.compute(xs, 320.0, info)
        assert_array_almost_equal(gamma, np.array([0.24498518, 0.37599375]))
        info_expected = {'xA': [0.2984689816415618, 0.2984689816416465, 0.6206891985361339, 0.13622980149879493, 0.13622980149879535],
                         'lnGammaC': [-0.07366516449835842, -0.4930109821121367],
                         'lnGammaR': [-0.0269148055981549, -0.6493863357396057],
                         'lnGammaA': [0.3455651486138014, 1.5183910727597576]}
        self.assertTrue(info.keys() == info_expected.keys())
        for key, values in info.items():
            values_expected = info_expected[key]
            for value, value_expected in zip(values, values_expected):
                self.assertAlmostEqual(value, value_expected)

    def test_water_octane_cresol_ternary(self):
        # 1-water 2-octane 3-cresol
        rs = np.array([0.759999999,	5.848379999,	4.28675])
        tau = np.zeros((3, 3))
        alpha = np.zeros((3, 3))
        tau[0][1] = 4.72050125  # water-octane
        tau[1][0] = -1.12867914  # octane-water
        tau[0][2] = -0.188289766  # water-cresol
        tau[2][0] = -2.17789106  # cresol-water
        tau[1][2] = 2.08049143  # octane-cresol
        tau[2][1] = -0.0657422022  # cresol-octane
        alpha[0][1] = alpha[1][0] = 0.2
        alpha[0][2] = alpha[2][0] = 0.2
        alpha[1][2] = alpha[2][1] = 0.2
        nu = [[2, -2], [], [1]]
        delta = [[1.0, 1.0], [], [5.234]]
        WaterEGBE = AssociationNRTL(rs, tau, alpha, nu, delta)
        info = {}
        xs = np.array([0.2, 0.3, 0.5])
        gamma = WaterEGBE.compute(xs, 300.0, info)
        assert_array_almost_equal(gamma, np.array([0.33712236,  1.41257177, -0.55279752]))
        info_expected = {'xA': [0.10525594794963455, 0.8046881234758262, 0.44045425967908014],
                         'lnGammaC': [-0.4216079737580175, -0.04228011500818901, -0.0025274745776799473],
                         'lnGammaR': [-1.1016244223969927, 0.9380176499248769, -0.10915092053099013],
                         'lnGammaA': [1.8603547551628437, 0.5168342360510096, -0.44111912659305846]}
        self.assertTrue(info.keys() == info_expected.keys())
        for key, values in info.items():
            values_expected = info_expected[key]
            for value, value_expected in zip(values, values_expected):
                self.assertAlmostEqual(value, value_expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
