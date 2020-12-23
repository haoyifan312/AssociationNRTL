import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from AssociationNRTl import AssociationNRTL
import numpy as np


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
        info_expected = {'gammaA': [2.2941267426117498, 0.05074053854893874],
                         'gammaC': [-0.26668843103865236, -0.0018881073985488744],
                         'gammaR': [0.1234948602480419, 0.006216847427475997],
                         'xA': [0.7650107902819817, 0.5300215805551459]}
        self.assertTrue(info.keys() == info_expected.keys())
        for key, values in info.items():
            values_expected = info_expected[key]
            for value, value_expected in zip(values, values_expected):
                self.assertAlmostEqual(value, value_expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
