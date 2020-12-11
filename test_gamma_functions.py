import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from gamma_functions import flory_huggins


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





if __name__ == "__main__":
    unittest.main(verbosity=2)