import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal

from AssociationNRTLSAC import AssociationNRTLSACMolecule, AssociationNRTLSAC


class AssociationNRTLSACTests(unittest.TestCase):
    """Tests for AssociationNRTL.py"""

    def test_water_acetone_chloroform_ternary(self):
        water = AssociationNRTLSACMolecule(name='water',
                                           r=0.76,
                                           Y=0.492,
                                           nu_A=2,
                                           delta_A=1.0,
                                           nu_D=2,
                                           delta_D=1.0)
        acetone = AssociationNRTLSACMolecule(name='acetone',
                                             r=2.57,
                                             X=0.202,
                                             Y=0.726,
                                             nu_A=2,
                                             delta_A=0.623)
        chcl3 = AssociationNRTLSACMolecule(name='chloroform',
                                           r=2.87,
                                           X=0.269,
                                           Y=0.297,
                                           nu_D=1,
                                           delta_D=0.145)
        calc_obj = AssociationNRTLSAC((water, acetone, chcl3))
        x = np.array([0.1, 0.2, 0.7])
        t = 320.0
        info = {}
        gamma = calc_obj.compute(x, t, info)
        assert_array_almost_equal(gamma, np.array([2.82045156, -0.63987893, 0.0910138]))
        info_expected = {
            'xA': [0.5086144361924878, 0.6242600895001653, 0.39442931336565745, 0.8179158005218449],
            'lnGammaC': [-0.2549021276030393, -2.311033762408734e-06, -0.0029431203572827525],
            'lnGammaR': [0.19516633719528442, 0.05670029405867785, 0.02046078814951252],
            'lnGammaA': [2.880187349616815, -0.6965769103859772, 0.07349612777843442]
        }
        self.assertTrue(info.keys() == info_expected.keys())
        for key, values in info.items():
            values_expected = info_expected[key]
            for value, value_expected in zip(values, values_expected):
                self.assertAlmostEqual(value, value_expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
