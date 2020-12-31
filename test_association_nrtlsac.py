import numpy as np
import unittest

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
        print(gamma)
        print(info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
