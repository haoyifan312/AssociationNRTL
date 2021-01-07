import matplotlib.pyplot as plt
import numpy as np

from AssociationNRTLSAC import AssociationNRTLSAC, AssociationNRTLSACMolecule
import database as db

# This file illustrates the functionality of AssociationNRTLSAC class
# reference Hao and Chen, I&ECR 2019

# Use chloroform-water binary case as an example
# molecular parameters for each component are wrapped in AssociationNRTLSACMolecule object
# molecular parameters can be added when initializing the object
chloroform = AssociationNRTLSACMolecule(name='chloroform',
                                        r=2.87,
                                        X=0.269,
                                        Y=0.297,
                                        nu_D=1,
                                        delta_D=0.145)
# molecular parameters can also be added from the attributes
water = AssociationNRTLSACMolecule(name='water')
water.r = 0.76
water.Y = 0.492
water.nu_D = water.nu_A = 2
water.delta_D = water.delta_A = 1.0

# AssociationNRTLSAC class is constructed by wrapping all AssociationNRTLSACMolecule objects in the "molecules" argument
ChloroformWaterBinary = AssociationNRTLSAC(molecules=(chloroform, water))

# activity coefficient is calculated by compute function by pass in temperature and mole fraction
x = np.array([0.2, 0.8])
T = 300.0   # K
lnGamma = ChloroformWaterBinary.compute(x, T)
print(f'Calculated ln(gamma):\n{lnGamma}')

# additional information of calculated values can be retrieved by passing in an empty dictionary
info = {}
lnGamma = ChloroformWaterBinary.compute(x, T, info)
print(f'Additional calculated values:\n{info}')

# molecular parameters published in Table 4 of the reference are stored in a local database
db.show_all_data()
# molecular parameters can be retrieved from the component name (case insensative)
# and return as AssociationNRTLSACMolecule object
acetone = db.get_molecular_data('acetone')
print(acetone)
water_acetone_chloroform_ternary = AssociationNRTLSAC(molecules=(water, acetone, chloroform))
x = np.array([0.1, 0.2, 0.7])
T = 320.0
lnGamma_ternary = water_acetone_chloroform_ternary.compute(x, T)
print(f'Calculated ln(gamma) of water-acetone-chloroform ternary systems are {lnGamma_ternary}')

# user could modify their own copy of the database
# new molecular parameters could be added by AssociationNRTLSACMolecule object
test_molecule = AssociationNRTLSACMolecule(name='test molecule',
                                           r=1.23,
                                           X=0.5,
                                           nu_A=2,
                                           delta_A=1.3)
db.add_to_db(test_molecule)
retrieved_test_molecule = db.get_molecular_data('test molecule')
print(retrieved_test_molecule)
# existing molecular parameters could be modified
test_molecule.r = 3.14
db.update_molecule_in_db(test_molecule)
print(db.get_molecular_data('test molecule'))
# existing molecular parameters could be deleted
db.delete_molecule_from_db('test molecule')

