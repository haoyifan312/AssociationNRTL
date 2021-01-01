import numpy as np
from dataclasses import dataclass

from gamma_functions import nrtl
from AssociationNRTL import AssociationNRTL


@dataclass
class AssociationNRTLSACMolecule:
    """
    reference Hao and Chen, I&ECR 2019
    name: molecule name
    r: volume parameter, as in Eqn 6, 18, 19, Table 4
    X and Y: conceptual segment numbers of nonpolar and polar segments,
    as in Table 4, Eqn 7, 10, 11
    nu_D, nu_A: HB donor and acceptor site numbers, as A and D in Table 4,
    and in Eqn 15, 18, 19
    delta_D, delta_A: association strength parameter for HB donor and
    acceptor sites, as in Table 4, Eqn 24
    nu_additional_sites and delta_additional_sites: optional,
    by default each molecule can have at most 1 type of HB donor site and
    1 type of HB acceptor site. If additional types of association sites
    are needed, they can be specified in the same format as nu_i and delta_i
    as in the AssociationNRTL class
    """

    name: str = ''
    r: float = 0.0
    X: float = 0.0
    Y: float = 0.0
    nu_D: int = 0
    delta_D: float = 0.0
    nu_A: int = 0
    delta_A: float = 0.0
    nu_additional_sites: list = None
    delta_additional_sites: list = None


class AssociationNRTLSAC(AssociationNRTL):
    # interaction between segment X and Y
    tau_seg = np.zeros((2, 2))
    alpha_seg = np.zeros((2, 2))
    tau_seg[0][1] = 1.643
    tau_seg[1][0] = 1.834
    alpha_seg[0][1] = alpha_seg[1][0] = 0.2

    def __init__(self, molecules: list,
                 solver_verbose: int = 0,
                 solver_method: str = 'dogbox'):
        """
        Association NRTL-SAC model, reference Hao and Chen, I&ECR 2019
        :param molecules:
        :param solver_verbose: optional, default 0,
        verbose option to print solver info,
        refer to numpy.optimize.least_square verbose argument
        :param solver_method: optional, default 'dogbox', solver method to
        solve unbonded site fraction xA,
        refer to numpy.optimize.least_square method argument
        """
        size = len(molecules)
        rs = [molecule.r for molecule in molecules]
        for i, r in enumerate(rs):
            if r <= 0.0:
                raise Exception(f'molecule {molecules[i].name} has incorrect r of {r}')

        self.Xs = [molecule.X for molecule in molecules]
        self.Ys = [molecule.Y for molecule in molecules]

        nu_i = []
        delta_i = []
        for m in molecules:
            nu = []
            delta = []
            if m.nu_D > 0 and m.delta_D > 0:
                nu.append(m.nu_D)
                delta.append(m.delta_D)
            if m.nu_A > 0 and m.delta_A > 0:
                nu.append(-m.nu_A)
                delta.append(m.delta_A)
            if m.nu_additional_sites is not None and m.delta_additional_sites is not None:
                nu.extend(m.nu_additional_sites)
                delta.extend(m.delta_additional_sites)
            if len(nu) != len(delta):
                raise Exception(f'nu and delta are not same size for molecule {m.name}.')
            nu_i.append(nu)
            delta_i.append(delta)

        super().__init__(np.array(rs),
                         np.zeros((size, size)),
                         np.zeros((size, size)),
                         nu_i,
                         delta_i,
                         solver_verbose,
                         solver_method)

    def gamma_nrtl_segment(self, x: np.array):
        """
        compute activity coefficients of segments using NRTL as Eqn 8 and 9
        :param x: segment mole fraction
        :return: lnGamma
        """
        return nrtl(x, self.tau_seg, self.alpha_seg)

    def compute(self, x: np.array, temp: float, info: dict = None):
        """
        compute activity coefficients from Association NRTL-SAC model
        :param x: mole fraction
        :param temp: temperature (K)
        :param info: optional, information dictionary which will hold gammaC, gammaR, gammaA, and solved xA
        :return: ln(gamma) -> np.array
        """
        if x.shape[0] != self.size:
            raise Exception(f'Size of x ({x.shape[0]}) is not consitant with system size {self.size}')

        # combinatorial term
        gammaC = self.modified_flory_huggins(x)

        # association term
        gammaA = self.association_term(temp, x, info)

        # residual term
        sum_sum_riJ_xJ = total_x = total_y = 0.0    # denominator of Eqn 10
        for xi, seg_x, seg_y in zip(x, self.Xs, self.Ys):
            sum_sum_riJ_xJ += xi*(seg_x + seg_y)
            total_x += xi*seg_x
            total_y += xi*seg_y
        x_seg = np.array((total_x, total_y))/sum_sum_riJ_xJ     # x_j in Eqn 10
        # compute lnGamma^lc_m as in Eqn 8
        gamma_m = self.gamma_nrtl_segment(x_seg)

        # compute lnGamma^lc,I_m as in Eqn 9 and total gammaR from Eqn 7
        gammaR = np.zeros((self.size,))
        for i, (seg_x, seg_y) in enumerate(zip(self.Xs, self.Ys)):
            gamma_in_comp = np.zeros((2, ))
            if abs(seg_x*seg_y) > 0.0:      # segment activities will be one if not both segments exist
                seg_total = seg_x + seg_y
                x_x = seg_x/seg_total
                x_y = seg_y/seg_total
                gamma_in_comp = self.gamma_nrtl_segment(np.array((x_x, x_y)))
            gammaR[i] = seg_x*(gamma_m[0] - gamma_in_comp[0]) + seg_y*(gamma_m[1] - gamma_in_comp[1])

        if info is not None:
            info['lnGammaC'] = list(gammaC)
            info['lnGammaR'] = list(gammaR)
            info['lnGammaA'] = list(gammaA)

        return gammaC + gammaR + gammaA


def get_molecule_from_database(molecule, **kwargs):
    name = kwargs.get('name', None)
    molecule = AssociationNRTLSACMolecule(name=name)
    return True

