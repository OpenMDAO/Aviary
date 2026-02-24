import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.examples.external_subsystems.OAS_mass.OAS_wing_mass_analysis import OAStructures


class Test_OAStructures(unittest.TestCase):
    """Test OAS wing mass component."""

    @use_tempdirs
    def test_OAS_wing_mass_analysis(self):
        # run program
        prob = om.Problem()

        # mesh example
        prob.model.add_subsystem(
            'OAS',
            OAStructures(
                symmetry=True,
                wing_weight_ratio=1.0,
                S_ref_type='projected',
                n_point_masses=1,
                num_twist_cp=4,
                num_box_cp=51,
            ),
        )

        prob.setup()

        # test data taken from the OpenAeroStruct example aeroelastic wingbox example
        # and the aircraft_for_bench_FwFm.csv benchmark data file.  All length units are in meters
        # and all mass units are in kilograms for this test data.
        # fmt: off
        prob['OAS.box_upper_x'] = np.array(
            [
                0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23,
                0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
                0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51,
                0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
            ]
        )
        prob['OAS.box_lower_x'] = np.array(
            [
                0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23,
                0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
                0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51,
                0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
            ]
        )
        prob['OAS.box_upper_y'] = np.array(
            [
                0.0447, 0.046, 0.0472, 0.0484, 0.0495, 0.0505, 0.0514, 0.0523, 0.0531, 0.0538,
                0.0545, 0.0551, 0.0557, 0.0563, 0.0568, 0.0573, 0.0577, 0.0581, 0.0585, 0.0588,
                0.0591, 0.0593, 0.0595, 0.0597, 0.0599, 0.06, 0.0601, 0.0602, 0.0602, 0.0602,
                0.0602, 0.0602, 0.0601, 0.06, 0.0599, 0.0598, 0.0596, 0.0594, 0.0592, 0.0589,
                0.0586, 0.0583, 0.058, 0.0576, 0.0572, 0.0568, 0.0563, 0.0558, 0.0553, 0.0547,
                0.0541,
            ]
        )
        prob['OAS.box_lower_y'] = np.array(
            [
                -0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532,
                -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583,
                -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598,
                -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582,
                -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519,
                -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444,
            ]
        )
        # fmt: on

        prob['OAS.twist_cp'] = np.array([-6.0, -6.0, -4.0, 0.0])
        prob['OAS.spar_thickness_cp'] = np.array([0.004, 0.005, 0.008, 0.01])
        prob['OAS.skin_thickness_cp'] = np.array([0.005, 0.01, 0.015, 0.025])
        prob['OAS.t_over_c_cp'] = np.array([0.08, 0.08, 0.10, 0.08])
        prob['OAS.airfoil_t_over_c'] = 0.13
        prob['OAS.fuel'] = 18163.652864
        prob['OAS.fuel_reserve'] = 1360.77711
        prob['OAS.CD0'] = 0.0078
        prob['OAS.cruise_Mach'] = 0.785
        prob['OAS.cruise_altitude'] = 11303.682962301647
        prob['OAS.cruise_range'] = 6482000.0
        prob['OAS.cruise_SFC'] = 0.53 / 3600
        prob['OAS.engine_mass'] = 3356.583538
        prob['OAS.engine_location'] = np.array([4.825, -1.0, 0.0])

        prob.run_model()

        print('wing mass = ', prob.model.get_val('OAS.wing_mass', units='lbm'))
        print('fuel burn = ', prob.model.get_val('OAS.fuel_burn', units='lbm'))


if __name__ == '__main__':
    unittest.main()
