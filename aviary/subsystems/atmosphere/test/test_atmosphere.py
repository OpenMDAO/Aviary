import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.atmosphere.atmosphere import AtmosphereComp


class USatm1976TestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        prob = om.Problem()

        prob.model.add_subsystem('atmo',AtmosphereComp(data_source='USatm1976', delta_T_Kelvin=0, num_nodes=9), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True, check=False,)
        prob.set_val('h', [-1000, 0, 10950, 11000, 11100, 20000, 32000], units='m')
        
        prob.run_model()

        prob.check_partials(method='cs')

    def test_case1(self):
        tol = 1e-5
        self.prob.run_model()

        assert_near_equal(self.prob['temp'], [294.65  288.15  216.975 216.65  216.65  216.65  216.65  221.65  228.65], tol)

        # USATM1976 test values
        # Temperatures (K): [294.65  288.15  216.975 216.65  216.65  216.65  216.65  221.65  228.65 ]
        # Pressure (Pa) [113929.1    101325.      22811.08    22632.06    22277.98    12044.57
        # 5474.889    2511.023     868.0187]
        # Density (kg/m**3) [1.346995   1.224999   0.3662468  0.3639178  0.3582242  0.1936736
        # 0.0880348  0.03946579 0.013225  ]
        # Viscosity (Pa*s) [1.82057492e-05 1.78938028e-05 1.42339868e-05 1.42161308e-05
        # 1.42161308e-05 1.42161308e-05 1.42161308e-05 1.44895749e-05
        # 1.48679326e-05]
        # Speed of Sound (m/s) [344.07756866 340.26121619 295.26229189 295.04107699 295.04107699
        # 295.04107699 295.04107699 298.42623913 303.1019573 ]



        
        assert_near_equal(self.prob[Dynamic.Atmosphere.MACH], np.ones(2), tol)
        assert_near_equal(self.prob.get_val('EAS', units='m/s'), 343.3 * np.ones(2), tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FlightConditionsTestCase2(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.EAS),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=1.05 * np.ones(2), units='kg/m**3'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, val=344 * np.ones(2), units='m/s'
        )
        self.prob.model.set_input_defaults('EAS', val=318.4821143 * np.ones(2), units='m/s')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-5
        self.prob.run_model()

        assert_near_equal(self.prob[Dynamic.Atmosphere.DYNAMIC_PRESSURE], 1297.54 * np.ones(2), tol)
        assert_near_equal(self.prob[Dynamic.Mission.VELOCITY], 1128.61 * np.ones(2), tol)
        assert_near_equal(self.prob[Dynamic.Atmosphere.MACH], np.ones(2), tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FlightConditionsTestCase3(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.MACH),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=1.05 * np.ones(2), units='kg/m**3'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, val=344 * np.ones(2), units='m/s'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.MACH, val=np.ones(2), units='unitless'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-5
        self.prob.run_model()

        assert_near_equal(self.prob[Dynamic.Atmosphere.DYNAMIC_PRESSURE], 1297.54 * np.ones(2), tol)
        assert_near_equal(self.prob[Dynamic.Mission.VELOCITY], 1128.61 * np.ones(2), tol)
        assert_near_equal(self.prob.get_val('EAS', units='m/s'), 318.4821143 * np.ones(2), tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
