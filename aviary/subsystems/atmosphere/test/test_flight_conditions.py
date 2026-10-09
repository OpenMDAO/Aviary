import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class FlightConditionsTest(unittest.TestCase):
    def test_case_tas(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.TAS),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=1.22 * np.ones(2), units='kg/m**3'
        )
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, val=344 * np.ones(2), units='m/s'
        )
        prob.model.set_input_defaults(Dynamic.Mission.VELOCITY, val=344 * np.ones(2), units='m/s')

        prob.setup(check=False, force_alloc_complex=True)

        tol = 1e-5
        prob.run_model()

        expected_values = {
            Dynamic.Atmosphere.DYNAMIC_PRESSURE: (1507.6 * np.ones(2), 'lbf/ft**2'),
            Dynamic.Atmosphere.MACH: (np.ones(2), 'unitless'),
            'EAS': (343.3 * np.ones(2), 'm/s'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_case_eas(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.EAS),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=1.05 * np.ones(2), units='kg/m**3'
        )
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, val=344 * np.ones(2), units='m/s'
        )
        prob.model.set_input_defaults('EAS', val=318.4821143 * np.ones(2), units='m/s')

        prob.setup(check=False, force_alloc_complex=True)

        tol = 1e-5
        prob.run_model()

        expected_values = {
            Dynamic.Atmosphere.DYNAMIC_PRESSURE: (1297.54 * np.ones(2), 'lbf/ft**2'),
            Dynamic.Mission.VELOCITY: (1128.61 * np.ones(2), 'ft/s'),
            Dynamic.Atmosphere.MACH: (np.ones(2), 'unitless'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_case_mach(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.MACH),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=1.05 * np.ones(2), units='kg/m**3'
        )
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, val=344 * np.ones(2), units='m/s'
        )
        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, val=np.ones(2), units='unitless')

        prob.setup(check=False, force_alloc_complex=True)

        tol = 1e-5
        prob.run_model()

        expected_values = {
            Dynamic.Atmosphere.DYNAMIC_PRESSURE: (1297.54 * np.ones(2), 'lbf/ft**2'),
            Dynamic.Mission.VELOCITY: (1128.61 * np.ones(2), 'ft/s'),
            'EAS': (318.4821143 * np.ones(2), 'm/s'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
