import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.mission.two_dof.ode.test.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class CruiseODETestCase(unittest.TestCase):
    """Test SimpleCruiseODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        subsystem_options = {'aerodynamics': {'method': 'cruise', 'output_alpha': True}}

        self.prob.model = SimpleCruiseODE(
            num_nodes=2,
            aviary_options=aviary_options,
            subsystems=default_mission_subsystems,
            subsystem_options=subsystem_options,
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.MACH, np.array([0, 0]), units='unitless'
        )

        setup_model_options(self.prob, aviary_options)

    def _set_inputs(self):
        self.prob.set_val(Dynamic.Atmosphere.MACH, [0.7, 0.7], units='unitless')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)
        self.prob.set_val('time', np.array([0, 8280.30660691]), units='s')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, val=37500 * np.ones(2), units='ft')
        self.prob.set_val('mass', val=np.linspace(171481, 171581 - 10000, 2), units='lbm')

        set_params_for_unit_tests(self.prob)

    def test_cruise(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        self._set_inputs()

        self.prob.run_model()

        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: ([0.0, 0.0], 'ft/s**2'),
            Dynamic.Mission.DISTANCE: ([0.0, 923.39168758], 'NM'),
            'time': ([0.0, 8280.30660691], 's'),
            Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS: ([3.88463177, 4.90286726], 'm/s'),
            Dynamic.Mission.ALTITUDE_RATE_MAX: ([3.88463177, 4.90286726], 'm/s'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=1e-6)

    def test_partials(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        self._set_inputs()

        self.prob.run_model()

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*USatm*', '*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
