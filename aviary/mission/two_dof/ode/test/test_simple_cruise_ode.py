import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.mission.two_dof.ode.test.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


@use_tempdirs
class CruiseODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        # Explicit options for the GASP-based simple cruise ODE.
        aviary_options = AviaryValues()
        aviary_options.set_val(
            Aircraft.Engine.DATA_FILE, get_path('models/engines/turbofan_23k_1.csv')
        )
        aviary_options.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 28690.0, units='lbf')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        aviary_options.set_val(Settings.VERBOSITY, Verbosity.QUIET)
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

    def test_cruise(self):
        # test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

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

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(self.prob[Dynamic.Mission.VELOCITY_RATE], np.array([0.0, 0.0]), tol)
        assert_near_equal(self.prob[Dynamic.Mission.DISTANCE], np.array([0.0, 923.39168758]), tol)
        assert_near_equal(self.prob['time'], np.array([0, 8280.30660691]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
            np.array([3.88463177, 4.90286726]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX],
            np.array([3.88463177, 4.90286726]),
            tol,
        )

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*USatm*', '*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
