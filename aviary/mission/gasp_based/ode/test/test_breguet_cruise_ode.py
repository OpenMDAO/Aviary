import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.breguet_cruise_ode import (
    BreguetCruiseODESolution,
    E_BreguetCruiseODESolution,
)
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class CruiseODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.prob.model = BreguetCruiseODESolution(
            num_nodes=2,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
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

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(self.prob[Dynamic.Mission.VELOCITY_RATE], np.array([1.0, 1.0]), tol)
        assert_near_equal(self.prob[Dynamic.Mission.DISTANCE], np.array([0.0, 882.58196388]), tol)
        assert_near_equal(self.prob['time'], np.array([0, 7913.75811534]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
            np.array([3.43157488, 4.43286636]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX],
            np.array([-17.63008441, -16.62879293]),
            tol,
        )

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*USatm*', '*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class ElectricCruiseODETestCase(unittest.TestCase):
    """This test uses a makeup electrical engine to test electrical Breguet cruise ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        aviary_options.set_val(
            Aircraft.Engine.DATA_FILE,
            'mission/gasp_based/ode/test/test_data/turbofan_23k_electrified.deck',
        )
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options)
        )

        self.prob.model = E_BreguetCruiseODESolution(
            num_nodes=2,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.MACH, np.array([0, 0]), units='unitless'
        )

        setup_model_options(self.prob, aviary_options)

    def test_electric_cruise(self):
        # test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Atmosphere.MACH, [0.7, 0.7], units='unitless')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(self.prob[Dynamic.Mission.VELOCITY_RATE], np.array([1.0, 1.0]), tol)
        assert_near_equal(self.prob[Dynamic.Mission.DISTANCE], np.array([0.0, 66.66754421]), tol)
        assert_near_equal(self.prob['time'], np.array([0, 597.78110206]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
            np.array([3.439203, 4.440962]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX],
            np.array([-17.622456, -16.62070]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL],
            np.array([4.67455501, 4.33784647]),
            tol,
        )

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*USatm*', '*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
