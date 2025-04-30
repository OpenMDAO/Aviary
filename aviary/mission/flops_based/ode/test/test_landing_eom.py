import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.flops_based.ode.landing_eom import (
    FlareEOM,
    FlareSumForces,
    GlideSlopeForces,
    GroundSumForces,
)
from aviary.models.N3CC.N3CC_data import detailed_landing_flare, inputs
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic


class FlareEOMTest(unittest.TestCase):
    """Test against data of detailed_landing_flare from models/N3CC/N3CC_data.py."""

    def setUp(self):
        prob = self.prob = om.Problem()

        time, _ = detailed_landing_flare.get_item('time')
        nn = len(time)
        aviary_options = inputs
        engines = [build_engine_deck(aviary_options)]
        preprocess_options(aviary_options, engine_models=engines)

        prob.model.add_subsystem(
            'landing_flare_eom',
            FlareEOM(num_nodes=nn, aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        do_validation_test(
            self.prob,
            'landing_flare_eom',
            input_validation_data=detailed_landing_flare,
            output_validation_data=detailed_landing_flare,
            input_keys=[
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
            ],
            tol=1e-2,
            atol=1e-8,
            rtol=5e-10,
        )

    def test_IO(self):
        exclude_inputs = {
            'angle_of_attack',
            'acceleration_vertical',
            'forces_vertical',
            'angle_of_attack_rate',
            'acceleration_horizontal',
            'forces_horizontal',
        }
        exclude_outputs = {
            'forces_vertical',
            'acceleration_horizontal',
            'forces_perpendicular',
            'acceleration_vertical',
            'net_alpha_rate',
            'forces_horizontal',
            'required_thrust',
        }
        assert_match_varnames(
            self.prob.model, exclude_inputs=exclude_inputs, exclude_outputs=exclude_outputs
        )


class OtherTest(unittest.TestCase):
    """
    Test against data of detailed landing glide slope forces, flare sum forces,
    and ground sum forces from models/N3CC/N3CC_data.py.
    """

    def test_GlideSlopeForces(self):
        # test on single component GlideSlopeForces

        tol = 1e-6
        aviary_options = inputs
        prob = om.Problem()

        # use data from detailed_landing_flare in models/N3CC/N3CC_data.py
        prob.model.add_subsystem(
            'glide', GlideSlopeForces(num_nodes=2, aviary_options=aviary_options), promotes=['*']
        )
        prob.model.set_input_defaults(Dynamic.Vehicle.MASS, np.array([106292, 106292]), units='lbm')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, np.array([47447.13138523, 44343.01567596]), units='N'
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.LIFT,
            np.array([482117.47027692, 568511.57097785]),
            units='N',
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.ANGLE_OF_ATTACK, np.array([5.086, 6.834]), units='deg'
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, np.array([-3.0, -2.47]), units='deg'
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['forces_perpendicular'], np.array([135087.0, 832087.6]), tol)
        assert_near_equal(prob['required_thrust'], np.array([-44751.64, -391905.6]), tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-12)

    def test_FlareSumForces(self):
        # test on single component FlareSumForces

        tol = 1e-6
        aviary_options = inputs
        prob = om.Problem()
        prob.model.add_subsystem(
            'flare', FlareSumForces(num_nodes=2, aviary_options=aviary_options), promotes=['*']
        )

        # use data from detailed_landing_flare in models/N3CC/N3CC_data.py
        prob.model.set_input_defaults(Dynamic.Vehicle.MASS, np.array([106292, 106292]), units='lbm')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, np.array([47447.13138523, 44343.01567596]), units='N'
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.LIFT,
            np.array([482117.47027692, 568511.57097785]),
            units='N',
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, np.array([4980.3, 4102]), units='N'
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.ANGLE_OF_ATTACK, np.array([5.086, 6.834]), units='deg'
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, np.array([-3.0, -2.47]), units='deg'
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['forces_horizontal'], np.array([17173.03, 15710.98]), tol)
        assert_near_equal(prob['forces_vertical'], np.array([11310.84, 97396.16]), tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-12)

    def test_GroundSumForces(self):
        # test on single component GroundSumForces

        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            'ground', GroundSumForces(num_nodes=2, friction_coefficient=0.025), promotes=['*']
        )

        # use data from detailed_landing_flare in models/N3CC/N3CC_data.py
        prob.model.set_input_defaults(Dynamic.Vehicle.MASS, np.array([106292, 106292]), units='lbm')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, np.array([47447.13138523, 44343.01567596]), units='N'
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.LIFT,
            np.array([482117.47027692, 568511.57097785]),
            units='N',
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, np.array([4980.3, 4102]), units='N'
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['forces_horizontal'], np.array([42466.83, 40241.02]), tol)
        assert_near_equal(prob['forces_vertical'], np.array([9307.098, 95701.199]), tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    # unittest.main()
    test = FlareEOMTest()
    test.setUp()
    test.test_case()
