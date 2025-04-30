import unittest
from copy import deepcopy

import openmdao.api as om

from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.models.N3CC.N3CC_data import (
    detailed_takeoff_climbing,
    detailed_takeoff_ground,
    inputs,
    takeoff_subsystem_options,
)
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

takeoff_subsystem_options = deepcopy(takeoff_subsystem_options)


class TakeoffODETest(unittest.TestCase):
    """Test detailed takeoff ODE."""

    def test_case_ground(self):
        prob = self._make_prob(climbing=False)

        do_validation_test(
            prob,
            'takeoff_ode_ground',
            input_validation_data=detailed_takeoff_ground,
            output_validation_data=detailed_takeoff_ground,
            input_keys=[
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE,
            ],
            tol=1e-2,
            atol=1e-9,
            rtol=1e-11,
            check_values=False,
            check_partials=True,
        )

    def test_case_climbing(self):
        prob = self._make_prob(climbing=True)

        do_validation_test(
            prob,
            'takeoff_ode_climbing',
            input_validation_data=detailed_takeoff_climbing,
            output_validation_data=detailed_takeoff_climbing,
            input_keys=[
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE,
            ],
            tol=1e-2,
            atol=1e-9,
            rtol=1e-11,
            check_values=False,
            check_partials=True,
        )

    @staticmethod
    def _make_prob(climbing):
        prob = om.Problem()

        time, _ = detailed_takeoff_climbing.get_item('time')
        nn = len(time)
        aviary_options = inputs
        engines = [build_engine_deck(aviary_options)]

        preprocess_options(aviary_options, engine_models=engines)

        default_mission_subsystems = get_default_mission_subsystems('FLOPS', engines)

        prob.model.add_subsystem(
            'takeoff_ode',
            TakeoffODE(
                num_nodes=nn,
                aviary_options=aviary_options,
                subsystem_options=takeoff_subsystem_options,
                core_subsystems=default_mission_subsystems,
                climbing=climbing,
                friction_key=Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1.0, units='ft**2')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        set_aviary_initial_values(prob, aviary_options)

        return prob


if __name__ == '__main__':
    unittest.main()
