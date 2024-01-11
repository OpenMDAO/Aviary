import unittest
from copy import deepcopy

import openmdao.api as om

from aviary.interface.default_phase_info.height_energy import default_mission_subsystems
from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.models.N3CC.N3CC_data import (
    detailed_takeoff_climbing, detailed_takeoff_ground, takeoff_subsystem_options, inputs)
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic as _Dynamic
from aviary.variable_info.variables import Mission

Dynamic = _Dynamic.Mission

takeoff_subsystem_options = deepcopy(takeoff_subsystem_options)


class TakeoffODETest(unittest.TestCase):
    def test_case_ground(self):
        prob = self._make_prob(climbing=False)

        do_validation_test(
            prob,
            'takeoff_ode_ground',
            input_validation_data=detailed_takeoff_ground,
            output_validation_data=detailed_takeoff_ground,
            input_keys=[
                'angle_of_attack',
                Dynamic.FLIGHT_PATH_ANGLE,
                Dynamic.ALTITUDE,
                Dynamic.VELOCITY,
                Dynamic.MASS,
                Dynamic.LIFT,
                Dynamic.THRUST_TOTAL,
                Dynamic.DRAG],
            output_keys=[
                Dynamic.RANGE_RATE,
                Dynamic.ALTITUDE_RATE,
                Dynamic.VELOCITY_RATE],
            tol=1e-2, atol=1e-9, rtol=1e-11,
            check_values=False, check_partials=True)

    def test_case_climbing(self):
        prob = self._make_prob(climbing=True)

        do_validation_test(
            prob,
            'takeoff_ode_climbing',
            input_validation_data=detailed_takeoff_climbing,
            output_validation_data=detailed_takeoff_climbing,
            input_keys=[
                'angle_of_attack',
                Dynamic.FLIGHT_PATH_ANGLE,
                Dynamic.ALTITUDE,
                Dynamic.VELOCITY,
                Dynamic.MASS,
                Dynamic.LIFT,
                Dynamic.THRUST_TOTAL,
                Dynamic.DRAG],
            output_keys=[
                Dynamic.RANGE_RATE,
                Dynamic.ALTITUDE_RATE,
                Dynamic.VELOCITY_RATE],
            tol=1e-2, atol=1e-9, rtol=1e-11,
            check_values=False, check_partials=True)

    @staticmethod
    def _make_prob(climbing):
        prob = om.Problem()

        time, _ = detailed_takeoff_climbing.get_item('time')
        nn = len(time)
        aviary_options = inputs

        prob.model.add_subsystem(
            "takeoff_ode",
            TakeoffODE(
                num_nodes=nn,
                aviary_options=aviary_options,
                subsystem_options=takeoff_subsystem_options,
                core_subsystems=default_mission_subsystems,
                climbing=climbing,
                friction_key=Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        return prob


if __name__ == "__main__":
    unittest.main()
