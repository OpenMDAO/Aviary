import unittest
from copy import deepcopy

import openmdao.api as om

from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.models.N3CC.N3CC_data import (
    detailed_takeoff_climbing, detailed_takeoff_ground, takeoff_subsystem_options, inputs)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic, Mission

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
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.MASS,
                Dynamic.Mission.LIFT,
                Dynamic.Mission.THRUST_TOTAL,
                Dynamic.Mission.DRAG],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE],
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
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.MASS,
                Dynamic.Mission.LIFT,
                Dynamic.Mission.THRUST_TOTAL,
                Dynamic.Mission.DRAG],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE],
            tol=1e-2, atol=1e-9, rtol=1e-11,
            check_values=False, check_partials=True)

    @staticmethod
    def _make_prob(climbing):
        prob = om.Problem()

        time, _ = detailed_takeoff_climbing.get_item('time')
        nn = len(time)
        aviary_options = inputs

        default_mission_subsystems = get_default_mission_subsystems(
            'FLOPS', build_engine_deck(aviary_options))

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

    def test_IO(self):
        prob = self._make_prob(climbing=False)
        exclude_inputs = {
            'engine_scaling.thrust_net_max_unscaled',
            'vectorize_performance.fuel_flow_rate_negative_0',
            'takeoff_eom.forces_vertical',
            'vectorize_performance.thrust_net_0',
            'takeoff_eom.distance_rate',
            'takeoff_eom.altitude_rate',
            'vectorize_performance.nox_rate_0',
            'engine_scaling.shaft_power_max_unscaled',
            'takeoff_eom.acceleration_vertical',
            'vectorize_performance.t4_0',
            'takeoff_eom.acceleration_horizontal',
            'takeoff_eom.forces_horizontal',
            'angle_of_attack',
            'throttle_max',
            'vectorize_performance.thrust_net_max_0',
            'core_aerodynamics.aero_forces.CD',
            'vectorize_performance.shaft_power_0',
            'engine_scaling.electric_power_in_unscaled',
            'core_aerodynamics.ground_effect.base_drag_coefficient',
            'engine_scaling.shaft_power_unscaled',
            'engine_scaling.fuel_flow_rate_unscaled',
            'core_aerodynamics.ground_effect_drag',
            'core_aerodynamics.ground_effect.base_lift_coefficient',
            'engine_scaling.thrust_net_unscaled',
            'core_aerodynamics.aero_forces.CL',
            'vectorize_performance.electric_power_in_0',
            'engine_scaling.nox_rate_unscaled',
            'v_stall'}
        exclude_outputs = {
            'turbofan_22k.shaft_power',
            'core_aerodynamics.takeoff_polar.lift_coefficient',
            'turbofan_22k.throttle_max',
            'core_aerodynamics.takeoff_polar.drag_coefficient',
            'turbofan_22k.thrust_net',
            'core_aerodynamics.ground_effect.drag_coefficient',
            'turbofan_22k.max_interpolation.thrust_net_max_unscaled',
            'v_over_v_stall', 'turbofan_22k.interpolation.thrust_net_unscaled',
            'takeoff_eom.climb_gradient_forces_horizontal',
            'core_aerodynamics.ground_effect.lift_coefficient',
            'v_stall',
            'turbofan_22k.interpolation.electric_power_in_unscaled',
            'turbofan_22k.nox_rate',
            'turbofan_22k.interpolation.thrust_net_max_unscaled',
            'takeoff_eom.climb_gradient_forces_vertical',
            'core_aerodynamics.climb_drag_coefficient',
            'turbofan_22k.shaft_power_max',
            'takeoff_eom.forces_horizontal',
            'viscosity',
            'turbofan_22k.electric_power_in',
            'turbofan_22k.interpolation.nox_rate_unscaled',
            'turbofan_22k.interpolation.fuel_flow_rate_unscaled',
            'turbofan_22k.thrust_net_max',
            'takeoff_eom.acceleration_horizontal',
            'takeoff_eom.forces_vertical',
            'turbofan_22k.fuel_flow_rate_negative',
            'drhos_dh',
            'takeoff_eom.acceleration_vertical',
            'EAS'}
        assert_match_varnames(prob.model,
                              exclude_inputs=exclude_inputs,
                              exclude_outputs=exclude_outputs)


if __name__ == "__main__":
    unittest.main()
