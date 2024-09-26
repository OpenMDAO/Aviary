import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.flops_based.ode.takeoff_eom import (
    TakeoffEOM, StallSpeed, DistanceRates, Accelerations, VelocityRate,
    FlightPathAngleRate, SumForces, ClimbGradientForces)
from aviary.models.N3CC.N3CC_data import (
    detailed_takeoff_climbing, detailed_takeoff_ground, inputs)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic, Mission


class TakeoffEOMTest(unittest.TestCase):
    """
    Test detailed takeoff equation of motion
    """

    def test_case_ground(self):
        prob = self._make_prob(climbing=False)

        do_validation_test(
            prob,
            'takeoff_eom_ground',
            input_validation_data=detailed_takeoff_ground,
            output_validation_data=detailed_takeoff_ground,
            input_keys=[
                'angle_of_attack',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.MASS,
                Dynamic.Mission.LIFT,
                Dynamic.Mission.THRUST_TOTAL,
                Dynamic.Mission.DRAG],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE],
            tol=1e-2)

    def test_case_climbing(self):
        prob = self._make_prob(climbing=True)

        do_validation_test(
            prob,
            'takeoff_eom_climbing',
            input_validation_data=detailed_takeoff_climbing,
            output_validation_data=detailed_takeoff_climbing,
            input_keys=[
                'angle_of_attack',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.MASS,
                Dynamic.Mission.LIFT,
                Dynamic.Mission.THRUST_TOTAL,
                Dynamic.Mission.DRAG],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE],
            tol=1e-2, atol=1e-9, rtol=1e-11)

    @staticmethod
    def _make_prob(climbing):
        prob = om.Problem()

        time, _ = detailed_takeoff_climbing.get_item('time')
        nn = len(time)
        aviary_options = inputs

        prob.model.add_subsystem(
            "takeoff_eom",
            TakeoffEOM(num_nodes=nn,
                       aviary_options=aviary_options,
                       climbing=climbing,
                       friction_key=Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT
                       ),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        return prob

    def test_IO(self):
        prob = self._make_prob(climbing=False)
        exclude_inputs = {
            'angle_of_attack', 'acceleration_horizontal',
            'acceleration_vertical', 'forces_vertical', 'forces_horizontal'}
        exclude_outputs = {
            'acceleration_horizontal', 'acceleration_vertical',
            'climb_gradient_forces_vertical', 'forces_horizontal',
            'forces_vertical', 'climb_gradient_forces_horizontal'}
        assert_match_varnames(prob.model,
                              exclude_inputs=exclude_inputs,
                              exclude_outputs=exclude_outputs)

    def test_StallSpeed(self):
        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            "stall_speed", StallSpeed(num_nodes=2), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DENSITY, np.array([1, 2]), units="kg/m**3"
        )
        prob.model.set_input_defaults(
            "area", 10, units="m**2"
        )
        prob.model.set_input_defaults(
            "lift_coefficient_max", 5000, units="unitless"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob["stall_speed"], np.array(
                [0.01980571, 0.01400475]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_DistanceRates_1(self):
        """
        climbing = True
        """

        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            "dist_rates", DistanceRates(num_nodes=2, climbing=True), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, np.array([0.612, 4.096]), units="rad"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, np.array([5.23, 2.7]), units="m/s"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob[Dynamic.Mission.DISTANCE_RATE], np.array(
                [4.280758, -1.56085]), tol
        )
        assert_near_equal(
            prob[Dynamic.Mission.ALTITUDE_RATE], np.array(
                [3.004664, -2.203122]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_DistanceRates_2(self):
        """
        climbing = False
        """

        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            "dist_rates", DistanceRates(num_nodes=2, climbing=False), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, np.array([0.0, 0.0]), units="rad"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, np.array([1.0, 2.0]), units="m/s"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob[Dynamic.Mission.DISTANCE_RATE], np.array([1.0, 2.0]), tol)
        assert_near_equal(
            prob[Dynamic.Mission.ALTITUDE_RATE], np.array([0.0, 0.0]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_Accelerations(self):
        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            "acceleration", Accelerations(num_nodes=2), promotes=["*"]
        )
        prob.model.set_input_defaults(
            "forces_horizontal", [100.0, 200.0], units="N"
        )
        prob.model.set_input_defaults(
            "forces_vertical", [50.0, 100.0], units="N"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob["acceleration_horizontal"], np.array(
                [100.0, 200.0]), tol
        )
        assert_near_equal(
            prob["acceleration_vertical"], np.array(
                [50.0, 100.0]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_VelocityRate(self):
        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            "vel_rate", VelocityRate(num_nodes=2), promotes=["*"]
        )
        prob.model.set_input_defaults(
            "acceleration_horizontal", [100.0, 200.0], units="m/s**2"
        )
        prob.model.set_input_defaults(
            "acceleration_vertical", [50.0, 100.0], units="m/s**2"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DISTANCE_RATE, [160.98, 166.25], units="m/s"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.ALTITUDE_RATE, [1.72, 11.91], units="m/s"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob[Dynamic.Mission.VELOCITY_RATE], np.array(
                [100.5284, 206.6343]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_FlightPathAngleRate(self):
        tol = 1e-6
        prob = om.Problem()
        prob.model.add_subsystem(
            "gamma_rate", FlightPathAngleRate(num_nodes=2), promotes=["*"]
        )
        prob.model.set_input_defaults(
            "acceleration_horizontal", [100.0, 200.0], units="m/s**2"
        )
        prob.model.set_input_defaults(
            "acceleration_vertical", [50.0, 100.0], units="m/s**2"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DISTANCE_RATE, [160.98, 166.25], units="m/s"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.ALTITUDE_RATE, [1.72, 11.91], units="m/s"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE], np.array(
                [0.3039257, 0.51269018]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_SumForcese_1(self):
        """
        climbing = True
        """

        tol = 1e-6
        prob = om.Problem()
        aviary_options = inputs
        prob.model.add_subsystem(
            "sum1", SumForces(num_nodes=2, climbing=True, aviary_options=aviary_options), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.MASS, np.array([106292, 106292]), units="lbm"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DRAG, np.array([47447.13138523, 44343.01567596]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.LIFT, np.array([482117.47027692, 568511.57097785]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.THRUST_TOTAL, np.array([4980.3, 4102]), units="N"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob["forces_horizontal"], np.array(
                [-42466.83, -40241.02]), tol
        )
        assert_near_equal(
            prob["forces_vertical"], np.array(
                [9307.0983, 95701.1990]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_SumForcese_2(self):
        """
        climbing = False
        """

        tol = 1e-6
        prob = om.Problem()
        aviary_options = inputs
        prob.model.add_subsystem(
            "sum2", SumForces(num_nodes=2, climbing=False, aviary_options=aviary_options), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.MASS, np.array([106292, 106292]), units="lbm"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DRAG, np.array([47447.13138523, 44343.01567596]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.LIFT, np.array([482117.47027692, 568511.57097785]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.THRUST_TOTAL, np.array([4980.3, 4102]), units="N"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob["forces_horizontal"], np.array([-42234.154, -37848.486]), tol
        )
        assert_near_equal(
            prob["forces_vertical"], np.array([0.0, 0.0]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_ClimbGradientForces(self):
        """
        climbing = False
        """

        tol = 1e-6
        prob = om.Problem()
        aviary_options = inputs
        prob.model.add_subsystem(
            "climb_grad", ClimbGradientForces(num_nodes=2, aviary_options=aviary_options), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.MASS, np.array([106292, 106292]), units="lbm"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DRAG, np.array([47447.13138523, 44343.01567596]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.LIFT, np.array([482117.47027692, 568511.57097785]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.THRUST_TOTAL, np.array([4980.3, 4102]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, np.array([0.612, 4.096]), units="rad"
        )
        prob.model.set_input_defaults(
            "angle_of_attack", np.array([5.086, 6.834]), units="rad"
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob["climb_gradient_forces_horizontal"],
            np.array([-317261.63, 344951.97]), tol
        )
        assert_near_equal(
            prob["climb_gradient_forces_vertical"],
            np.array([90485.14, 843986.59]), tol
        )

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
