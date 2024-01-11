import unittest

import openmdao.api as om

from aviary.interface.default_phase_info.height_energy import default_mission_subsystems
from aviary.mission.flops_based.ode.landing_ode import FlareODE
from aviary.models.N3CC.N3CC_data import (
    detailed_landing_flare, inputs, landing_subsystem_options)
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic as _Dynamic

Dynamic = _Dynamic.Mission


class FlareODETest(unittest.TestCase):
    def test_case(self):
        prob = om.Problem()

        time, _ = detailed_landing_flare.get_item('time')
        nn = len(time)
        aviary_options = inputs

        prob.model.add_subsystem(
            "landing_flare_ode",
            FlareODE(
                num_nodes=nn,
                subsystem_options=landing_subsystem_options,
                core_subsystems=default_mission_subsystems,
                aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            'landing_flare_ode',
            input_validation_data=detailed_landing_flare,
            output_validation_data=detailed_landing_flare,
            input_keys=[
                'angle_of_attack',
                Dynamic.FLIGHT_PATH_ANGLE,
                Dynamic.VELOCITY,
                Dynamic.MASS,
                Dynamic.LIFT,
                Dynamic.THRUST_TOTAL,
                Dynamic.DRAG],
            output_keys=[
                Dynamic.RANGE_RATE,
                Dynamic.ALTITUDE_RATE],
            tol=1e-2, atol=5e-9, rtol=5e-9,
            check_values=False, check_partials=True)


if __name__ == "__main__":
    unittest.main()
