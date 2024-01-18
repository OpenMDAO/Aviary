import unittest

import openmdao.api as om

from aviary.mission.flops_based.ode.landing_eom import FlareEOM
from aviary.models.N3CC.N3CC_data import (
    detailed_landing_flare, inputs)
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic


class FlareEOMTest(unittest.TestCase):
    def test_case(self):
        prob = om.Problem()

        time, _ = detailed_landing_flare.get_item('time')
        nn = len(time)
        aviary_options = inputs

        prob.model.add_subsystem(
            "landing_flare_eom",
            FlareEOM(num_nodes=nn, aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            'landing_flare_eom',
            input_validation_data=detailed_landing_flare,
            output_validation_data=detailed_landing_flare,
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
                Dynamic.Mission.ALTITUDE_RATE],
            tol=1e-2, atol=1e-8, rtol=5e-10)


if __name__ == "__main__":
    unittest.main()
