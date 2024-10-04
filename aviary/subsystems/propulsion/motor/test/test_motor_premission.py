import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.motor.model.motor_premission import MotorPreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic


class TestGearbox(unittest.TestCase):

    @use_tempdirs
    def test_motor_map(self):
        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')

        prob.model.add_subsystem(
            'motor_map', MotorPreMission(aviary_inputs=options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        # prob.set_val('torque_unscaled', np.linspace(0, 1800, nn), 'N*m')
        prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1.12)

        prob.run_model()

        torque_max = prob.get_val(Aircraft.Engine.Motor.TORQUE_MAX, 'N*m')
        mass = prob.get_val(Aircraft.Engine.Motor.MASS, 'kg')

        torque_max_expected = 2016
        mass_expected = 93.36999121578062

        assert_near_equal(torque_max, torque_max_expected, tolerance=1e-9)
        assert_near_equal(mass, mass_expected, tolerance=1e-9)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
