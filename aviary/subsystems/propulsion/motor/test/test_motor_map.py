import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap
from aviary.variable_info.variables import Aircraft, Dynamic


class TestGearbox(unittest.TestCase):

    @use_tempdirs
    def test_motor_map(self):
        nn = 3

        prob = om.Problem()

        prob.model.add_subsystem('motor_map', MotorMap(num_nodes=3), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Mission.THROTTLE, np.linspace(0, 1, nn))
        prob.set_val(Dynamic.Mission.RPM, np.linspace(0, 6000, nn))
        prob.set_val('torque_unscaled', np.linspace(0, 1800, nn), 'N*m')
        prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1.12)

        prob.run_model()

        torque = prob.get_val(Dynamic.Mission.TORQUE)
        efficiency = prob.get_val('motor_efficiency')

        torque_expected = np.array([0.0, 900.0, 1800.0]) * 1.12
        eff_expected = [0.871, 0.958625, 0.954]
        assert_near_equal(torque, torque_expected, tolerance=1e-9)
        assert_near_equal(efficiency, eff_expected, tolerance=1e-9)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
