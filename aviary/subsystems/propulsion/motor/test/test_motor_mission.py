import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.motor.model.motor_mission import MotorMission
from aviary.variable_info.variables import Aircraft, Dynamic


class TestMotorMission(unittest.TestCase):

    @use_tempdirs
    def test_motor_map(self):
        nn = 3

        prob = om.Problem()

        prob.model.add_subsystem(
            'motor_map', MotorMission(num_nodes=nn), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Mission.THROTTLE, np.linspace(0, 1, nn))
        prob.set_val(Dynamic.Mission.RPM, np.linspace(0, 6000, nn))
        # prob.set_val('torque_unscaled', np.linspace(0, 1800, nn), 'N*m')
        prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1.12)

        prob.run_model()

        torque = prob.get_val(Dynamic.Mission.TORQUE, 'N*m')
        max_torque = prob.get_val(Dynamic.Mission.TORQUE_MAX, 'N*m')
        efficiency = prob.get_val('motor_efficiency')
        shp = prob.get_val(Dynamic.Mission.SHAFT_POWER)
        max_shp = prob.get_val(Dynamic.Mission.SHAFT_POWER_MAX)
        power = prob.get_val(Dynamic.Mission.ELECTRIC_POWER_IN, 'kW')

        torque_expected = np.array([0.0, 900.0, 1800.0]) * 1.12
        max_torque_expected = [2016, 2016, 2016]
        eff_expected = [0.871, 0.958625, 0.954]
        shp_expected = [0.0, 316.67253948185123, 1266.690157927405]
        max_shp_expected = [0.0, 633.3450789637025, 1266.690157927405]
        power = [0.0, 330.3403723894654, 1327.7674611398375]

        assert_near_equal(torque, torque_expected, tolerance=1e-9)
        assert_near_equal(max_torque, max_torque_expected, tolerance=1e-9)
        assert_near_equal(efficiency, eff_expected, tolerance=1e-9)
        assert_near_equal(shp, shp_expected, tolerance=1e-9)
        assert_near_equal(max_shp, max_shp_expected, tolerance=1e-9)
        assert_near_equal(power, power, tolerance=1e-9)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
