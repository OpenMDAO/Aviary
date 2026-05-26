import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.motor.model.motor_mission import MotorMission
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic


@use_tempdirs
class TestMotorMission(unittest.TestCase):
    def test_motor_mission(self):
        nn = 3

        filename = 'electric_motor_1800Nm_6000rpm.csv'
        options = AviaryValues()
        options.set_val(Aircraft.Engine.Motor.DATA_FILE, get_path(filename))
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')
        options.set_val(
            Aircraft.Engine.FIXED_RPM, 6000, 'rpm'
        )  # set fixed RPM so we can manually set it

        prob = om.Problem()

        prob.model.add_subsystem('motor_mission', MotorMission(num_nodes=nn), promotes=['*'])

        setup_model_options(prob, options)

        prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.RPM, val=np.ones(nn) * 6000, units='rpm'
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.linspace(0, 1, nn))
        prob.set_val(Dynamic.Vehicle.Propulsion.RPM, np.linspace(0, 6000, nn))

        prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1.12)

        prob.run_model()

        torque = prob.get_val(Dynamic.Vehicle.Propulsion.TORQUE, 'N*m')
        efficiency = prob.get_val('efficiency')
        shp = prob.get_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER)
        power = prob.get_val(Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN, 'kW')

        torque_expected = np.array([0.0, 900.0, 1800.0]) * 1.12
        eff_expected = [0.871, 0.958625, 0.954]
        shp_expected = [0.0, 316.67253948185123, 1266.690157927405]
        power = [0.0, 330.3403723894654, 1327.7674611398375]

        assert_near_equal(torque, torque_expected, tolerance=1e-9)
        assert_near_equal(efficiency, eff_expected, tolerance=1e-9)
        assert_near_equal(shp, shp_expected, tolerance=1e-9)
        assert_near_equal(power, power, tolerance=1e-9)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
