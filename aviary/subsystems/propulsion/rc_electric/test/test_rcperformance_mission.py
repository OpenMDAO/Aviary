import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission
from aviary.variable_info.variables import Aircraft, Dynamic


class TestRCPropMission(unittest.TestCase):
    @use_tempdirs
    def test_residual(self):
        nn = 3

        prob = om.Problem()

        prob.model.add_subsystem('rc_prop_group', RCPropMission(num_nodes=nn), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.linspace(0, 1, nn))
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.PEAK_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
        prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
        prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
        prob.set_val(Aircraft.Engine.NUM_ENGINES, 1, units='unitless')
        prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')

        prob.run_model()

        battery_power = prob.get_val('battery.power', units='W')
        esc_power = prob.get_val('esc.power', units='W')
        motor_power = prob.get_val('motor.power', units='W')
        prop_power = prob.get_val(Dynamic.Vehicle.Propulsion.PROP_POWER, units='W')
        power_residual = battery_power + esc_power + motor_power - prop_power
        print(battery_power, esc_power, motor_power, prop_power)
        assert_near_equal(power_residual, np.zeros(3), tolerance=1e-5)
        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)


        # torque = prob.get_val(Dynamic.Vehicle.Propulsion.TORQUE, 'N*m')
        # max_torque = prob.get_val(Dynamic.Vehicle.Propulsion.TORQUE_MAX, 'N*m')
        # efficiency = prob.get_val('motor_efficiency')
        # shp = prob.get_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER)
        # max_shp = prob.get_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX)
        # power = prob.get_val(Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN, 'kW')

        # torque_expected = np.array([0.0, 900.0, 1800.0]) * 1.12
        # max_torque_expected = [2016, 2016, 2016]
        # eff_expected = [0.871, 0.958625, 0.954]
        # shp_expected = [0.0, 316.67253948185123, 1266.690157927405]
        # max_shp_expected = [0.0, 633.3450789637025, 1266.690157927405]
        # power = [0.0, 330.3403723894654, 1327.7674611398375]

        # assert_near_equal(torque, torque_expected, tolerance=1e-9)
        # assert_near_equal(max_torque, max_torque_expected, tolerance=1e-9)
        # assert_near_equal(efficiency, eff_expected, tolerance=1e-9)
        # assert_near_equal(shp, shp_expected, tolerance=1e-9)
        # assert_near_equal(max_shp, max_shp_expected, tolerance=1e-9)
        # assert_near_equal(power, power, tolerance=1e-9)

        # partial_data = prob.check_partials(out_stream=None, method='cs')
        # assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
