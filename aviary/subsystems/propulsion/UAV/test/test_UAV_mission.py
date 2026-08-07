import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.UAV.model.UAV_mission import UAVPropMission
from aviary.variable_info.UAV_variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues



class TestUAVMission(unittest.TestCase):
    @use_tempdirs
    def test_residual(self):
        nn = 3

        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, 1)
        options.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 100, units='A')
        prob.model.add_subsystem('rc_prop_group', UAVPropMission(num_nodes=nn, aviary_options= options), promotes=['*'])





        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.full(nn, 0.3))
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')

        prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
        prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
        prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
        prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')



        prob.run_model()

        battery_power = prob.get_val('battery.power', units='W')
        esc_power = prob.get_val('esc.power', units='W')
        motor_power = prob.get_val('motor.power', units='W')
        prop_power = prob.get_val(Dynamic.Vehicle.Propulsion.PROP_POWER, units='W')


        rpm = prob.get_val(Dynamic.Vehicle.Propulsion.RPM, units='rev/s')
        rpm_constraint = prob.get_val('prop.rpm_constraint', units='rev/s')
        current_flow = prob.get_val(Dynamic.Vehicle.Propulsion.CURRENT, units='A')
        rpm_defect = prob.get_val('rpm_balance.rpm_defect', units='rev/s')
        expected = battery_power + esc_power + motor_power - prop_power

        assert_near_equal(current_flow, np.full(nn, 30.0), tolerance=1e-10)
        assert_near_equal(battery_power, np.full(nn, 621.0), tolerance=1e-8)
        assert_near_equal(esc_power, np.full(nn, -190.69342459), tolerance=5e-5)
        assert_near_equal(motor_power, np.full(nn, -31.84218984), tolerance=5e-4)
        assert_near_equal(prop_power, np.full(nn, 261.66742626), tolerance=5e-5)


        assert_near_equal(rpm, np.full(nn, 23.40146028), tolerance=5e-4)
        assert_near_equal(rpm_constraint, np.full(nn, -65.0), tolerance=5e-4)
        assert_near_equal(rpm_defect, np.full(nn, 36.59853972), tolerance=5e-5)


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





if __name__ == '__main__':
    unittest.main()
