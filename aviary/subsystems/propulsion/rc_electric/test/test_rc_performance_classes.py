import unittest
import aviary.api as av
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import Battery
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import Motor
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import Propeller
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import ElectronicSpeedController
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import Vectorization
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import PropCoefficients
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import PowerResiduals
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import PowerImplicit
from aviary.variable_info.dbf_variables import Aircraft, Dynamic

class TestBattery(unittest.TestCase):
    @use_tempdirs
    def test_battery(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('battery', Battery(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
        prob.set_val(Dynamic.Vehicle.Propulsion.CURRENT, np.full(nn, 10.0), units='A')

        prob.run_model()

        power = prob.get_val('power', units='W')
        expected_power = np.full(nn, 222.0) - np.full(nn, 5.0)
        assert_near_equal(power, expected_power, tolerance=1e-5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)
        
class TestMotor(unittest.TestCase):
    @use_tempdirs
    def test_motor(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('motor', Motor(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
        prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
        prob.set_val('voltage_in', 22.2, units='V')
        prob.set_val('current', np.full(nn, 10.0), units='A')
                     
        prob.set_val(Dynamic.Vehicle.Propulsion.CURRENT, np.full(nn, 10.0), units='A')


        prob.run_model()

        # voltage_prop = voltage_in - current*R = 22.2-10**2 * 0.032 = 21.88 #Volts
        # RPM = voltage_prop * KV = 21.88 * 420 = 9189 #Rpm's


        
        expected_current = np.full(nn, 10.0)

        rpm = prob.get_val(Dynamic.Vehicle.Propulsion.RPM, units='rpm')

        #power = -I**2 * R - idle * voltage_prop = -10**2 * 0.032 - 0.91 * 21.88 = -32 - 19.9 = -51.9 W

        # current_constraint = Dynamic.Vehicle.Propulsion.CURRENT - Aircraft.Engine.Motor.MAX_CONT_CURRENT = 10 - 120 = -110 A


        assert_near_equal(rpm, np.full(nn, 9189.6), tolerance=1e-5)
       
        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)
        
class TestPropeller(unittest.TestCase):
    @use_tempdirs
    def test_propeller(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('propeller', Propeller(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Dynamic.Vehicle.Propulsion.RPM, np.full(nn, 1000))
        prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
        prob.set_val('ct', np.full(nn, 0.1))
        prob.set_val('cp', np.full(nn, 0.05))


        prob.run_model()

        
        thrust = prob.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='N')
        expected_thrust = np.full(nn, 8158.136)
        assert_near_equal(thrust, expected_thrust, tolerance=1e-5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

class TestESC(unittest.TestCase):
    @use_tempdirs
    def test_esc(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('esc', ElectronicSpeedController(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val('voltage_in', 22.2, units='V')
        prob.set_val(Dynamic.Vehicle.Propulsion.CURRENT, np.full(nn, 10.0), units='A')
        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.full(nn, 0.8))

        prob.run_model()

       
        
        current_out = prob.get_val('current_out', units='A')
        assert_near_equal(current_out, np.full(nn, 10.0), tolerance=1e-5)
    

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            )
        
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

        
class TestPowerResiduals(unittest.TestCase):
    @use_tempdirs
    def test_power_residuals(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('power_residuals', PowerResiduals(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val('power_batt', np.full(nn, 222.0), units='W')
        prob.set_val('power_esc', np.full(nn, 222.0), units='W')
        prob.set_val('power_motor', np.full(nn, 222.0), units='W')
        prob.set_val(Dynamic.Vehicle.Propulsion.PROP_POWER, np.full(nn, 666.0), units='W')

        prob.run_model()

        residual = prob.get_val('power_net', units='W') #expected 0
        expected_residual = np.full(nn, 0.0)
        assert_near_equal(residual, expected_residual, tolerance=1e-5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            )
        
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

        
class TestPowerImplicit(unittest.TestCase):
    @use_tempdirs
    def test_power_implicit(self):
        nn = 3
        comp = PowerImplicit(num_nodes=nn)
        prob = om.Problem()
        prob.model.add_subsystem('power_implicit', PowerImplicit(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)
        prob.set_val('power_batt', np.full(nn, 222.0), units='W')
        prob.set_val('power_esc', np.full(nn, 222.0), units='W')
        prob.set_val('power_motor', np.full(nn, 222.0), units='W')
        prob.set_val(Dynamic.Vehicle.Propulsion.PROP_POWER, np.full(nn, 666.0), units='W')

        inputs = {
            'power_batt': np.full(nn, 222.0),
            'power_esc': np.full(nn, 222.0),
            'power_motor': np.full(nn, 222.0),
            Dynamic.Vehicle.Propulsion.PROP_POWER: np.full(nn, 666.0),
        }
        
        prob.run_model()

        residuals = {Dynamic.Vehicle.Propulsion.CURRENT: np.zeros(nn)}
        comp.apply_nonlinear(inputs,{}, residuals)

        expected_residual = np.full(nn, 0.0)
        assert_near_equal(residuals[Dynamic.Vehicle.Propulsion.CURRENT], np.zeros(nn), 1e-5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
        )

        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)


class TestVectorization(unittest.TestCase):
    @use_tempdirs
    def test_vectorization(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('vectorization', Vectorization(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')


        prob.run_model()

        assert_near_equal(prob.get_val('temp_diameter', units='inch'), np.full(nn, 20),  tolerance=1e-5) 
        assert_near_equal(prob.get_val('temp_pitch', units='inch'), np.full(nn, 10.0), tolerance=1e-5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
        )

        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)


class TestPropCoefficients(unittest.TestCase):
    @use_tempdirs
    def test_prop_coefficients(self):
        nn = 3

        prob = om.Problem()
        prob.model.add_subsystem('prop_coefficients', PropCoefficients(vec_size=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Mission.VELOCITY, np.full(nn, 13.6), units='m/s')
        prob.set_val(Dynamic.Vehicle.Propulsion.RPM, np.full(nn, 100), units='rev/s')
        prob.set_val('temp_diameter', 20, units='inch')
        prob.set_val('temp_pitch', np.full(nn, 12), units='inch')

        prob.run_model()

        ct = prob.get_val('ct')
        cp = prob.get_val('cp')
        self.assertTrue(np.all(np.isfinite(ct)))
        self.assertTrue(np.all(np.isfinite(cp)))
       

        
   

if __name__ == '__main__':
    unittest.main()