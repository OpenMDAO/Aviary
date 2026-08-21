import unittest
import aviary.api as av
from aviary.models.external_subsystems.UAV.propulsion.prop_builder import PropBuilder
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from aviary.models.external_subsystems.UAV.propulsion.model.prop_performance import (
    Battery, Motor, Propeller, ElectronicSpeedController, Vectorization, PropCoefficients
)

from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft, Dynamic

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
        prob.model.list_inputs(units=True, prom_name=True)
        prob.model.list_outputs(units=True, prom_name=True, residuals=True)
        voltage_out = prob.get_val('voltage_out', units='V')
        power = prob.get_val('power', units='W')
        assert_near_equal(voltage_out, np.full(nn, 21.7), tolerance=1e-8)
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
        prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
        prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
        prob.set_val('voltage_in', 22.2, units='V')
        prob.set_val('current', np.full(nn, 10.0), units='A')
                     
        prob.set_val(Dynamic.Vehicle.Propulsion.CURRENT, np.full(nn, 10.0), units='A')


        prob.run_model()

        # voltage_prop = voltage_in - current*R = 22.2-10**2 * 0.032 = 21.88 #Volts
        # RPM = voltage_prop * KV = 21.88 * 420 = 9189 #Rpm's


        
        rpm = prob.get_val(Dynamic.Vehicle.Propulsion.RPM, units='rpm')
        power = prob.get_val('power', units='W')
        

        

        assert_near_equal(rpm, np.full(nn, 9189.6), tolerance=1e-5)
        assert_near_equal(power, np.full(nn, -23.1108), tolerance=1e-5)
        
       
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
        prop_power = prob.get_val(Dynamic.Vehicle.Propulsion.PROP_POWER, units='W')
        rpm_constraint = prob.get_val('rpm_constraint', units='rev/s')
        expected_thrust = np.full(nn, 8158.136)


        assert_near_equal(thrust, expected_thrust, tolerance=1e-5)
        assert_near_equal(prop_power, np.full(nn, 2072190.544), tolerance=1e-3)
        assert_near_equal(rpm_constraint, np.full(nn, 875.0), tolerance=1e-8)

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
        prob.model.list_inputs(units=True, prom_name=True)
        prob.model.list_outputs(units=True, prom_name=True, residuals=True)
       
        
        current_out = prob.get_val('current_out', units='A')
        efficiency = prob.get_val('efficiency', units='unitless')
        voltage_out = prob.get_val('voltage_out', units='V')
        power = prob.get_val('power', units='W')

        
        assert_near_equal(current_out, np.full(nn, 10.0), tolerance=1e-5)
        assert_near_equal(efficiency, np.full(nn, 0.9448241882511333), tolerance=1e-6)
        assert_near_equal(voltage_out, np.full(nn, 16.78007758254013), tolerance=1e-6)
        assert_near_equal(power, np.full(nn, -12.24902944), tolerance=1e-6)
    

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
        prob.model.list_inputs(units=True, prom_name=True)
        prob.model.list_outputs(units=True, prom_name=True, residuals=True)
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
        prob.model.list_inputs(units=True, prom_name=True)
        prob.model.list_outputs(units=True, prom_name=True, residuals=True)
        ct = prob.get_val('ct')
        cp = prob.get_val('cp')
        self.assertTrue(np.all(np.isfinite(ct)))
        self.assertTrue(np.all(np.isfinite(cp)))
       

        
   

if __name__ == '__main__':
    unittest.main()