import unittest

import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.energy.battery_builder import BatteryBuilder
import aviary.api as av


class TestBatteryDerivs(unittest.TestCase):
    def setUp(self):
        self.prob = prob = om.Problem()

        self.options = av.AviaryValues()

        self.battery = BatteryBuilder()

    def test_battery_premission(self):
        prob = self.prob
        prob.model.add_subsystem('battery_premission',
                                 self.battery.build_pre_mission(self.options),
                                 promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(av.Aircraft.Battery.PACK_ENERGY_DENSITY, 550, units='kW*h/kg')
        prob.set_val(av.Aircraft.Battery.PACK_MASS, 1200, units='lbm')
        prob.set_val(av.Aircraft.Battery.ADDITIONAL_MASS, 115, units='lbm')

        prob.run_model()

        mass_expected = 1_315
        energy_expected = 1_077_735.47112

        mass = prob.get_val(av.Aircraft.Battery.MASS, 'lbm')
        energy = prob.get_val(av.Aircraft.Battery.ENERGY_CAPACITY, 'MJ')

        assert_near_equal(mass, mass_expected, tolerance=1e-10)
        assert_near_equal(energy, energy_expected, tolerance=1e-10)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)

    def test_battery_mission(self):
        prob = self.prob
        prob.model.add_subsystem('battery_mission',
                                 subsys=self.battery.build_mission(num_nodes=4),
                                 promotes=['*'])

        efficiency = 0.95
        prob.model.set_input_defaults(
            av.Aircraft.Battery.ENERGY_CAPACITY, 10_000, units='kJ')
        prob.model.set_input_defaults(
            av.Aircraft.Battery.EFFICIENCY, efficiency, units='unitless')
        prob.model.set_input_defaults(av.Dynamic.Mission.CUMULATIVE_ELECTRIC_ENERGY_USED, [
                                      0, 2_000, 5_000, 9_500], units='kJ')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        soc_expected = np.array([1., 0.7894736842105263, 0.4736842105263159, 0.])
        soc = prob.get_val(av.Dynamic.Mission.BATTERY_STATE_OF_CHARGE, 'unitless')

        assert_near_equal(soc, soc_expected, tolerance=1e-10)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)


class TestBattery(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = BatteryBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
