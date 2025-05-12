import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import aviary.api as av
from aviary.subsystems.propulsion.gearbox.gearbox_builder import GearboxBuilder


class TestGearbox(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.options = av.AviaryValues()

        self.gearbox = GearboxBuilder()

    def test_gearbox_premission(self):
        prob = self.prob
        prob.model.add_subsystem(
            'gearbox_premission', self.gearbox.build_pre_mission(self.options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(av.Aircraft.Engine.RPM_DESIGN, 6195, units='rpm')
        prob.set_val(av.Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN, 375, units='hp')
        prob.set_val(av.Aircraft.Engine.Gearbox.GEAR_RATIO, 12.6, units=None)
        prob.set_val(av.Aircraft.Engine.Gearbox.SPECIFIC_TORQUE, 103, units='N*m/kg')

        prob.run_model()

        mass = prob.get_val(av.Aircraft.Engine.Gearbox.MASS, 'lb')
        torque_max = prob.get_val('gearbox_premission.torque_comp.torque_max', 'lbf*ft')

        torque_max_expected = 4005.84967696
        mass_expected = 116.25002688
        assert_near_equal(mass, mass_expected, tolerance=1e-6)
        assert_near_equal(torque_max, torque_max_expected, tolerance=1e-6)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)

    def test_gearbox_mission(self):
        prob = self.prob
        num_nodes = 3
        prob.model.add_subsystem(
            'gearbox_mission',
            self.gearbox.build_mission(num_nodes=num_nodes, aviary_inputs=self.options),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(av.Dynamic.Vehicle.Propulsion.RPM + '_in', [5000, 6195, 6195], units='rpm')
        prob.set_val(av.Dynamic.Vehicle.Propulsion.SHAFT_POWER + '_in', [100, 200, 375], units='hp')
        prob.set_val(
            av.Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX + '_in', [375, 300, 375], units='hp'
        )
        prob.set_val(av.Aircraft.Engine.Gearbox.GEAR_RATIO, 12.6, units=None)
        prob.set_val(av.Aircraft.Engine.Gearbox.EFFICIENCY, 0.98, units=None)

        prob.run_model()

        shaft_power = prob.get_val(av.Dynamic.Vehicle.Propulsion.SHAFT_POWER + '_out', 'hp')
        rpm = prob.get_val(av.Dynamic.Vehicle.Propulsion.RPM + '_out', 'rpm')
        torque = prob.get_val(av.Dynamic.Vehicle.Propulsion.TORQUE + '_out', 'ft*lbf')
        shaft_power_max = prob.get_val(av.Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX + '_out', 'hp')

        shaft_power_expected = [98.0, 196.0, 367.5]
        rpm_expected = [396.82539683, 491.66666667, 491.66666667]
        torque_expected = [1297.0620786, 2093.72409783, 3925.73268342]
        shaft_power_max_expected = [367.5, 294.0, 367.5]

        assert_near_equal(shaft_power, shaft_power_expected, tolerance=1e-6)
        assert_near_equal(rpm, rpm_expected, tolerance=1e-6)
        assert_near_equal(torque, torque_expected, tolerance=1e-6)
        assert_near_equal(shaft_power_max, shaft_power_max_expected, tolerance=1e-6)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)


class TestGearboxBuilder(av.TestSubsystemBuilderBase):
    def setUp(self):
        self.subsystem_builder = GearboxBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
