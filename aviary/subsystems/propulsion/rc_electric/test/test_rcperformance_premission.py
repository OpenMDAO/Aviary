import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_premission import RCPropPreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class TestRCPropPre(unittest.TestCase):
    @use_tempdirs
    def test_premission_calcs(self):
        prob = om.Problem()
        options = AviaryValues()
        # options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')

        prob.model.add_subsystem(
            'rc_calcs', RCPropPreMission(m=1.3132, b=0.01, aviary_inputs=options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.MASS, 0.707, units='kg')
        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.PEAK_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')

        prob.run_model()

        kv = prob.get_val(Aircraft.Engine.Motor.KV, 'rpm/V')
        resistance = prob.get_val(Aircraft.Engine.Motor.RESISTANCE, 'ohm')
        energy = prob.get_val(Aircraft.Battery.ENERGY_CAPACITY, 'W*h')

        kv_expected = 547.1766667
        resistance_expected = 0.05582266503
        energy_expected = 109.11522

        assert_near_equal(kv, kv_expected, tolerance=1e-9)
        assert_near_equal(resistance, resistance_expected, tolerance=1e-9)
        assert_near_equal(energy, energy_expected, tolerance=1e-9)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
