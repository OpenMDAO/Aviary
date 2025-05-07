import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.taxi_eom import TaxiFuelComponent
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Dynamic, Mission


class TaxiFuelComponentTestCase(unittest.TestCase):
    """Test the computation of fuel consumed during taxi in TaxiFuelComponent component."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Mission.Taxi.DURATION, 0.1677, units='h')

        self.prob.model.add_subsystem('taxi', TaxiFuelComponent(), promotes=['*'])

        setup_model_options(self.prob, aviary_options)

    def test_fuel_consumed(self):
        self.prob.setup(force_alloc_complex=True)

        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            -1512,
            units='lbm/h',
        )
        self.prob.set_val(Mission.Summary.GROSS_MASS, 175400.0, units='lbm')

        self.prob.run_model()

        assert_near_equal(self.prob['taxi_fuel_consumed'], 1512 * 0.1677, 1e-6)
        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
