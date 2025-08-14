import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.simple_mass.fuselage import FuselageMass
from aviary.variable_info.variables import Aircraft


class FuselageMassTestCase(unittest.TestCase):
    """Fuselage mass test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'fuselage',
            FuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=2.0, units='m')

        self.prob.model.set_input_defaults('base_diameter', val=0.5, units='m')

        self.prob.model.set_input_defaults('tip_diameter', val=0.3)

        self.prob.model.set_input_defaults('curvature', val=0.0, units='m')

        self.prob.model.set_input_defaults('y_offset', val=0.0, units='m')

        self.prob.model.set_input_defaults('z_offset', val=0.0, units='m')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        self.prob.run_model()

        tol = 1e-3

        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 373.849, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
