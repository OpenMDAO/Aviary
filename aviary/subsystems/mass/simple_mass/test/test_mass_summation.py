import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.simple_mass.mass_summation import SimpleMassSummation, StructureMass
from aviary.variable_info.variables import Aircraft


class MassSummationTest(unittest.TestCase):
    """Total mass summation test case."""

    def test_case(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem('tot', SimpleMassSummation(), promotes=['*'])

        self.prob.setup()

        self.prob.set_val(Aircraft.Fuselage.MASS, val=100.0)
        self.prob.set_val(Aircraft.Wing.MASS, val=4.2)
        self.prob.set_val(Aircraft.HorizontalTail.MASS, val=4.25)
        self.prob.set_val(Aircraft.VerticalTail.MASS, val=4.5)

        self.prob.run_model()

        # om.n2(self.prob)

        tol = 1e-10

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 112.95, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class StructureMassTest(unittest.TestCase):
    """Total structure summation mass test case."""

    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'tot',
            StructureMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Aircraft.Fuselage.MASS, val=100.0)
        self.prob.set_val(Aircraft.Wing.MASS, val=4.2)
        self.prob.set_val(Aircraft.HorizontalTail.MASS, val=4.25)
        self.prob.set_val(Aircraft.VerticalTail.MASS, val=4.5)

    def test_case(self):
        self.prob.run_model()

        tol = 1e-10

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 112.95, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


if __name__ == '__main__':
    unittest.main()
