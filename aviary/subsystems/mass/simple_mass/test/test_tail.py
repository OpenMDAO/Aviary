import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.simple_mass.tail import HorizontalTailMass, VerticalTailMass
from aviary.variable_info.variables import Aircraft


class TailMassTestCase(unittest.TestCase):
    """Tail mass test case."""

    def test_horizontal_tail(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'Tail',
            HorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        # self.prob.model.set_input_defaults(Aircraft.HorizontalTail.SPAN, val=1, units='m')
        # self.prob.model.set_input_defaults(Aircraft.HorizontalTail.ROOT_CHORD, val=1, units='m')
        # self.prob.model.set_input_defaults('tip_chord_tail', val=0.5, units='m')
        prob.model.set_input_defaults('thickness_ratio', val=0.12)
        # self.prob.model.set_input_defaults('skin_thickness', val=0.002, units='m')
        # self.prob.model.set_input_defaults('twist_tail', val=np.zeros(10), units='deg')

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-4

        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 10.6966719, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_vertical_tail(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'Tail',
            VerticalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        # self.prob.model.set_input_defaults(Aircraft.VerticalTail.SPAN, val=1, units='m')
        # self.prob.model.set_input_defaults(Aircraft.VerticalTail.ROOT_CHORD, val=1, units='m')
        # self.prob.model.set_input_defaults('tip_chord_tail', val=0.5, units='m')
        prob.model.set_input_defaults('thickness_ratio', val=0.12)
        # self.prob.model.set_input_defaults('skin_thickness', val=0.002, units='m')
        # self.prob.model.set_input_defaults('twist_tail', val=np.zeros(10), units='deg')

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-4

        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 10.6966719, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


if __name__ == '__main__':
    unittest.main()
