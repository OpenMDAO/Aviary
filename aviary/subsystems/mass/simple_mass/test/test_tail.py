import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import numpy as np

from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG
from aviary.variable_info.variables import Aircraft

class TailMassTestCase(unittest.TestCase):
    """
    Tail mass test case.

    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "Tail",
            TailMassAndCOG(),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        tail_type = self.prob.model.Tail.options['tail_type']

        if tail_type == 'horizontal':
            self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.SPAN,
            val=1,
            units="m"
            )

            self.prob.model.set_input_defaults(
                Aircraft.HorizontalTail.ROOT_CHORD,
                val=1,
                units="m"
            )
        else:
            self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SPAN,
            val=1,
            units="m"
            )

            self.prob.model.set_input_defaults(
                Aircraft.VerticalTail.ROOT_CHORD,
                val=1,
                units="m"
            )

        self.prob.model.set_input_defaults(
            "tip_chord_tail",
            val=0.5,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "thickness_ratio",
            val=0.12
        )

        self.prob.model.set_input_defaults(
            "skin_thickness",
            val=0.002,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "twist_tail",
            val=np.zeros(10),
            units="deg"
        )

        self.prob.setup(
            check=False,
            force_alloc_complex=True)
    
    def test_case(self):

        tail_type = self.prob.model.Tail.options['tail_type']
        
        self.prob.run_model()

        tol = 1e-4

        if tail_type == 'horizontal':
            assert_near_equal(
                self.prob[Aircraft.HorizontalTail.MASS],
                4.22032, 
                tol)
        else:
            assert_near_equal(
                self.prob[Aircraft.VerticalTail.MASS],
                4.22032, 
                tol)
        
        partial_data = self.prob.check_partials(
            out_stream=None,
            method="cs") 
        
        assert_check_partials(
            partial_data,
            atol=1e-15,
            rtol=1e-15)

if __name__ == "__main__":
    unittest.main()