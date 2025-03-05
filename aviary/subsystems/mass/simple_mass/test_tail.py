import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG

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

        self.prob.model.set_input_defaults(
            "tail_mass",
            val=10, 
            units="kg")

        self.prob.setup(
            check=False,
            force_alloc_complex=True)
    
    def test_case(self):
        
        self.prob.run_model()

        tol = 1e-10

        assert_near_equal(
            self.prob["tail_mass"],
            100, # still need to calculate by hand
            tol)
        
        partial_data = self.prob.check_partials(
            out_stream=None,
            method="fd") # Finite difference because cs is used in tail mass calculation right now
        
        assert_check_partials(
            partial_data,
            atol=1e-15,
            rtol=1e-15)

if __name__ == "__main__":
    unittest.main()