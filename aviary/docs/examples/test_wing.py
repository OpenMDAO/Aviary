import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal



class WingMassTestCase(unittest.TestCase):
    """
    Wing mass test case

    """

    def setUp(self):

        #self.prob = om.Problem()
        #self.prob.model.add_subsystem(
        #    "wing",
        #    WingMassAndCOG(),
        #    promotes_inputs=["*"],
        #    promotes_outputs=['*'],
        #)

        self.prob.model.set_input_defaults(
            "wing_mass", val=10, units="kg"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):

        self.prob.run_model()

        tol = 1e-10 
        assert_near_equal(self.prob["wing_mass"], 100, tol) # Need to calculate first -- filler value for now
        
        partial_data = self.prob.check_partials(out_stream=None, method="fd") # finite difference used because cs is used in wing.py calculation
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)
    
        
if __name__ == "__main__":
    unittest.main()