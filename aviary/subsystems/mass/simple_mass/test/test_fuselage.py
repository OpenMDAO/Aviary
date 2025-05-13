import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.simple_mass.fuselage import FuselageMassAndCOG

class FuselageMassTestCase(unittest.TestCase):
    """
    Fuselage mass test case.

    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "fuselage",
            FuselageMassAndCOG(),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.prob.model.set_input_defaults(
            "length",
            val=2.0,
            units="m")
        
        self.prob.model.set_input_defaults(
            "diameter",
            val=0.4,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "taper_ratio",
            val=0.9999999999
        )

        self.prob.model.set_input_defaults(
            "curvature",
            val=0.0,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "y_offset",
            val=0.0,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "z_offset",
            val=0.0,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "is_hollow",
            val=True
        )
        
        self.prob.setup(
            check=False,
            force_alloc_complex=True)
        
    def test_case(self):

        self.prob.run_model()

        tol=1e-3

        assert_near_equal(
            self.prob["total_weight"],
            167.35489,
            tol)
        
        partial_data = self.prob.check_partials(
            out_stream=None,
            method="cs") 
        
        from pprint import pprint
        pprint(partial_data)
        
        assert_check_partials(
            partial_data,
            atol=1e-15,
            rtol=1e-15)
        
if __name__ == "__main__":
    unittest.main()