import unittest

import numpy as np
import jax.numpy as jnp
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


"""
The little bit of path code below is not important overall. This is for me to test 
within the Docker container and VS Code before I push everything fully to the Github 
repository. These lines can be deleted as things are updated further.

"""

import sys
import os


module_path = os.path.abspath("/home/omdao/Aviary/aviary/subsystems/mass")
if module_path not in sys.path:
    sys.path.append(module_path)

from simple_mass.fuselage import FuselageMassAndCOG
from simple_mass.wing import WingMassAndCOG
from simple_mass.tail import TailMassAndCOG
from simple_mass.mass_summation import MassSummation, StructureMass

class MassSummationTest(unittest.TestCase):
    """
    Total mass summation test case.

    """

    def test_case(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'tot',
            MassSummation(),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

        # self.prob.model.set_input_defaults(
        #     'span',
        #     val=1.0,
        #     units='m'
        # )

        # self.prob.model.set_input_defaults(
        #     'span_tail',
        #     val=1.0,
        #     units='m'
        # )

        # self.prob.model.set_input_defaults(
        #     "root_chord",
        #     val=1,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "root_chord_tail",
        #     val=1,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "tip_chord",
        #     val=0.5,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "tip_chord_tail",
        #     val=0.5,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "thickness_ratio",
        #     val=0.12
        # )

        # self.prob.model.set_input_defaults(
        #     "skin_thickness",
        #     val=0.002,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "twist",
        #     val=jnp.zeros(10),
        #     units="deg"
        # )

        # self.prob.model.set_input_defaults(
        #     "twist_tail",
        #     val=jnp.zeros(10),
        #     units="deg"
        # )

        # self.prob.model.set_input_defaults(
        #     "length",
        #     val=2.5,
        #     units="m")
        
        # self.prob.model.set_input_defaults(
        #     "diameter",
        #     val=0.5,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "taper_ratio",
        #     val=0.5
        # )

        # self.prob.model.set_input_defaults(
        #     "curvature",
        #     val=0.0,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "y_offset",
        #     val=0.0,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     "z_offset",
        #     val=0.0,
        #     units="m"
        # )

        # self.prob.model.set_input_defaults(
        #     'thickness',
        #     val=0.05,
        #     units='m'
        # )

        # self.prob.model.set_input_defaults(
        #     "is_hollow",
        #     val=True
        # )

        # n_points = 10 # = num_sections
        # x = jnp.linspace(0, 1, n_points)
        # max_thickness_chord_ratio = 0.12
        # thickness_dist = 5 * max_thickness_chord_ratio * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        # self.prob.model.set_input_defaults(
        #     "thickness_dist",
        #     val=thickness_dist,
        #     units="m"
        # )

        self.prob.setup(
            check=False,
            force_alloc_complex=True
        )

        self.prob.run_model()

        om.n2(self.prob)

        tol = 1e-10
        assert_near_equal(
            self.prob['total_weight_wing'],
            4.22032,
            tol
        )

        partial_data = self.prob.check_partials(
            out_stream=None,
            method="cs"
        )

        assert_check_partials(
            partial_data
        )


class StructureMassTest(unittest.TestCase):
    """
    Total structure summation mass test case.

    """

    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            "tot",
            StructureMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.prob.setup(
            check=False,
            force_alloc_complex=True
        )

        self.prob.set_val('fuse_mass', val=100.0)
        self.prob.set_val('wing_mass', val=4.2)
        self.prob.set_val('tail_mass', val=4.25)
    
    def test_case(self):

        self.prob.run_model()

        tol = 1e-10
        assert_near_equal(
            self.prob['structure_mass'],
            108.45,
            tol
        )

        partial_data = self.prob.check_partials(
            out_stream=None,
            method="cs"
        )

        assert_check_partials(partial_data)

if __name__ == "__main__":
    unittest.main()



