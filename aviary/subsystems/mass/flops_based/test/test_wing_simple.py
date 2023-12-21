import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_simple import SimpleWingBendingFact
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class SimpleWingBendingFactTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    # Only dataset that uses the simple wing.
    @parameterized.expand(get_flops_case_names(only=['LargeSingleAisle2FLOPS', 'LargeSingleAisle2FLOPSalt']),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "wing",
            SimpleWingBendingFact(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.AREA,
                        Aircraft.Wing.SPAN,
                        Aircraft.Wing.TAPER_RATIO,
                        Aircraft.Wing.THICKNESS_TO_CHORD,
                        Aircraft.Wing.STRUT_BRACING_FACTOR,
                        Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                        Aircraft.Wing.ASPECT_RATIO,
                        Aircraft.Wing.SWEEP],
            output_keys=[Aircraft.Wing.BENDING_FACTOR,
                         Aircraft.Wing.ENG_POD_INERTIA_FACTOR],
            atol=1e-11,
            rtol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
