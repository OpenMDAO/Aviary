import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_detailed import \
    DetailedWingBendingFact
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class DetailedWingBendingFactTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()


class DetailedWingBendingTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    # Skip model that doesn't use detailed wing.
    @parameterized.expand(get_flops_case_names(omit=['LargeSingleAisle2FLOPS', 'LargeSingleAisle2FLOPSalt']),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        self.prob.model.add_subsystem(
            "wing",
            DetailedWingBendingFact(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
                        Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
                        Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
                        Mission.Design.GROSS_MASS,
                        Aircraft.Engine.POD_MASS,
                        Aircraft.Wing.ASPECT_RATIO,
                        Aircraft.Wing.ASPECT_RATIO_REF,
                        Aircraft.Wing.STRUT_BRACING_FACTOR,
                        Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                        Aircraft.Engine.WING_LOCATIONS,
                        Aircraft.Wing.THICKNESS_TO_CHORD,
                        Aircraft.Wing.THICKNESS_TO_CHORD_REF],
            output_keys=[Aircraft.Wing.BENDING_FACTOR,
                         Aircraft.Wing.ENG_POD_INERTIA_FACTOR],
            method='fd',
            atol=1e-3,
            rtol=1e-5)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
