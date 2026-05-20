import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.geometry.flops_based.landing_gear import MainGearLength, NoseGearLength
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class LandingGearLengthTest(unittest.TestCase):
    """This component is unrepresented in our test data."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only='AdvancedSingleAisle'), name_func=print_case)
    def test_derivs(self, case_name):
        prob = self.prob
        model = prob.model

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Engine.NUM_ENGINES: inputs.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Engine.NUM_WING_ENGINES: inputs.get_val(Aircraft.Engine.NUM_WING_ENGINES),
        }

        model.add_subsystem('main', MainGearLength(**options), promotes=['*'])
        model.add_subsystem('nose', NoseGearLength(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Engine.WING_LOCATIONS,
                Aircraft.Wing.DIHEDRAL,
                Aircraft.Wing.SPAN,
            ],
            output_keys=[
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
            ],
            version=Version.ALTERNATE,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
