import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized
import numpy as np

from aviary.subsystems.geometry.flops_based.landing_gear import MainGearLength, NoseGearLength
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
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

    def test_multiengine(self):
        options = {
            Aircraft.Engine.NUM_ENGINES: np.array([5, 2]),
            Aircraft.Engine.NUM_WING_ENGINES: np.array([4, 2]),
        }

        self.prob.model.add_subsystem('main', MainGearLength(**options), promotes=['*'])

        self.prob.setup(force_alloc_complex=True)  # complex step is great for checking partials

        # 2. Set the input variables
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 150.0, units='ft')
        self.prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 13.0, units='ft')
        self.prob.set_val(Aircraft.Wing.SPAN, 120.0, units='ft')
        self.prob.set_val(Aircraft.Wing.DIHEDRAL, 3.0, units='deg')
        self.prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([4.0, 9.0]), units='ft')
        self.prob.set_val(
            Aircraft.Engine.WING_LOCATIONS, np.array([0.2, 0.8, 0.5]), units='unitless'
        )

        self.prob.run_model()

        main_gear_length = self.prob.get_val(
            Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch'
        )[0]

        assert_near_equal(main_gear_length, 166.54100624, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
