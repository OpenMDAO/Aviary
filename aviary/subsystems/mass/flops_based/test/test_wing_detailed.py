import unittest

import numpy as np
import openmdao.api as om

from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_detailed import \
    DetailedWingBendingFact
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission, Settings
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


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
            DetailedWingBendingFact(
                aviary_options=get_flops_inputs(case_name, preprocess=True)),
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

    def test_case_multiengine(self):
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')

        engine_options = AviaryValues()
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.DATA_FILE,
                               'models/engines/turbofan_28k.deck')
        engine_options.set_val(Settings.VERBOSITY, 0)
        engineModel1 = EngineDeck(options=engine_options)
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engineModel2 = EngineDeck(options=engine_options)
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 4)
        engineModel3 = EngineDeck(options=engine_options)

        preprocess_propulsion(aviary_options, [engineModel1, engineModel2, engineModel3])

        prob.model.add_subsystem('detailed_wing', DetailedWingBendingFact(
            aviary_options=aviary_options), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        input_keys = [Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
                      Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
                      Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
                      Mission.Design.GROSS_MASS,
                      Aircraft.Wing.ASPECT_RATIO,
                      Aircraft.Wing.ASPECT_RATIO_REF,
                      Aircraft.Wing.STRUT_BRACING_FACTOR,
                      Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                      Aircraft.Wing.THICKNESS_TO_CHORD,
                      Aircraft.Wing.THICKNESS_TO_CHORD_REF]

        for key in input_keys:
            val, units = aviary_options.get_item(key)
            prob.set_val(key, val, units)

        prob.set_val(Aircraft.Engine.POD_MASS, np.array([1130, 300, 845]), units='lbm')

        wing_locations = np.zeros(0)
        wing_locations = np.append(wing_locations, [0.5])
        wing_locations = np.append(wing_locations, [0.90])
        wing_locations = np.append(wing_locations, [0.2, 0.8])

        prob.set_val(Aircraft.Engine.WING_LOCATIONS, wing_locations)

        prob.run_model()

        bending_factor = prob.get_val(Aircraft.Wing.BENDING_FACTOR)
        pod_inertia = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)

        # manual computation of expected thrust reverser mass
        bending_factor_expected = 11.59165669761
        # 0.9600334354133278 if the factors are additive
        pod_inertia_expected = 0.9604608395586276
        assert_near_equal(bending_factor, bending_factor_expected, tolerance=1e-10)
        assert_near_equal(pod_inertia, pod_inertia_expected, tolerance=1e-10)

        partial_data = prob.check_partials(
            out_stream=None, compact_print=True, show_only_incorrect=True, form='central', method="fd")
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)

    def test_extreme_engine_loc(self):

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')

        engine_options = AviaryValues()
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.DATA_FILE,
                               'models/engines/turbofan_28k.deck')
        engine_options.set_val(Settings.VERBOSITY, 0)
        engineModel = EngineDeck(options=engine_options)

        prob = self.prob

        preprocess_propulsion(aviary_options, [engineModel])

        self.prob.model.add_subsystem(
            "wing",
            DetailedWingBendingFact(aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.prob.setup(check=False)

        input_keys = [Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
                      Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
                      Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
                      Mission.Design.GROSS_MASS,
                      Aircraft.Wing.ASPECT_RATIO,
                      Aircraft.Wing.ASPECT_RATIO_REF,
                      Aircraft.Wing.STRUT_BRACING_FACTOR,
                      Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                      Aircraft.Wing.THICKNESS_TO_CHORD,
                      Aircraft.Wing.THICKNESS_TO_CHORD_REF]

        for key in input_keys:
            val, units = aviary_options.get_item(key)
            prob.set_val(key, val, units)

        prob.set_val(Aircraft.Engine.POD_MASS, np.array([1130]), units='lbm')

        prob.set_val(Aircraft.Engine.WING_LOCATIONS, [1.0])

        prob.run_model()

        pod_inertia_factor = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)
        assert_near_equal(pod_inertia_factor, 0.84, tolerance=1e-10)

        prob.set_val(Aircraft.Engine.WING_LOCATIONS, [0.0])

        prob.run_model()

        pod_inertia_factor = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)
        assert_near_equal(pod_inertia_factor, 1.0, tolerance=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
    # test = DetailedWingBendingTest()
    # test.setUp()
    # test.test_case(case_name='LargeSingleAisle1FLOPS')
    # test.test_extreme_engine_loc()
