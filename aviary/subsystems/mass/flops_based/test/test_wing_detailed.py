import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_detailed import (
    BWBDetailedWingBendingFact,
    DetailedWingBendingFact,
)
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings

omit_cases = ['LargeSingleAisle2FLOPS', 'LargeSingleAisle2FLOPSalt']
omit_cases.append('BWBsimpleFLOPS')
omit_cases.append('BWBdetailedFLOPS')


@use_tempdirs
class DetailedWingBendingTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    # Skip model that doesn't use detailed wing and BWB cases.
    @parameterized.expand(get_flops_case_names(omit=omit_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Engine.NUM_ENGINES: inputs.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Engine.NUM_WING_ENGINES: inputs.get_val(Aircraft.Engine.NUM_WING_ENGINES),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Aircraft.Wing.INPUT_STATION_DIST: inputs.get_val(Aircraft.Wing.INPUT_STATION_DIST),
            Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL: inputs.get_val(
                Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL
            ),
            Aircraft.Wing.NUM_INTEGRATION_STATIONS: inputs.get_val(
                Aircraft.Wing.NUM_INTEGRATION_STATIONS
            ),
        }

        self.prob.model.add_subsystem(
            'wing',
            DetailedWingBendingFact(**options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
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
                Aircraft.Wing.THICKNESS_TO_CHORD_REF,
            ],
            output_keys=[
                Aircraft.Wing.BENDING_MATERIAL_FACTOR,
                Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
            ],
            method='fd',
            atol=1e-3,
            rtol=1e-5,
        )

    def test_case_multiengine(self):
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
        aviary_options.set_val(Settings.VERBOSITY, 0)

        engine_options = AviaryValues()
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.DATA_FILE, 'models/engines/turbofan_28k.csv')
        engine_options.set_val(Settings.VERBOSITY, 0)
        engineModel1 = EngineDeck(options=engine_options)
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engineModel2 = EngineDeck(options=engine_options)
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 4)
        engineModel3 = EngineDeck(options=engine_options)

        preprocess_propulsion(aviary_options, [engineModel1, engineModel2, engineModel3])

        options = {
            Aircraft.Engine.NUM_ENGINES: aviary_options.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Engine.NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Engine.NUM_WING_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Aircraft.Wing.INPUT_STATION_DIST: aviary_options.get_val(
                Aircraft.Wing.INPUT_STATION_DIST
            ),
            Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL: aviary_options.get_val(
                Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL
            ),
            Aircraft.Wing.NUM_INTEGRATION_STATIONS: aviary_options.get_val(
                Aircraft.Wing.NUM_INTEGRATION_STATIONS
            ),
        }

        prob.model.add_subsystem(
            'detailed_wing', DetailedWingBendingFact(**options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        input_keys = [
            Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
            Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
            Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
            Mission.Design.GROSS_MASS,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.ASPECT_RATIO_REF,
            Aircraft.Wing.STRUT_BRACING_FACTOR,
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
            Aircraft.Wing.THICKNESS_TO_CHORD,
            Aircraft.Wing.THICKNESS_TO_CHORD_REF,
        ]

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

        BENDING_MATERIAL_FACTOR = prob.get_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR)
        pod_inertia = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)

        BENDING_MATERIAL_FACTOR_expected = 11.59165669761
        # 0.9600334354133278 if the factors are additive
        pod_inertia_expected = 0.9604608395586276
        assert_near_equal(
            BENDING_MATERIAL_FACTOR, BENDING_MATERIAL_FACTOR_expected, tolerance=1e-10
        )
        assert_near_equal(pod_inertia, pod_inertia_expected, tolerance=1e-10)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
        )
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)

    def test_case_fuselage_engines(self):
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')

        engine_options = AviaryValues()
        engine_options.set_val(Settings.VERBOSITY, 0)
        engine_options.set_val(Aircraft.Engine.DATA_FILE, 'models/engines/turbofan_28k.csv')
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 0)
        engine_options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 2)
        engineModel = EngineDeck(options=engine_options)

        preprocess_propulsion(aviary_options, [engineModel])

        options = {
            Aircraft.Engine.NUM_ENGINES: aviary_options.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Engine.NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Engine.NUM_WING_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Aircraft.Wing.INPUT_STATION_DIST: aviary_options.get_val(
                Aircraft.Wing.INPUT_STATION_DIST
            ),
            Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL: aviary_options.get_val(
                Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL
            ),
            Aircraft.Wing.NUM_INTEGRATION_STATIONS: aviary_options.get_val(
                Aircraft.Wing.NUM_INTEGRATION_STATIONS
            ),
        }

        prob.model.add_subsystem(
            'detailed_wing',
            DetailedWingBendingFact(**options),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        input_keys = [
            Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
            Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
            Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
            Mission.Design.GROSS_MASS,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.ASPECT_RATIO_REF,
            Aircraft.Wing.STRUT_BRACING_FACTOR,
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
            Aircraft.Wing.THICKNESS_TO_CHORD,
            Aircraft.Wing.THICKNESS_TO_CHORD_REF,
        ]

        for key in input_keys:
            val, units = aviary_options.get_item(key)
            prob.set_val(key, val, units)

        prob.set_val(Aircraft.Engine.POD_MASS, np.array([0]), units='lbm')

        wing_location = np.zeros(0)
        wing_location = np.append(wing_location, [0.0])

        prob.set_val(Aircraft.Engine.WING_LOCATIONS, wing_location)

        prob.run_model()

        BENDING_MATERIAL_FACTOR = prob.get_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR)
        pod_inertia = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)

        BENDING_MATERIAL_FACTOR_expected = 11.59165669761
        pod_inertia_expected = 0.84
        assert_near_equal(
            BENDING_MATERIAL_FACTOR, BENDING_MATERIAL_FACTOR_expected, tolerance=1e-10
        )
        assert_near_equal(pod_inertia, pod_inertia_expected, tolerance=1e-10)

    def test_case_fuselage_multiengine(self):
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')

        engine_options = AviaryValues()
        engine_options.set_val(Aircraft.Engine.DATA_FILE, 'models/engines/turbofan_28k.csv')
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 0)
        engineModel1 = EngineDeck(options=engine_options)

        engine_options.set_val(Aircraft.Engine.DATA_FILE, 'models/engines/turbofan_22k.csv')
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 0)
        engine_options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 2)
        engineModel2 = EngineDeck(options=engine_options)

        preprocess_propulsion(aviary_options, [engineModel1, engineModel2])

        options = {
            Aircraft.Engine.NUM_ENGINES: aviary_options.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Engine.NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Engine.NUM_WING_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Aircraft.Wing.INPUT_STATION_DIST: aviary_options.get_val(
                Aircraft.Wing.INPUT_STATION_DIST
            ),
            Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL: aviary_options.get_val(
                Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL
            ),
            Aircraft.Wing.NUM_INTEGRATION_STATIONS: aviary_options.get_val(
                Aircraft.Wing.NUM_INTEGRATION_STATIONS
            ),
        }

        prob.model.add_subsystem(
            'detailed_wing',
            DetailedWingBendingFact(**options),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        input_keys = [
            Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
            Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
            Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
            Mission.Design.GROSS_MASS,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.ASPECT_RATIO_REF,
            Aircraft.Wing.STRUT_BRACING_FACTOR,
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
            Aircraft.Wing.THICKNESS_TO_CHORD,
            Aircraft.Wing.THICKNESS_TO_CHORD_REF,
        ]

        for key in input_keys:
            val, units = aviary_options.get_item(key)
            prob.set_val(key, val, units)

        prob.set_val(Aircraft.Engine.POD_MASS, np.array([1130, 0]), units='lbm')

        wing_locations = np.zeros(0)
        wing_locations = np.append(wing_locations, [0.5])

        prob.set_val(Aircraft.Engine.WING_LOCATIONS, wing_locations)

        prob.run_model()

        BENDING_MATERIAL_FACTOR = prob.get_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR)
        pod_inertia = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)

        BENDING_MATERIAL_FACTOR_expected = 11.59165669761
        pod_inertia_expected = 0.84
        assert_near_equal(
            BENDING_MATERIAL_FACTOR, BENDING_MATERIAL_FACTOR_expected, tolerance=1e-10
        )
        assert_near_equal(pod_inertia, pod_inertia_expected, tolerance=1e-10)

    def test_extreme_engine_loc(self):
        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
        aviary_options.set_val(Settings.VERBOSITY, 0)

        engine_options = AviaryValues()
        engine_options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)
        engine_options.set_val(Aircraft.Engine.DATA_FILE, 'models/engines/turbofan_28k.csv')
        engine_options.set_val(Settings.VERBOSITY, 0)
        engineModel = EngineDeck(options=engine_options)

        prob = self.prob

        preprocess_propulsion(aviary_options, [engineModel])

        options = {
            Aircraft.Engine.NUM_ENGINES: aviary_options.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Engine.NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Engine.NUM_WING_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: aviary_options.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Aircraft.Wing.INPUT_STATION_DIST: aviary_options.get_val(
                Aircraft.Wing.INPUT_STATION_DIST
            ),
            Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL: aviary_options.get_val(
                Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL
            ),
            Aircraft.Wing.NUM_INTEGRATION_STATIONS: aviary_options.get_val(
                Aircraft.Wing.NUM_INTEGRATION_STATIONS
            ),
        }

        self.prob.model.add_subsystem(
            'wing',
            DetailedWingBendingFact(**options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.prob.setup(check=False)

        input_keys = [
            Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
            Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
            Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
            Mission.Design.GROSS_MASS,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.ASPECT_RATIO_REF,
            Aircraft.Wing.STRUT_BRACING_FACTOR,
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
            Aircraft.Wing.THICKNESS_TO_CHORD,
            Aircraft.Wing.THICKNESS_TO_CHORD_REF,
        ]

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


@use_tempdirs
class BWBSimpleWingBendingTest(unittest.TestCase):
    """
    The BWB detailed wing bending material factor when detailed wing data is not provided.
    Here, "simple wing" is relative simple.
    """

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [3], units='unitless')
        aviary_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, [0], units='unitless')
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 0, units='unitless')
        aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST, [0.0, 32.29, 1.0], units='unitless'
        )
        aviary_options.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0, units='unitless')
        aviary_options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 50, units='unitless')

        prob.model.add_subsystem(
            'fuselage',
            BWBDetailedWingBendingFact(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=874099.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 3.4488821, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO_REF, 3.4488821, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD, 0.11, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_REF, 0.11, units='unitless')

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.POD_MASS, np.array([0]), units='lbm')
        prob.set_val(Aircraft.Wing.SPAN, val=238.080049)

        wing_location = np.zeros(0)
        wing_location = np.append(wing_location, [0.0])
        prob.set_val(Aircraft.Engine.WING_LOCATIONS, wing_location)

        prob.set_val('BWB_CHORD_PER_SEMISPAN_DIST', [137.5, 91.3717, 14.2848], units='unitless')
        prob.set_val('BWB_THICKNESS_TO_CHORD_DIST', [0.11, 0.11, 0.11], units='unitless')
        prob.set_val('BWB_LOAD_PATH_SWEEP_DIST', [0.0, 15.337244816, 15.337244816], units='deg')

        prob.run_model()

        BENDING_MATERIAL_FACTOR = prob.get_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR)
        pod_inertia = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)

        BENDING_MATERIAL_FACTOR_expected = 2.68745091  # FLOPS BT = 2.6874568870727225
        pod_inertia_expected = 1.0
        assert_near_equal(BENDING_MATERIAL_FACTOR, BENDING_MATERIAL_FACTOR_expected, tolerance=1e-9)
        assert_near_equal(prob.get_val('calculated_wing_area'), 9165.70396358, tolerance=1e-9)
        # current BWB data set does not check the following
        assert_near_equal(pod_inertia, pod_inertia_expected, tolerance=1e-9)


@use_tempdirs
class BWBDetailedWingBendingTest(unittest.TestCase):
    """The BWB detailed wing bending material factor when detailed wing data is provided."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [3], units='unitless')
        aviary_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, [0], units='unitless')
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 0, units='unitless')
        aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [
                0,
                40.110378036763386,
                0.58970640947549269,
                0.62389754201920156,
                0.65808867456291043,
                0.6922798071066194,
                0.72647093965032838,
                0.76059368992894993,
                0.79485320473774623,
                0.82904433728145521,
                0.86323546982516419,
                0.89742660236887306,
                0.93154935264749461,
                0.96580886745629091,
                0.99999999999999989,
            ],
            units='unitless',
        )
        aviary_options.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0, units='unitless')
        aviary_options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 100, units='unitless')

        prob.model.add_subsystem(
            'fuselage',
            BWBDetailedWingBendingFact(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=874099, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 3.4488821, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO_REF, 3.4488821, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD, 0.11, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_REF, 0.11, units='unitless')

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.POD_MASS, np.array([0]), units='lbm')
        prob.set_val(Aircraft.Wing.SPAN, val=253.72075607352679)

        wing_location = np.zeros(0)
        wing_location = np.append(wing_location, [0.0])
        prob.set_val(Aircraft.Engine.WING_LOCATIONS, wing_location)

        prob.set_val(
            'BWB_CHORD_PER_SEMISPAN_DIST',
            [
                112.3001936860821,
                55,
                0.30710475250759373,
                0.26559671759953107,
                0.22682397329496512,
                0.19735121704228803,
                0.17348580652677914,
                0.15515935948335116,
                0.14503878425041333,
                0.13560203166834967,
                0.12602851455611114,
                0.11652337970896007,
                0.10701824486180898,
                0.0975131100146579,
                0.088007975167506816,
            ],
            units='unitless',
        )
        prob.set_val(
            'BWB_THICKNESS_TO_CHORD_DIST',
            [
                0.11,
                0.11,
                0.1132,
                0.0928,
                0.0822,
                0.0764,
                0.0742,
                0.0746,
                0.0758,
                0.0758,
                0.0756,
                0.0756,
                0.0758,
                0.076,
                0.076,
            ],
            units='unitless',
        )
        prob.set_val(
            'BWB_LOAD_PATH_SWEEP_DIST',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9],
            units='deg',
        )

        prob.run_model()

        BENDING_MATERIAL_FACTOR = prob.get_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR)
        pod_inertia = prob.get_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR)

        BENDING_MATERIAL_FACTOR_expected = 3.93931503  # FLOPS BT = 3.9724796254619563
        pod_inertia_expected = 1.0
        assert_near_equal(BENDING_MATERIAL_FACTOR, BENDING_MATERIAL_FACTOR_expected, tolerance=1e-9)
        assert_near_equal(prob.get_val('calculated_wing_area'), 5399.4057051, tolerance=1e-9)
        # current BWB data set does not check the following
        assert_near_equal(pod_inertia, pod_inertia_expected, tolerance=1e-9)


if __name__ == '__main__':
    unittest.main()
