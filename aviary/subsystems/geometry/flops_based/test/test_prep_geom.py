import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.geometry.flops_based.canard import Canard
from aviary.subsystems.geometry.flops_based.characteristic_lengths import (
    WingCharacteristicLength,
    OtherCharacteristicLengths,
)
from aviary.subsystems.geometry.flops_based.fuselage import FuselagePrelim
from aviary.subsystems.geometry.flops_based.nacelle import Nacelles
from aviary.subsystems.geometry.flops_based.prep_geom import (
    _BWBWing,
    PrepGeom,
    _Fuselage,
    _FuselageRatios,
    _Prelim,
    _Tail,
    _Wing,
)
from aviary.subsystems.geometry.flops_based.utils import Names
from aviary.subsystems.geometry.flops_based.wing import WingPrelim
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    do_validation_test,
    flops_validation_test,
    get_flops_case_names,
    get_flops_data,
    get_flops_inputs,
    get_flops_outputs,
    print_case,
)
from aviary.variable_info.functions import override_aviary_vars, setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft

unit_data_sets = get_flops_case_names(
    only=[
        'LargeSingleAisle2FLOPS',
        'LargeSingleAisle2FLOPSdw',
        'LargeSingleAisle2FLOPSalt',
        'LargeSingleAisle1FLOPS',
    ]
)
wetted_area_overide = get_flops_case_names(
    only=[
        'AdvancedSingleAisle',
        'LargeSingleAisle2FLOPS',
        'LargeSingleAisle2FLOPSdw',
        'LargeSingleAisle2FLOPSalt',
        'LargeSingleAisle1FLOPS',
    ]
)


# TODO: We have no integration tests for canard, so canard-related names are commented
# out.
@use_tempdirs
class PrepGeomTest(unittest.TestCase):
    """Test computation of derived values of aircraft geometry for aerodynamics analysis."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        class PreMission(om.Group):
            def initialize(self):
                self.options.declare(
                    'aviary_options',
                    types=AviaryValues,
                    desc='collection of Aircraft/Mission specific options',
                )

            def setup(self):
                self.add_subsystem('prep_geom', PrepGeom(), promotes=['*'])

            def configure(self):
                aviary_options = self.options['aviary_options']

                override_aviary_vars(self, aviary_options)

        keys = [
            Aircraft.Fuselage.NUM_FUSELAGES,
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
            Aircraft.VerticalTail.NUM_TAILS,
            Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
            Aircraft.Engine.NUM_ENGINES,
            Aircraft.Propulsion.TOTAL_NUM_ENGINES,
        ]

        options = get_flops_data(case_name, preprocess=True, keys=keys)
        model_options = {}
        for key in keys:
            model_options[key] = options.get_item(key)[0]

        prob = self.prob

        prob.model.add_subsystem('premission', PreMission(aviary_options=options), promotes=['*'])

        prob.model_options['*'] = model_options

        prob.setup(check=False, force_alloc_complex=True)

        output_keys = [
            Aircraft.Fuselage.AVG_DIAMETER,
            Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
            Aircraft.Fuselage.CROSS_SECTION,
            Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
            Aircraft.Fuselage.FINENESS,
            Aircraft.Fuselage.LENGTH_TO_DIAMETER,
            # Aircraft.Canard.CHARACTERISTIC_LENGTH,
            # Aircraft.Canard.FINENESS,
            Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
            Aircraft.HorizontalTail.FINENESS,
            Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
            Aircraft.Nacelle.FINENESS,
            Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
            Aircraft.VerticalTail.FINENESS,
            Aircraft.Wing.CHARACTERISTIC_LENGTH,
            Aircraft.Wing.FINENESS,
        ]

        if case_name not in wetted_area_overide:
            output_keys.extend(
                [
                    Aircraft.Canard.WETTED_AREA,
                    Aircraft.Design.TOTAL_WETTED_AREA,
                    Aircraft.Fuselage.WETTED_AREA,
                    Aircraft.HorizontalTail.WETTED_AREA,
                    Aircraft.Nacelle.TOTAL_WETTED_AREA,
                    Aircraft.Nacelle.WETTED_AREA,
                    Aircraft.VerticalTail.WETTED_AREA,
                    Aircraft.Wing.WETTED_AREA,
                ]
            )

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_HEIGHT,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Fuselage.WETTED_AREA_SCALER,
                # Aircraft.Canard.AREA,
                # Aircraft.Canard.ASPECT_RATIO,
                # Aircraft.Canard.THICKNESS_TO_CHORD,
                # Aircraft.Canard.WETTED_AREA_SCALER,
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.ASPECT_RATIO,
                Aircraft.HorizontalTail.TAPER_RATIO,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.HorizontalTail.WETTED_AREA_SCALER,
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Nacelle.AVG_LENGTH,
                # Aircraft.Nacelle.WETTED_AREA_SCALER,
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.ASPECT_RATIO,
                Aircraft.VerticalTail.TAPER_RATIO,
                Aircraft.VerticalTail.THICKNESS_TO_CHORD,
                Aircraft.VerticalTail.WETTED_AREA_SCALER,
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.GLOVE_AND_BAT,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD,
                Aircraft.Wing.WETTED_AREA_SCALER,
            ],
            output_keys=output_keys,
            aviary_option_keys=[
                Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                Aircraft.Fuselage.NUM_FUSELAGES,
            ],
            tol=1e-2,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_prelim_fuselage(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', FuselagePrelim(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Fuselage.LENGTH, 22.0)
        prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, 7.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 4.0)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_prelim_wing(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', WingPrelim(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.AREA, 22.0)
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, 7.0)
        prob.set_val(Aircraft.Wing.SPAN, 4.0)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class _PrelimTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets, name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        keys = [Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]
        flops_inputs = get_flops_inputs(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem('prelim', _Prelim(**options), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=get_flops_data(case_name),
            output_validation_data=local_variables[case_name],
            input_keys=[
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.ASPECT_RATIO,
                Aircraft.HorizontalTail.TAPER_RATIO,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.ASPECT_RATIO,
                Aircraft.VerticalTail.TAPER_RATIO,
                Aircraft.VerticalTail.THICKNESS_TO_CHORD,
                Aircraft.Wing.AREA,
                Aircraft.Wing.GLOVE_AND_BAT,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD,
            ],
            output_keys=[
                Names.CROOT,
                Names.CROOTB,
                Names.CROTM,
                Names.CROTVT,
                Names.CRTHTB,
                Names.SPANHT,
                Names.SPANVT,
                Names.XDX,
                Names.XMULT,
                Names.XMULTH,
                Names.XMULTV,
            ],
            aviary_option_keys=Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
            tol=1e-3,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class _WingTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets, name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        keys = [Aircraft.Fuselage.NUM_FUSELAGES]
        flops_inputs = get_flops_inputs(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem('wings', _Wing(**options), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=_hybrid_input_data(case_name),
            output_validation_data=get_flops_outputs(case_name),
            input_keys=[
                Names.CROOT,
                Names.CROOTB,
                Names.XDX,
                Names.XMULT,
                Aircraft.Wing.AREA,
                Aircraft.Wing.WETTED_AREA_SCALER,
            ],
            output_keys=Aircraft.Wing.WETTED_AREA,
            aviary_option_keys=Aircraft.Fuselage.NUM_FUSELAGES,
            check_values=case_name not in wetted_area_overide,
            tol=1e-3,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class _TailTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets, name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        keys = [
            Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
        ]
        flops_inputs = get_flops_data(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem('tails', _Tail(**options), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=_hybrid_input_data(case_name),
            output_validation_data=get_flops_outputs(case_name),
            input_keys=[
                Names.XMULTH,
                Names.XMULTV,
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.HorizontalTail.WETTED_AREA_SCALER,
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.WETTED_AREA_SCALER,
            ],
            output_keys=[Aircraft.HorizontalTail.WETTED_AREA, Aircraft.VerticalTail.WETTED_AREA],
            aviary_option_keys=Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
            check_values=case_name not in wetted_area_overide,
            tol=1e-3,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class _FuselageTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets, name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        keys = [Aircraft.Fuselage.NUM_FUSELAGES]
        flops_inputs = get_flops_inputs(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem('fuse', _Fuselage(**options), promotes=['*'])
        prob.model.add_subsystem('fuseratio', _FuselageRatios(), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        outputs = [
            Aircraft.Fuselage.CROSS_SECTION,
            Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
            Aircraft.Fuselage.LENGTH_TO_DIAMETER,
        ]

        if case_name not in wetted_area_overide:
            outputs.extend(Aircraft.Fuselage.WETTED_AREA)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=_hybrid_input_data(case_name),
            output_validation_data=get_flops_outputs(case_name),
            input_keys=[
                Names.CROOTB,
                Names.CROTVT,
                Names.CRTHTB,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.WETTED_AREA_SCALER,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.VerticalTail.THICKNESS_TO_CHORD,
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.GLOVE_AND_BAT,
                Aircraft.Wing.THICKNESS_TO_CHORD,
            ],
            output_keys=[Aircraft.HorizontalTail.WETTED_AREA, Aircraft.VerticalTail.WETTED_AREA],
            aviary_option_keys=Aircraft.Fuselage.NUM_FUSELAGES,
            check_values=case_name not in wetted_area_overide,
            tol=1e-2,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class NacellesTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets, name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        keys = [Aircraft.Engine.NUM_ENGINES]
        flops_inputs = get_flops_inputs(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem('nacelles', Nacelles(**options), promotes=['*'])

        setup_model_options(
            prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: (np.array([2]), 'unitless')})
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Nacelle.AVG_LENGTH,
                Aircraft.Nacelle.WETTED_AREA_SCALER,
            ],
            output_keys=[Aircraft.Nacelle.TOTAL_WETTED_AREA, Aircraft.Nacelle.WETTED_AREA],
            aviary_option_keys=[Aircraft.Engine.NUM_ENGINES],
            tol=1e-3,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class CanardTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        prob = self.prob

        prob.model.add_subsystem('canard', Canard(), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            'canard_test',
            input_validation_data=Canard_test_data,
            output_validation_data=Canard_test_data,
            input_keys=[
                Aircraft.Canard.AREA,
                Aircraft.Canard.THICKNESS_TO_CHORD,
                Aircraft.Canard.WETTED_AREA_SCALER,
            ],
            output_keys=Aircraft.Canard.WETTED_AREA,
            aviary_option_keys=[
                Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                Aircraft.Fuselage.NUM_FUSELAGES,
            ],
            tol=1e-2,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class CharacteristicLengthsTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets, name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        keys = [
            Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
        ]
        flops_inputs = get_flops_inputs(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem(
            'wing_characteristic_lengths', WingCharacteristicLength(**options), promotes=['*']
        )

        keys = [
            Aircraft.Engine.NUM_ENGINES,
        ]
        flops_inputs = get_flops_inputs(case_name, keys=keys)
        options = {}
        for key in keys:
            options[key] = flops_inputs.get_item(key)[0]

        prob.model.add_subsystem(
            'other_characteristic_lengths', OtherCharacteristicLengths(**options), promotes=['*']
        )

        setup_model_options(
            prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: (np.array([2]), 'unitless')})
        )

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=_hybrid_input_data(case_name),
            output_validation_data=get_flops_outputs(case_name),
            input_keys=[
                Names.CROOT,
                Aircraft.Canard.AREA,
                Aircraft.Canard.ASPECT_RATIO,
                Aircraft.Canard.THICKNESS_TO_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.ASPECT_RATIO,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Nacelle.AVG_LENGTH,
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.ASPECT_RATIO,
                Aircraft.VerticalTail.THICKNESS_TO_CHORD,
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.GLOVE_AND_BAT,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD,
            ],
            output_keys=[
                Aircraft.Canard.CHARACTERISTIC_LENGTH,
                Aircraft.Canard.FINENESS,
                Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
                Aircraft.Fuselage.FINENESS,
                Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
                Aircraft.HorizontalTail.FINENESS,
                Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
                Aircraft.Nacelle.FINENESS,
                Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
                Aircraft.VerticalTail.FINENESS,
                Aircraft.Wing.CHARACTERISTIC_LENGTH,
                Aircraft.Wing.FINENESS,
            ],
            aviary_option_keys=[
                Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                Aircraft.Propulsion.TOTAL_NUM_ENGINES,
            ],
            tol=1e-2,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


def _hybrid_input_data(case_name):
    data = get_flops_inputs(case_name)
    data.update(local_variables[case_name])
    return data


_LargeSingleAisle_custom_test_data = AviaryValues(
    {
        Names.CROOT: (16.415788, 'unitless'),
        Names.CROOTB: (15.133300, 'unitless'),
        Names.CROTM: (0.9218747347874056, 'unitless'),
        Names.CROTVT: (19.156603, 'unitless'),
        Names.CRTHTB: (13.502073, 'unitless'),
        Names.SPANHT: (46.151923036857305, 'unitless'),
        Names.SPANVT: (22.293496809607955, 'unitless'),
        Names.XDX: (12.750000, 'unitless'),
        Names.XMULT: (2.050310, 'unitless'),
        Names.XMULTH: (2.048375, 'unitless'),
        Names.XMULTV: (2.046247, 'unitless'),
    }
)

_other_custom_test_data = AviaryValues(
    {
        Names.CROOT: (19.254769, 'unitless'),
        Names.CROOTB: (17.601290, 'unitless'),
        Names.CROTM: (0.914126224442001, 'unitless'),
        Names.CROTVT: (18.672780, 'unitless'),
        Names.CRTHTB: (14.205434, 'unitless'),
        Names.SPANHT: (47.09069715015741, 'unitless'),
        Names.SPANVT: (25.166513637040453, 'unitless'),
        Names.XDX: (12.675400, 'unitless'),
        Names.XMULT: (2.050981, 'unitless'),
        Names.XMULTH: (2.046247, 'unitless'),
        Names.XMULTV: (2.053197, 'unitless'),
    }
)

local_variables = {}
local_variables['LargeSingleAisle1FLOPS'] = _LargeSingleAisle_custom_test_data
local_variables['LargeSingleAisle2FLOPS'] = _other_custom_test_data
local_variables['LargeSingleAisle2FLOPSdw'] = _other_custom_test_data
local_variables['LargeSingleAisle2FLOPSalt'] = _other_custom_test_data

# NOTE: no current data set includes canard
Canard_test_data = AviaryValues(
    {
        Aircraft.Canard.AREA: (10.0, 'ft**2'),
        Aircraft.Canard.ASPECT_RATIO: (1.5, 'unitless'),
        Aircraft.Canard.CHARACTERISTIC_LENGTH: (2.58, 'ft'),
        Aircraft.Canard.FINENESS: (0.3000, 'unitless'),
        Aircraft.Canard.TAPER_RATIO: (0.5, 'unitless'),
        Aircraft.Canard.THICKNESS_TO_CHORD: (0.3, 'unitless'),
        Aircraft.Canard.WETTED_AREA: (21.16, 'ft**2'),
        Aircraft.Canard.WETTED_AREA_SCALER: (1.0, 'unitless'),
    }
)


@use_tempdirs
class BWBWingTest(unittest.TestCase):
    "BWBWing computation test using detailed wing inputs"

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 15, units='unitless')
        prob.model.add_subsystem('wing', _BWBWing(), promotes_outputs=['*'], promotes_inputs=['*'])
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(
            'BWB_INPUT_STATION_DIST',
            [
                0.0,
                32.29,
                0.56275201612903225,
                0.59918934811827951,
                0.63562668010752688,
                0.67206401209677424,
                0.7085013440860215,
                0.74486580141129033,
                0.78137600806451613,
                0.81781334005376349,
                0.85425067204301075,
                0.89068800403225801,
                0.92705246135752695,
                0.96356266801075263,
                1.0,
            ],
            units='unitless',
        )
        prob.set_val(
            'BWB_CHORD_PER_SEMISPAN_DIST',
            val=[
                137.5,
                11.0145643,
                0.327280116,
                0.283045195,
                0.241725260,
                0.210316280,
                0.184883023,
                0.165352613,
                0.154567162,
                0.144510459,
                0.134308006,
                0.124178427,
                0.114048849,
                0.103919271,
                0.0937896925,
            ],
        )
        prob.set_val(
            'BWB_THICKNESS_TO_CHORD_DIST',
            val=[
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
        )
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58)
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.run_model()

        out1 = prob.get_val(Aircraft.Wing.WETTED_AREA)
        exp1 = 17683.7562096
        assert_near_equal(out1, exp1, tolerance=1e-8)


@use_tempdirs
class BWBSimplePrepGeomTest(unittest.TestCase):
    """
    Test computation of derived values of aircraft geometry for aerodynamics analysis.
    In this test, we assume detailed wing is computed instead of provided and we assume
    fuselage has a simple layout. Note option Aircraft.Engine.NUM_ENGINES is updated.
    """

    def setUp(self):
        options = self.options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, val=True, units='unitless')
        options.set_val(Aircraft.BWB.DETAILED_WING_PROVIDED, val=False, units='unitless')
        options.set_val(Aircraft.Wing.INPUT_STATION_DIST, [0.0, 0.5, 1.0], units='unitless')
        options.set_val(Aircraft.BWB.MAX_NUM_BAYS, 0, units='unitless')
        options.set_val(Aircraft.BWB.NUM_BAYS, [2], units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 3, units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([3]), units='unitless')
        options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 3, units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('prep_geom', PrepGeom(), promotes=['*'])

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

        # BWBSimpleCabinLayout
        prob.set_val(Aircraft.Fuselage.LENGTH, 137.5, units='ft')
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 64.58, units='ft')
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 45.0, units='deg')
        prob.set_val(Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.11, units='unitless')
        prob.set_val('Rear_spar_percent_chord', 0.7, units='unitless')
        # BWBComputeDetailedWingDist
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.SWEEP, val=35.7)
        # BWBFuselagePrelim
        # skip
        # BWBWingPrelim
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)
        # _Prelim
        prob.set_val(Aircraft.HorizontalTail.AREA, val=0.0)
        prob.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, val=0.0)
        prob.set_val(Aircraft.HorizontalTail.TAPER_RATIO, val=0)
        prob.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.VerticalTail.AREA, val=0.0)
        prob.set_val(Aircraft.VerticalTail.ASPECT_RATIO, val=1.88925)
        prob.set_val(Aircraft.VerticalTail.TAPER_RATIO, val=0.0)
        prob.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, val=0.311)
        # _BWBWing
        # skip
        # _Tail
        prob.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0.311)
        # _BWBFuselage
        # skip
        # _FuselageRatios
        # Nacelles
        # DNAC = 12.608, XNAC = 17.433 originally. It is then scaled down by
        # SQRT(ESCALE) = sqrt(0.80963)
        # DNAC = 12.608 * sqrt(0.80963) = 11.3446080595
        # XNAC = 17.433 * sqrt(0.80963) = 15.6861161407
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, val=11.3446080595)
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH, val=15.6861161407)
        prob.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, val=1.0)
        # Canard
        prob.set_val(Aircraft.Canard.AREA, val=0.0)
        prob.set_val(Aircraft.Canard.THICKNESS_TO_CHORD, val=0.0)
        prob.set_val(Aircraft.Canard.WETTED_AREA_SCALER, val=0.0)
        # BWBWingCharacteristicLength
        # OtherCharacteristicLengths
        # TotalWettedArea
        # TotalWettedArea

    def test_case1(self):
        """
        Testing FLOPS data case:
        Aircraft.BWB.NUM_BAYS -- NBAY = 5
        Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH -- XLP = 96.25
        Aircraft.Wing.ROOT_CHORD -- XLW = 63.960195184412598
        Aircraft.Fuselage.CABIN_AREA -- ACABIN = 5173.1872025046832
        Aircraft.Fuselage.MAX_HEIGHT -- DF = 15.125
        Aircraft.Wing.INPUT_STATION_DIST -- ETAW = [0, 32.29, 1]
        BWB_CHORD_PER_SEMISPAN_DIST -- CHD = [137.5, 91.371707406303713, 14.284802944808163]
        BWB_THICKNESS_TO_CHORD_DIST -- TOC = [0.11, 0.11, 0.11]
        BWB_LOAD_PATH_SWEEP_DIST -- SWL = [0, 15.337244816188816, 15.337244816188816]
        Aircraft.Fuselage.AVG_DIAMETER -- XD = 39.8525
        Aircraft.Fuselage.PLANFORM_AREA -- FPAREA = 7390.267432149546
        Aircraft.Wing.AREA -- XW = 16555.972297926455
        Aircraft.Wing.ASPECT_RATIO -- AR = 3.4488821268812084
        Aircraft.Wing.LOAD_FRACTION -- PCTL = 0.53107174649913569
        prelim.prep_geom:_Names:CROOT -- CROOT = 105.31056813183183
        prelim.prep_geom:_Names:CROOTB -- CROOTB = 93.164834715181058
        prelim.prep_geom:_Names:CROTM -- CROTM = 0.88466747799284229
        prelim.prep_geom:_Names:CROTVT -- CROTVT = 0
        prelim.prep_geom:_Names:CRTHTB -- CRTHTB = 0
        prelim.prep_geom:_Names:SPANHT -- SPANHT = 0
        prelim.prep_geom:_Names:SPANVT -- SPANVT = 0
        prelim.prep_geom:_Names:XDX -- XDX = 39.8525
        prelim.prep_geom:_Names:XMULT -- XMULT = 2.04257
        prelim.prep_geom:_Names:XMULTH -- XMULTH = 2.04257
        prelim.prep_geom:_Names:XMULTV -- XMULTV = 2.04257
        Aircraft.Wing.WETTED_AREA -- SWET(1) = 33816.73
        Aircraft.HorizontalTail.WETTED_AREA -- HORIZONTAL=  0.0
        Aircraft.VerticalTail.WETTED_AREA -- VERTICAL TAIL SWET = 0.0
        Aircraft.Fuselage.DIAMETER_TO_WING_SPAN -- DB = 0.16739117852998228
        Aircraft.Fuselage.LENGTH_TO_DIAMETER -- BODYLD = 3.4502226961922089
        Aircraft.Nacelle.TOTAL_WETTED_AREA -- SWTNA = 498.27*3
        Aircraft.Nacelle.WETTED_AREA -- SWET(5)-SWET(7) = 498.26795086
        Aircraft.Wing.CHARACTERISTIC_LENGTH -- EL(1) = 69.539519845923053
        Aircraft.Fuselage.CHARACTERISTIC_LENGTH -- EL(4) = 137.5
        Aircraft.Nacelle.CHARACTERISTIC_LENGTH -- EL(5)-EL(7) = 15.686120387590995
        Aircraft.Design.TOTAL_WETTED_AREA -- SWTWG + SWTHT + SWTVT + SWTFU + SWTNA + SWTCN = 35311.536998566495
        """
        prob = self.prob

        prob.run_model()

        # BWBSimpleCabinLayout
        num_bays = prob.get_val(Aircraft.BWB.NUM_BAYS)
        assert_near_equal(num_bays, [5], tolerance=1e-9)
        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 96.25, tolerance=1e-8)
        root_chord = prob.get_val(Aircraft.Wing.ROOT_CHORD)
        assert_near_equal(root_chord, 63.96019518, tolerance=1e-8)
        area_cabin = prob.get_val(Aircraft.Fuselage.CABIN_AREA)
        assert_near_equal(area_cabin, 5173.1872025, tolerance=1e-8)
        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 15.125, tolerance=1e-8)

        # BWBComputeDetailedWingDist
        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [137.5, 91.37170741, 14.2848]
        assert_near_equal(out1, exp1, tolerance=1e-8)

        out2 = prob.get_val('BWB_THICKNESS_TO_CHORD_DIST')
        exp2 = [0.11, 0.11, 0.11]
        assert_near_equal(out2, exp2, tolerance=1e-8)

        out3 = prob.get_val('BWB_LOAD_PATH_SWEEP_DIST')
        exp3 = [0.0, 15.33723721, 15.33723721]
        assert_near_equal(out3, exp3, tolerance=1e-8)

        # BWBFuselagePrelim
        assert_near_equal(prob.get_val(Aircraft.Fuselage.AVG_DIAMETER), 39.8525, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.PLANFORM_AREA), 7390.26743215, tolerance=1e-8
        )

        # BWBWingPrelim
        assert_near_equal(prob.get_val(Aircraft.Wing.AREA), 16555.96944965, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Wing.ASPECT_RATIO),
            3.4488813,
            tolerance=1e-8,
        )
        assert_near_equal(
            prob.get_val(Aircraft.Wing.LOAD_FRACTION),
            0.531071664997850196,
            tolerance=1e-8,
        )
        # _Prelim
        assert_near_equal(
            prob.get_val('prelim.prep_geom:_Names:CROOT'), 105.310571594, tolerance=1e-8
        )
        assert_near_equal(
            prob.get_val('prelim.prep_geom:_Names:CROOTB'), 93.16483527, tolerance=1e-8
        )
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CROTM'), 0.88466745, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CROTVT'), 0.0, tolerance=1e-7)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CRTHTB'), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:SPANHT'), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:SPANVT'), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XDX'), 39.8525, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XMULT'), 2.04257, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XMULTH'), 2.04257, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XMULTV'), 2.04257, tolerance=1e-8)
        # _BWBWing
        assert_near_equal(prob.get_val(Aircraft.Wing.WETTED_AREA), 33816.72651876, tolerance=1e-8)
        # _Tail
        assert_near_equal(prob.get_val(Aircraft.HorizontalTail.WETTED_AREA), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val(Aircraft.VerticalTail.WETTED_AREA), 0.0, tolerance=1e-8)
        # _BWBFuselage
        # _FuselageRatios
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN), 0.167391212, tolerance=1e-8
        )
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER), 3.4502227, tolerance=1e-8
        )
        # Nacelles
        assert_near_equal(
            prob.get_val(Aircraft.Nacelle.TOTAL_WETTED_AREA), 1494.80385257, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Nacelle.WETTED_AREA), 498.26795086, tolerance=1e-8)
        # Canard
        assert_near_equal(prob.get_val(Aircraft.Canard.WETTED_AREA), 0.0, tolerance=1e-8)
        # BWBWingCharacteristicLength
        assert_near_equal(
            prob.get_val(Aircraft.Wing.CHARACTERISTIC_LENGTH), 69.53952222, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Wing.FINENESS), 0.11, tolerance=1e-8)
        # OtherCharacteristicLengths
        assert_near_equal(prob.get_val(Aircraft.Canard.CHARACTERISTIC_LENGTH), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val(Aircraft.Canard.FINENESS), 0.0, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH), 137.5, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Fuselage.FINENESS), 3.4502227, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH), 0.0, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.HorizontalTail.FINENESS), 0.11, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH), 15.68611614, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Nacelle.FINENESS), 1.38269353, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH), 0.0, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.VerticalTail.FINENESS), 0.11, tolerance=1e-8)
        # TotalWettedArea
        assert_near_equal(
            prob.get_val(Aircraft.Design.TOTAL_WETTED_AREA), 35311.53037134, tolerance=1e-8
        )


@use_tempdirs
class BWBDetailedPrepGeomTest(unittest.TestCase):
    """
    Test computation of derived values of aircraft geometry for aerodynamics analysis.
    In this test, we assume detailed wing is provided and we assume fuselage has a detailed layout.
    Note options Aircraft.Engine.NUM_ENGINES and Aircraft.Wing.INPUT_STATION_DIST are updated.
    """

    def setUp(self):
        options = self.options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 100, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 28, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 340, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 4, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 4, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS, 39.0, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 61.0, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 32.0, units='inch')
        options.set_val(Aircraft.BWB.MAX_NUM_BAYS, 0, units='unitless')
        options.set_val(Aircraft.BWB.NUM_BAYS, [2], units='unitless')
        options.set_val(Aircraft.Wing.DETAILED_WING, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499, 0.7, 0.75, 0.8, 0.85, 0.8999, 0.95, 1],
            units='unitless',
        )
        options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 15, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 3, units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([3]), units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('prep_geom', PrepGeom(), promotes=['*'])

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

        # BWBDetailedCabinLayout
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, val=45.0, units='deg')
        prob.set_val(Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, val=0.11, units='unitless')
        prob.set_val('Rear_spar_percent_chord', val=0.7, units='unitless')
        # BWBUpdateDetailedWingDist
        prob.set_val(
            Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
            val=[
                -1.0,
                58.03,
                0.4491,
                0.3884,
                0.3317,
                0.2886,
                0.2537,
                0.2269,
                0.2121,
                0.1983,
                0.1843,
                0.1704,
                0.1565,
                0.1426,
                0.1287,
            ],
        )
        prob.set_val(
            Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
            val=[
                -1.0,
                0.15,
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
        )
        prob.set_val(
            Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
            val=[0.0, 0, 0, 0, 0, 0, 0, 0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9],
        )
        prob.set_val(Aircraft.Wing.SPAN, val=253.720756)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        # BWBFuselagePrelim
        # skip
        # BWBWingPrelim
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)
        # _Prelim
        prob.set_val(Aircraft.HorizontalTail.AREA, val=0.0)
        prob.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, val=0.0)
        prob.set_val(Aircraft.HorizontalTail.TAPER_RATIO, val=0)
        prob.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.VerticalTail.AREA, val=0.0)
        prob.set_val(Aircraft.VerticalTail.ASPECT_RATIO, val=1.88925)
        prob.set_val(Aircraft.VerticalTail.TAPER_RATIO, val=0.0)
        prob.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, val=0.311)
        # _BWBWing
        # skip
        # _Tail
        prob.set_val(Aircraft.HorizontalTail.AREA, val=0.0)
        prob.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0.311)
        prob.set_val(Aircraft.HorizontalTail.WETTED_AREA_SCALER, val=1.0)
        prob.set_val(Aircraft.VerticalTail.AREA, val=0.0)
        prob.set_val(Aircraft.VerticalTail.WETTED_AREA_SCALER, val=1.0)
        # _BWBFuselage
        # skip
        # _FuselageRatios
        # Nacelles
        # DNAC = 12.608, XNAC = 17.433 originally. It is then scaled down by
        # SQRT(ESCALE) = sqrt(0.80963)
        # DNAC = 12.608 * sqrt(0.80963) = 11.3446080595
        # XNAC = 17.433 * sqrt(0.80963) = 15.6861161407
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, val=11.3446080595)
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH, val=15.6861161407)
        prob.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, val=1.0)
        # Canard
        prob.set_val(Aircraft.Canard.AREA, val=0.0)
        prob.set_val(Aircraft.Canard.THICKNESS_TO_CHORD, val=0.0)
        prob.set_val(Aircraft.Canard.WETTED_AREA_SCALER, val=0.0)
        # BWBWingCharacteristicLength
        # OtherCharacteristicLengths
        # TotalWettedArea
        # TotalWettedArea

    def test_case1(self):
        """
        Testing FLOPS data case:
        Aircraft.BWB.NUM_BAYS -- NBAY = 7
        Aircraft.Fuselage.LENGTH -- XL = 112.3
        Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH -- XLP = 78.61
        Aircraft.Fuselage.MAX_WIDTH -- WF = 80.22
        Aircraft.Fuselage.MAX_HEIGHT -- DF = 12.35
        Aircraft.Fuselage.CABIN_AREA -- ACABIN = 4697.33
        Aircraft.Wing.ROOT_CHORD -- XLW = 38.50
        Aircraft.Fuselage.AVG_DIAMETER -- XD = 46.2868886894979
        Aircraft.Fuselage.PLANFORM_AREA -- FPAREA = 6710.4740143724875
        Aircraft.Wing.AREA -- SW = 12109.9
        Aircraft.Wing.ASPECT_RATIO -- AR = 5.370
        Aircraft.Wing.LOAD_FRACTION -- PCTL = 0.4676
        prelim.prep_geom:_Names:CROOT -- CROOT = 72.085530500256112
        prelim.prep_geom:_Names:CROOTB -- CROOTB = 63.024672729456938
        prelim.prep_geom:_Names:CROTM -- CROTM = 0.87430407034644797
        prelim.prep_geom:_Names:CROTVT -- CROTVT = 0
        prelim.prep_geom:_Names:CRTHTB -- CRTHTB = 0
        prelim.prep_geom:_Names:SPANHT -- SPANHT = 0
        prelim.prep_geom:_Names:SPANVT -- SPANVT = 0
        prelim.prep_geom:_Names:XDX -- XDX = 46.2868886894979
        prelim.prep_geom:_Names:XMULT -- XMULT = 2.04257
        prelim.prep_geom:_Names:XMULTH -- XMULTH = 2.04257
        prelim.prep_geom:_Names:XMULTV -- XMULTV = 2.04257
        Aircraft.Wing.WETTED_AREA -- SWET(1) = 24713.661297561481
        Aircraft.HorizontalTail.WETTED_AREA -- HORIZONTAL TAIL SWET = 0.0
        Aircraft.VerticalTail.WETTED_AREA -- VERTICAL TAIL SWET = 0.0
        Aircraft.Fuselage.DIAMETER_TO_WING_SPAN -- DB = 0.18243240878599712
        Aircraft.Fuselage.LENGTH_TO_DIAMETER -- BODYLD = 2.4261771932742167
        Aircraft.Nacelle.TOTAL_WETTED_AREA -- SWTNA = 498.27*3
        Aircraft.Nacelle.WETTED_AREA -- SWET(5)-SWET(7) = 498.26822066361945
        Aircraft.Wing.CHARACTERISTIC_LENGTH -- EL(1) = 47.729164562158907
        Aircraft.Fuselage.CHARACTERISTIC_LENGTH -- EL(4) = 112.3001936860821
        Aircraft.Nacelle.CHARACTERISTIC_LENGTH -- EL(5)-EL(7) = 15.686120387590995
        """
        prob = self.prob
        options = self.options

        prob.run_model()

        # BWBDetailedCabinLayout
        num_bays = prob.get_val(Aircraft.BWB.NUM_BAYS)
        assert_near_equal(num_bays, [7], tolerance=1e-9)
        fuselage_length = prob.get_val(Aircraft.Fuselage.LENGTH)
        assert_near_equal(fuselage_length, 112.30019369, tolerance=1e-9)
        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 78.61013558, tolerance=1e-9)
        fuselage_width = prob.get_val(Aircraft.Fuselage.MAX_WIDTH)
        assert_near_equal(fuselage_width, 80.22075607, tolerance=1e-9)
        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 12.35302131, tolerance=1e-9)
        cabin_area = prob.get_val(Aircraft.Fuselage.CABIN_AREA)
        assert_near_equal(cabin_area, 4697.33181006, tolerance=1e-9)
        root_chord = prob.get_val(Aircraft.Wing.ROOT_CHORD)
        assert_near_equal(root_chord, 38.5, tolerance=1e-9)

        # BWBUpdateDetailedWingDist
        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [
            112.3001936861,
            55.0,
            0.3071047525,
            0.2655967176,
            0.2268239733,
            0.197351217,
            0.1734858065,
            0.1551593595,
            0.1450387842,
            0.1356020317,
            0.1260285145,
            0.1165233797,
            0.1070182448,
            0.09751311,
            0.0880079752,
        ]
        assert_near_equal(out1, exp1, tolerance=1e-8)

        out2 = prob.get_val('BWB_THICKNESS_TO_CHORD_DIST')
        exp2 = [
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
        ]
        assert_near_equal(out2, exp2, tolerance=1e-8)

        out3 = prob.get_val('BWB_LOAD_PATH_SWEEP_DIST')
        exp3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9]
        assert_near_equal(out3, exp3, tolerance=1e-8)

        # BWBFuselagePrelim
        assert_near_equal(prob.get_val(Aircraft.Fuselage.AVG_DIAMETER), 46.28688869, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.PLANFORM_AREA), 6710.47401437, tolerance=1e-8
        )

        # BWBWingPrelim
        assert_near_equal(prob.get_val(Aircraft.Wing.AREA), 12109.8797157, tolerance=1e-8)
        assert_near_equal(prob.get_val(Aircraft.Wing.ASPECT_RATIO), 5.36951675, tolerance=1e-8)
        assert_near_equal(prob.get_val(Aircraft.Wing.LOAD_FRACTION), 0.46761342, tolerance=1e-8)
        # _Prelim
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CROOT'), 72.0855305, tolerance=1e-8)
        assert_near_equal(
            prob.get_val('prelim.prep_geom:_Names:CROOTB'), 63.02467273, tolerance=1e-8
        )
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CROTM'), 0.87430407, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CROTVT'), 0.0, tolerance=1e-7)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:CRTHTB'), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:SPANHT'), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:SPANVT'), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XDX'), 46.28688869, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XMULT'), 2.04257, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XMULTH'), 2.04257, tolerance=1e-8)
        assert_near_equal(prob.get_val('prelim.prep_geom:_Names:XMULTV'), 2.04257, tolerance=1e-8)
        # _BWBWing
        assert_near_equal(prob.get_val(Aircraft.Wing.WETTED_AREA), 24713.66128988, tolerance=1e-8)
        # _Tail
        assert_near_equal(prob.get_val(Aircraft.HorizontalTail.WETTED_AREA), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val(Aircraft.VerticalTail.WETTED_AREA), 0.0, tolerance=1e-8)
        # _BWBFuselage
        # skip
        # _FuselageRatios
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN),
            0.18243241,
            tolerance=1e-8,
        )
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER), 2.42617719, tolerance=1e-8
        )
        # Nacelles
        assert_near_equal(
            prob.get_val(Aircraft.Nacelle.TOTAL_WETTED_AREA), 1494.80385258, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Nacelle.WETTED_AREA), 498.26795086, tolerance=1e-8)
        # Canard
        assert_near_equal(prob.get_val(Aircraft.Canard.WETTED_AREA), 0.0, tolerance=1e-8)
        # BWBWingCharacteristicLength
        assert_near_equal(
            prob.get_val(Aircraft.Wing.CHARACTERISTIC_LENGTH), 47.72916456, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Wing.FINENESS), 0.11, tolerance=1e-8)
        # OtherCharacteristicLengths
        assert_near_equal(prob.get_val(Aircraft.Canard.CHARACTERISTIC_LENGTH), 0.0, tolerance=1e-8)
        assert_near_equal(prob.get_val(Aircraft.Canard.FINENESS), 0.0, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH), 112.30019369, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Fuselage.FINENESS), 2.42617719, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH), 0.0, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.HorizontalTail.FINENESS), 0.11, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH), 15.68611614, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.Nacelle.FINENESS), 1.38269353, tolerance=1e-8)
        assert_near_equal(
            prob.get_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH), 0.0, tolerance=1e-8
        )
        assert_near_equal(prob.get_val(Aircraft.VerticalTail.FINENESS), 0.11, tolerance=1e-8)
        # TotalWettedArea
        assert_near_equal(
            prob.get_val(Aircraft.Design.TOTAL_WETTED_AREA), 26208.46514246, tolerance=1e-8
        )


if __name__ == '__main__':
    unittest.main()
