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


class BWBWingTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
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
        prob.model.add_subsystem('wing', _BWBWing(), promotes_outputs=['*'], promotes_inputs=['*'])
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
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
        assert_near_equal(out1, exp1, tolerance=1e-9)


class BWBPrepGeomTest(unittest.TestCase):
    """Test computation of derived values of aircraft geometry for aerodynamics analysis."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Wing.DETAILED_WING, val=1, units='unitless')
        options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499, 0.7, 0.75, 0.8, 0.85, 0.8999, 0.95, 1],
            units='unitless',
        )

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('prep_geom', PrepGeom(), promotes=['*'])

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

        # BWBSimpleCabinLayout
        prob.set_val(Aircraft.Fuselage.LENGTH, 137.5, units='ft')
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 64.58, units='ft')
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 45.0, units='deg')
        prob.set_val(Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.11, units='unitless')
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
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 7.710195)
        # BWBFuselagePrelim
        # prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, 15.125)
        # BWBWingPrelim
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)

    def ttest_case1(self):
        prob = self.prob
        options = self.options

        prob.run_model()

        # BWBSimpleCabinLayout
        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 96.25, tolerance=1e-9)
        root_chord = prob.get_val(Aircraft.Wing.ROOT_CHORD)
        assert_near_equal(root_chord, 63.96019518, tolerance=1e-9)
        area_cabin = prob.get_val(Aircraft.Fuselage.CABIN_AREA)
        assert_near_equal(area_cabin, 5173.1872025, tolerance=1e-9)
        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 15.125, tolerance=1e-9)

        # BWBUpdateDetailedWingDist
        out0 = options.get_val(Aircraft.Wing.INPUT_STATION_DIST)
        exp0 = [
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
        ]
        assert_near_equal(out0, exp0, tolerance=1e-10)

        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [
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
        ]
        assert_near_equal(out1, exp1, tolerance=1e-9)

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
        assert_near_equal(out2, exp2, tolerance=1e-10)

        out3 = prob.get_val('BWB_LOAD_PATH_SWEEP_DIST')
        exp3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9]
        assert_near_equal(out3, exp3, tolerance=1e-10)
        # # BWBWingPrelim
        assert_near_equal(prob.get_val(Aircraft.Wing.AREA), 8668.64638424, tolerance=1e-10)
        assert_near_equal(
            prob.get_val(Aircraft.Wing.ASPECT_RATIO), 6.6313480248646242, tolerance=1e-10
        )
        assert_near_equal(
            prob.get_val(Aircraft.Wing.LOAD_FRACTION), 0.531071664997850196, tolerance=1e-10
        )


if __name__ == '__main__':
    # unittest.main()
    test = BWBPrepGeomTest()
    test.setUp()
    test.test_case1()
