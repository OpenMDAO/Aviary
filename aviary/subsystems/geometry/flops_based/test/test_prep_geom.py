import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.geometry.flops_based.canard import Canard
from aviary.subsystems.geometry.flops_based.characteristic_lengths import \
    CharacteristicLengths
from aviary.subsystems.geometry.flops_based.fuselage import FuselagePrelim
from aviary.subsystems.geometry.flops_based.nacelle import Nacelles
from aviary.subsystems.geometry.flops_based.prep_geom import (PrepGeom,
                                                              _Fuselage,
                                                              _Prelim, _Tail,
                                                              _Wing)
from aviary.subsystems.geometry.flops_based.utils import Names
from aviary.subsystems.geometry.flops_based.wing import WingPrelim
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (do_validation_test,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_data,
                                                      get_flops_inputs,
                                                      get_flops_outputs,
                                                      print_case)
from aviary.variable_info.functions import override_aviary_vars
from aviary.variable_info.variables import Aircraft

unit_data_sets = get_flops_case_names(
    only=['LargeSingleAisle2FLOPS', 'LargeSingleAisle2FLOPSdw', 'LargeSingleAisle2FLOPSalt', 'LargeSingleAisle1FLOPS'])
wetted_area_overide = get_flops_case_names(
    only=['N3CC', 'LargeSingleAisle2FLOPS', 'LargeSingleAisle2FLOPSdw', 'LargeSingleAisle2FLOPSalt', 'LargeSingleAisle1FLOPS'])


# TODO: We have no integration tests for canard, so canard-related names are commented
# out.
class PrepGeomTest(unittest.TestCase):
    """
    Test computation of derived values of aircraft geometry for aerodynamics analysis
    """

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        class PreMission(om.Group):

            def initialize(self):
                self.options.declare(
                    'aviary_options', types=AviaryValues,
                    desc='collection of Aircraft/Mission specific options')

            def setup(self):
                aviary_options = self.options['aviary_options']

                self.add_subsystem('prep_geom', PrepGeom(aviary_options=aviary_options),
                                   promotes=['*'])

            def configure(self):
                aviary_options = self.options['aviary_options']

                override_aviary_vars(self, aviary_options)

        options = get_flops_data(case_name, preprocess=True,
                                 keys=[
                                     Aircraft.Fuselage.NUM_FUSELAGES,
                                     Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
                                     Aircraft.VerticalTail.NUM_TAILS,
                                     Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                                     Aircraft.Engine.NUM_ENGINES,
                                     Aircraft.Propulsion.TOTAL_NUM_ENGINES,
                                 ])

        prob = self.prob

        prob.model.add_subsystem(
            'premission', PreMission(aviary_options=options), promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        output_keys = [Aircraft.Fuselage.AVG_DIAMETER,
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
                       Aircraft.Wing.FINENESS]

        if case_name not in wetted_area_overide:
            output_keys.extend([Aircraft.Canard.WETTED_AREA,
                                Aircraft.Design.TOTAL_WETTED_AREA,
                                Aircraft.Fuselage.WETTED_AREA,
                                Aircraft.HorizontalTail.WETTED_AREA,
                                Aircraft.Nacelle.TOTAL_WETTED_AREA,
                                Aircraft.Nacelle.WETTED_AREA,
                                Aircraft.VerticalTail.WETTED_AREA,
                                Aircraft.Wing.WETTED_AREA])

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.LENGTH,
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
                        Aircraft.Wing.WETTED_AREA_SCALER],
            output_keys=output_keys,
            aviary_option_keys=[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                                Aircraft.Fuselage.NUM_FUSELAGES],
            tol=1e-2,
            atol=1e-6,
            rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_prelim_fuselage(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', FuselagePrelim(),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Fuselage.LENGTH, 22.0)
        prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, 7.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 4.0)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_prelim_wing(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', WingPrelim(),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.AREA, 22.0)
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, 7.0)
        prob.set_val(Aircraft.Wing.SPAN, 4.0)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class _PrelimTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets,
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'prelim',
            _Prelim(aviary_options=get_flops_inputs(case_name,
                                                    keys=[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION])),
            promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(prob,
                           case_name,
                           input_validation_data=get_flops_data(case_name),
                           output_validation_data=local_variables[case_name],
                           input_keys=[Aircraft.Fuselage.AVG_DIAMETER,
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
                                       Aircraft.Wing.THICKNESS_TO_CHORD],
                           output_keys=[Names.CROOT,
                                        Names.CROOTB,
                                        Names.CROTM,
                                        Names.CROTVT,
                                        Names.CRTHTB,
                                        Names.SPANHT,
                                        Names.SPANVT,
                                        Names.XDX,
                                        Names.XMULT,
                                        Names.XMULTH,
                                        Names.XMULTV],
                           aviary_option_keys=Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                           tol=1e-3,
                           atol=1e-6,
                           rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class _WingTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets,
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'wings',
            _Wing(aviary_options=get_flops_inputs(case_name,
                                                  keys=[Aircraft.Fuselage.NUM_FUSELAGES])),
            promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(prob,
                           case_name,
                           input_validation_data=_hybrid_input_data(case_name),
                           output_validation_data=get_flops_outputs(case_name),
                           input_keys=[Names.CROOT,
                                       Names.CROOTB,
                                       Names.XDX,
                                       Names.XMULT,
                                       Aircraft.Wing.AREA,
                                       Aircraft.Wing.WETTED_AREA_SCALER],
                           output_keys=Aircraft.Wing.WETTED_AREA,
                           aviary_option_keys=Aircraft.Fuselage.NUM_FUSELAGES,
                           check_values=case_name not in wetted_area_overide,
                           tol=1e-3,
                           atol=1e-6,
                           rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class _TailTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets,
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'tails',
            _Tail(aviary_options=get_flops_inputs(case_name, preprocess=True,
                                                  keys=[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                                                        Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES])),
            promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(prob,
                           case_name,
                           input_validation_data=_hybrid_input_data(case_name),
                           output_validation_data=get_flops_outputs(case_name),
                           input_keys=[Names.XMULTH,
                                       Names.XMULTV,
                                       Aircraft.HorizontalTail.AREA,
                                       Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                                       Aircraft.HorizontalTail.WETTED_AREA_SCALER,
                                       Aircraft.VerticalTail.AREA,
                                       Aircraft.VerticalTail.WETTED_AREA_SCALER],
                           output_keys=[Aircraft.HorizontalTail.WETTED_AREA,
                                        Aircraft.VerticalTail.WETTED_AREA],
                           aviary_option_keys=Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                           check_values=case_name not in wetted_area_overide,
                           tol=1e-3,
                           atol=1e-6,
                           rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class _FuselageTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets,
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'fuse',
            _Fuselage(aviary_options=get_flops_inputs(case_name,
                                                      keys=[Aircraft.Fuselage.NUM_FUSELAGES])),
            promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        outputs = [
            Aircraft.Fuselage.CROSS_SECTION,
            Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
            Aircraft.Fuselage.LENGTH_TO_DIAMETER
        ]

        if case_name not in wetted_area_overide:
            outputs.extend(Aircraft.Fuselage.WETTED_AREA)

        do_validation_test(prob,
                           case_name,
                           input_validation_data=_hybrid_input_data(case_name),
                           output_validation_data=get_flops_outputs(case_name),
                           input_keys=[Names.CROOTB,
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
                                       Aircraft.Wing.THICKNESS_TO_CHORD],
                           output_keys=[Aircraft.HorizontalTail.WETTED_AREA,
                                        Aircraft.VerticalTail.WETTED_AREA],
                           aviary_option_keys=Aircraft.Fuselage.NUM_FUSELAGES,
                           check_values=case_name not in wetted_area_overide,
                           tol=1e-2,
                           atol=1e-6,
                           rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class NacellesTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets,
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        flops_inputs = get_flops_inputs(case_name, preprocess=True,
                                        keys=[Aircraft.Engine.NUM_ENGINES,
                                              Aircraft.Fuselage.NUM_FUSELAGES,
                                              ])

        prob.model.add_subsystem(
            'nacelles',
            Nacelles(aviary_options=flops_inputs),
            promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(prob,
                              case_name,
                              input_keys=[Aircraft.Nacelle.AVG_DIAMETER,
                                          Aircraft.Nacelle.AVG_LENGTH,
                                          Aircraft.Nacelle.WETTED_AREA_SCALER],
                              output_keys=[Aircraft.Nacelle.TOTAL_WETTED_AREA,
                                           Aircraft.Nacelle.WETTED_AREA],
                              aviary_option_keys=[Aircraft.Engine.NUM_ENGINES],
                              tol=1e-3,
                              atol=1e-6,
                              rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class CanardTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        prob = self.prob

        prob.model.add_subsystem(
            'canard',
            Canard(aviary_options=AviaryValues()),
            promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(prob,
                           'canard_test',
                           input_validation_data=Canard_test_data,
                           output_validation_data=Canard_test_data,
                           input_keys=[Aircraft.Canard.AREA,
                                       Aircraft.Canard.THICKNESS_TO_CHORD,
                                       Aircraft.Canard.WETTED_AREA_SCALER],
                           output_keys=Aircraft.Canard.WETTED_AREA,
                           aviary_option_keys=[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                                               Aircraft.Fuselage.NUM_FUSELAGES],
                           tol=1e-2,
                           atol=1e-6,
                           rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class CharacteristicLengthsTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(unit_data_sets,
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        flops_inputs = get_flops_inputs(case_name, preprocess=True,
                                        keys=[Aircraft.Engine.NUM_ENGINES,
                                              Aircraft.Fuselage.NUM_FUSELAGES,
                                              Aircraft.VerticalTail.NUM_TAILS,
                                              Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
                                              ])

        prob.model.add_subsystem(
            'characteristic_lengths',
            CharacteristicLengths(aviary_options=flops_inputs),
            promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(prob,
                           case_name,
                           input_validation_data=_hybrid_input_data(case_name),
                           output_validation_data=get_flops_outputs(case_name),
                           input_keys=[Names.CROOT,
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
                                       Aircraft.Wing.THICKNESS_TO_CHORD],
                           output_keys=[Aircraft.Canard.CHARACTERISTIC_LENGTH,
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
                                        Aircraft.Wing.FINENESS],
                           aviary_option_keys=[
                               Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, Aircraft.Propulsion.TOTAL_NUM_ENGINES],
                           tol=1e-2,
                           atol=1e-6,
                           rtol=1e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


def _hybrid_input_data(case_name):
    data = get_flops_inputs(case_name)
    data.update(local_variables[case_name])
    return data


_LargeSingleAisle_custom_test_data = AviaryValues({
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
})

_other_custom_test_data = AviaryValues({
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
})

local_variables = {}
local_variables['LargeSingleAisle1FLOPS'] = _LargeSingleAisle_custom_test_data
local_variables['LargeSingleAisle2FLOPS'] = _other_custom_test_data
local_variables['LargeSingleAisle2FLOPSdw'] = _other_custom_test_data
local_variables['LargeSingleAisle2FLOPSalt'] = _other_custom_test_data

# NOTE: no current data set includes canard
Canard_test_data = AviaryValues({
    Aircraft.Canard.AREA: (10.0, 'ft**2'),
    Aircraft.Canard.ASPECT_RATIO: (1.5, 'unitless'),
    Aircraft.Canard.CHARACTERISTIC_LENGTH: (2.58, 'ft'),
    Aircraft.Canard.FINENESS: (0.3000, 'unitless'),
    Aircraft.Canard.TAPER_RATIO: (0.5, 'unitless'),
    Aircraft.Canard.THICKNESS_TO_CHORD: (0.3, 'unitless'),
    Aircraft.Canard.WETTED_AREA: (21.16, 'ft**2'),
    Aircraft.Canard.WETTED_AREA_SCALER: (1.0, 'unitless'),
})


if __name__ == "__main__":
    unittest.main()
