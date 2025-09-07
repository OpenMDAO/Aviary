import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.flops_based.wing_detailed import (
    BWBUpdateDetailedWingDist,
    BWBComputeDetailedWingDist,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings


class BWBUpdateDetailedWingDistTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        self.aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499, 0.7, 0.75, 0.8, 0.85, 0.8999, 0.95, 1],
            units='unitless',
        )
        prob.model.add_subsystem(
            'dist', BWBUpdateDetailedWingDist(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
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
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58)
        prob.set_val(Aircraft.Wing.SPAN, val=68.58)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=137.5)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 7.710195)
        prob.run_model()

        out0 = self.aviary_options.get_val(Aircraft.Wing.INPUT_STATION_DIST)
        exp0 = [
            0.0,
            32.29,
            0.9650043744531933,
            0.9679206765820939,
            0.9708369787109944,
            0.973753280839895,
            0.9766695829687956,
            0.9795800524934383,
            0.9825021872265967,
            0.9854184893554973,
            0.9883347914843977,
            0.9912510936132983,
            0.994161563137941,
            0.9970836978710994,
            1.0,
        ]
        assert_near_equal(out0, exp0, tolerance=1e-10)

        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [
            137.5,
            11.014564285714286,
            0.02619422572178478,
            0.022653834937299507,
            0.019346748323126276,
            0.016832895888014,
            0.014797317002041411,
            0.013234179060950715,
            0.012370953630796152,
            0.011566054243219598,
            0.010749489647127443,
            0.009938757655293088,
            0.009128025663458736,
            0.008317293671624381,
            0.007506561679790027,
        ]
        assert_near_equal(out1, exp1, tolerance=1e-10)

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


class BWBComputeDetailedWingDistTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        self.aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.5, 1.0],
            units='unitless',
        )
        prob.model.add_subsystem(
            'dist', BWBComputeDetailedWingDist(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58)
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=137.5)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 7.710195)
        prob.set_val(Aircraft.Wing.SWEEP, 35.7, units='deg')
        prob.run_model()

        out0 = self.aviary_options.get_val(Aircraft.Wing.INPUT_STATION_DIST)
        exp0 = [0.0, 32.29, 1.0]
        assert_near_equal(out0, exp0, tolerance=1e-10)

        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [137.5, 11.01456429, 14.2848]
        assert_near_equal(out1, exp1, tolerance=1e-10)

        out2 = prob.get_val('BWB_THICKNESS_TO_CHORD_DIST')
        exp2 = [0.11, 0.11, 0.11]
        assert_near_equal(out2, exp2, tolerance=1e-10)

        out3 = prob.get_val('BWB_LOAD_PATH_SWEEP_DIST')
        exp3 = [0.0, 36.40586234, 36.40586234]
        assert_near_equal(out3, exp3, tolerance=1e-10)


if __name__ == '__main__':
    # unittest.main()
    test = BWBComputeDetailedWingDistTest()
    test.setUp()
    test.test_case1()
