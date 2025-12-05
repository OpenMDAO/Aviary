import numpy as np
import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.flops_based.wing_detailed_bwb import (
    BWBComputeDetailedWingDist,
    BWBUpdateDetailedWingDist,
    BWBWingPrelim,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings


@use_tempdirs
class BWBUpdateDetailedWingDistTest(unittest.TestCase):
    """
    For BWB, test the updated detailed wing information when detailed wing information is given.
    """

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        options = self.aviary_options = AviaryValues()
        options.set_val(Settings.VERBOSITY, 1, units='unitless')
        options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499, 0.7, 0.75, 0.8, 0.85, 0.8999, 0.95, 1],
            units='unitless',
        )
        prob.model.add_subsystem(
            'dist', BWBUpdateDetailedWingDist(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, options)
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
            val=[0.0, 0, 0, 0, 0, 0, 0, 0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9],
        )
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=80.220756073526772)
        prob.set_val(Aircraft.Wing.SPAN, val=253.72075607352679)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=112.3001936860821)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 38.5)
        prob.run_model()

        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [
            112.300194,
            55.0000000,
            0.307104753,
            0.265596718,
            0.226823973,
            0.197351217,
            0.173485807,
            0.155159359,
            0.145038784,
            0.135602032,
            0.126028515,
            0.116523380,
            0.107018245,
            0.0975131100,
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
        assert_near_equal(out2, exp2, tolerance=1e-10)

        out3 = prob.get_val('BWB_LOAD_PATH_SWEEP_DIST')
        exp3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9]
        assert_near_equal(out3, exp3, tolerance=1e-10)

        # partial_data = self.prob.check_partials(out_stream=None, method='cs')
        # assert_check_partials(partial_data, atol=1e-9, rtol=1e-8)


@use_tempdirs
class BWBComputeDetailedWingDistTest(unittest.TestCase):
    """
    For BWB, test the updated detailed wing information when detailed wing information is not given.
    """

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        self.aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [
                0.0,
                0.5,
                1.0,
            ],  # always set [0, 0.5, 1] but actual value in the middle will be computed
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
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 63.96)
        prob.set_val(Aircraft.Wing.SWEEP, 35.7, units='deg')
        prob.run_model()

        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [137.5, 91.37142857, 14.2848]
        assert_near_equal(out1, exp1, tolerance=1e-10)

        out2 = prob.get_val('BWB_THICKNESS_TO_CHORD_DIST')
        exp2 = [0.11, 0.11, 0.11]
        assert_near_equal(out2, exp2, tolerance=1e-10)

        out3 = prob.get_val('BWB_LOAD_PATH_SWEEP_DIST')
        exp3 = [0.0, 15.33732285330093]
        assert_near_equal(out3, exp3, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


@use_tempdirs
class BWBWingPrelimTest(unittest.TestCase):
    """
    For BWB with given detailed wing information, test the computation of wing parameters.
    """

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        """Computed detailed wing case"""
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        self.aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.5, 1.0],
            units='unitless',
        )
        prob.model.add_subsystem(
            'dist', BWBWingPrelim(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58)
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.set_val(
            'BWB_CHORD_PER_SEMISPAN_DIST',
            val=[137.5, 91.37142857, 14.2848],
        )
        prob.run_model()

        assert_near_equal(prob.get_val(Aircraft.Wing.AREA), 16555.93625697, tolerance=1e-9)
        assert_near_equal(prob.get_val(Aircraft.Wing.ASPECT_RATIO), 3.44888827, tolerance=1e-9)
        assert_near_equal(prob.get_val(Aircraft.Wing.ASPECT_RATIO_REF), 3.44888827, tolerance=1e-9)
        assert_near_equal(
            prob.get_val(Aircraft.Wing.LOAD_FRACTION), 0.531071664997850196, tolerance=1e-9
        )

    def test_case2(self):
        """Provided detailed wing case"""
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        self.aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST,
            [0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499, 0.7, 0.75, 0.8, 0.85, 0.8999, 0.95, 1],
            units='unitless',
        )
        prob.model.add_subsystem(
            'dist', BWBWingPrelim(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=80.220756073526772)
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)
        prob.set_val(Aircraft.Wing.SPAN, val=253.720756)
        prob.set_val(
            'BWB_CHORD_PER_SEMISPAN_DIST',
            val=[
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
        )
        prob.run_model()

        assert_near_equal(prob.get_val(Aircraft.Wing.AREA), 12109.87971617, tolerance=1e-9)
        assert_near_equal(prob.get_val(Aircraft.Wing.ASPECT_RATIO), 5.36951675, tolerance=1e-9)
        assert_near_equal(prob.get_val(Aircraft.Wing.ASPECT_RATIO_REF), 5.36951675, tolerance=1e-9)
        assert_near_equal(
            prob.get_val(Aircraft.Wing.LOAD_FRACTION), 0.46761341784858923, tolerance=1e-9
        )


if __name__ == '__main__':
    unittest.main()
