import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.flops_based.bwb_wing_detailed import (
    BWBComputeDetailedWingDist,
    BWBUpdateDetailedWingDist,
    BWBWingPrelim,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings


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
            val=[0.0, 0, 0, 0, 0, 0, 0, 0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9],
        )
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58)
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=137.5)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 7.710195)
        prob.run_model()

        out0 = prob.get_val('BWB_INPUT_STATION_DIST')
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

        out0 = prob.get_val('BWB_INPUT_STATION_DIST')
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
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        self.aviary_options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 15, units='unitless')
        prob.model.add_subsystem(
            'dist', BWBWingPrelim(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58)
        prob.set_val(Aircraft.Wing.GLOVE_AND_BAT, val=121.05)
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
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
                11.014564549160855,
                0.32728011592741935,
                0.28304519489247315,
                0.24172526041666667,
                0.21031628024193549,
                0.18488302251344085,
                0.16535261256720429,
                0.1545671622983871,
                0.14451045866935486,
                0.13430800571236559,
                0.12417842741935484,
                0.11404884912634408,
                0.10391927083333334,
                0.093789692540322586,
            ],
        )
        prob.run_model()

        assert_near_equal(prob.get_val(Aircraft.Wing.AREA), 8668.64638424, tolerance=1e-10)
        assert_near_equal(
            prob.get_val(Aircraft.Wing.ASPECT_RATIO), 6.6313480248646242, tolerance=1e-10
        )
        assert_near_equal(
            prob.get_val(Aircraft.Wing.LOAD_FRACTION), 0.531071664997850196, tolerance=1e-10
        )


if __name__ == '__main__':
    unittest.main()
