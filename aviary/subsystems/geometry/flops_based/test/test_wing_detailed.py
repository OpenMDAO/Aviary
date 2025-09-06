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
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=137.5)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.11)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 7.710195)
        prob.run_model()

        import pdb

        pdb.set_trace()
        out0 = self.aviary_options.get_val(Aircraft.Wing.INPUT_STATION_DIST)
        exp0 = []
        assert_near_equal(out0, exp0, tolerance=1e-10)

        out1 = prob.get_val('BWB_CHORD_PER_SEMISPAN_DIST')
        exp1 = [
            137.5,
            11.014564285714286,
            0.32728011592742,
            0.28304519489247,
            0.24172526041667,
            0.2103162802419,
            0.18488302251344,
            0.1653526125672,
            0.1545671622984,
            0.14451045866935,
            0.1343080057124,
            0.12417842741935,
            0.11404884912634,
            0.10391927083333,
            0.09378969254032,
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


# From FLOPS
# ETAW = (0, 32.29, 0.96500437445319331, 0.96792067658209391, 0.97083697871099439, 0.97375328083989499, 0.97666958296879558, 0.97958005249343827, 0.98250218722659666, 0.98541848935549725, 0.98833479148439773, 0.99125109361329833, 0.99416156313794102, 0.99708369787109941, 1)
# CHD = (137.5, 11.014564549160855, 0.026194225721784779, 0.022653834937299507, 0.019346748323126276, 0.016832895888014, 0.014797317002041411, 0.013234179060950715, 0.012370953630796152, 0.011566054243219598, 0.010749489647127443, 0.0099387576552930883, 0.0091280256634587355, 0.008317293671624381, 0.0075065616797900274)
# TOC = (0.11, 0.11, 0.1132, 0.0928, 0.0822 0.0764, 0.0742, 0.0746, 0.0758, 0.0758, 0.0756, 0.0756, 0.0758, 0.076, 0.076)
# SWL = (0, 0, 0, 0, 0, 0, 0, 0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9)


if __name__ == '__main__':
    unittest.main()
