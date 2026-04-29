import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.meta_model import MetaModelGroup
from aviary.variable_info.enums import FlapType
from aviary.variable_info.variables import Aircraft, Dynamic

"""
All data is from validation files using standalone flaps model
"""


class MetaModelTestCasePlain(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        options = {
            Aircraft.Wing.FLAP_TYPE: FlapType.PLAIN,
        }
        self.prob.model = MetaModelGroup(**options)
        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val('flap_defl_ratio', 40 / 60)
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val('flap_defl', 40.0, units='deg')
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)
        self.prob.set_val('slat_defl_ratio', 10 / 20)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89761)
        self.prob.set_val('reynolds', 164.78406)
        self.prob.set_val(Dynamic.Atmosphere.MACH, 0.18368)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89761)
        self.prob.set_val('body_to_span_ratio', 0.09239)
        self.prob.set_val('chord_to_body_ratio', 0.12679)

    def test_case(self):
        self.prob.run_model()
        tol = 1e-4

        expected_values = {
            'VDEL1': 1,
            'VDEL2': 0.55667,
            'VDEL3': 0.76500,
            'fus_lift': 0.05498,
            'VLAM1': 0.97217,
            'VLAM2': 1.09948,
            'VLAM3': 0.97217,
            'VLAM4': 1.19742,
            'VLAM5': 1,
            'VLAM6': 0.80000,
            'VLAM7': 0.73500,
            'VLAM10': 0.74000,
            'VLAM11': 0.84232,
            'VLAM13': 1.03209,
            'VLAM14': 0.99082,
        }

        for var_name, reg_data in expected_values.items():
            with self.subTest(var=var_name):
                ans = self.prob[var_name]
                assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(data, atol=1e-4, rtol=1e-4)


class MetaModelTestCaseSingleSlotted(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        options = {
            Aircraft.Wing.FLAP_TYPE: FlapType.SINGLE_SLOTTED,
        }
        self.prob.model = MetaModelGroup(**options)
        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val('flap_defl', 40.0, units='deg')

    def test_case(self):
        self.prob.run_model()
        tol = 1e-4

        expected_values = {
            'VDEL1': 1,
            'VLAM4': 1.25725,
            'VLAM5': 1,
            'VLAM6': 1.0,
        }

        for var_name, reg_data in expected_values.items():
            with self.subTest(var=var_name):
                ans = self.prob[var_name]
                assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(data, atol=1e-4, rtol=1e-4)


class MetaModelTestCaseFowler(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        options = {
            Aircraft.Wing.FLAP_TYPE: FlapType.FOWLER,
        }
        self.prob.model = MetaModelGroup(**options)
        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val('flap_defl', 40.0, units='deg')

    def test_case(self):
        self.prob.run_model()
        tol = 1e-4

        expected_values = {
            'VLAM5': 1.0,
            'VLAM6': 1.11,
        }

        for var_name, reg_data in expected_values.items():
            with self.subTest(var=var_name):
                ans = self.prob[var_name]
                assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(data, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
