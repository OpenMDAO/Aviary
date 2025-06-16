import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.empennage import EmpennageSize, TailSize, TailVolCoef
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft

tol = 5e-4
partial_tols = {'atol': 1e-8, 'rtol': 1e-8}


class TestTailVolCoef(
    unittest.TestCase
):  # note, this component is not used in the large single aisle
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'hvc',
            TailVolCoef(orientation='horizontal'),
            promotes=['aircraft:*'],
        )
        self.prob.model.add_subsystem(
            'vvc',
            TailVolCoef(orientation='vertical'),
            promotes=['aircraft:*'],
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=12.615, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8054, units='ft')
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_large_sinle_aisle_1_volcoefs(self):
        self.prob.run_model()

        # These are not actual GASP values.
        assert_near_equal(self.prob[Aircraft.HorizontalTail.VOLUME_COEFFICIENT], 1.52233, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.VOLUME_COEFFICIENT], 0.11623, tol)

    def test_partials(self):
        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)


class TestTailComp(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()

    def test_large_single_aisle_1_htail(self):
        prob = self.prob
        prob.model.add_subsystem('tail', TailSize(orientation='horizontal'), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.HorizontalTail.VOLUME_COEFFICIENT, 1.189, units='unitless')
        prob.set_val(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        prob.set_val(Aircraft.HorizontalTail.MOMENT_RATIO, 0.2307, units='unitless')
        prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.615, units='ft')
        prob.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, 4.75, units='unitless')
        prob.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 0.352, units='unitless')

        prob.run_model()

        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], 375.9, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.SPAN], 42.25, tol)

        # (potentially not actual GASP value, it is calculated twice in different places)
        assert_near_equal(prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.16130387591471, tol)

        assert_near_equal(prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.57573, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.7, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)

    def test_large_single_aisle_1_vtail(self):
        prob = self.prob
        prob.model.add_subsystem('tail', TailSize(orientation='vertical'), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        prob.set_val(Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless')
        prob.set_val(Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless')
        prob.set_val(Aircraft.Wing.SPAN, 117.8, units='ft')
        prob.set_val(Aircraft.VerticalTail.ASPECT_RATIO, 1.67, units='unitless')
        prob.set_val(Aircraft.VerticalTail.TAPER_RATIO, 0.801, units='unitless')

        prob.run_model()

        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 469.3, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.SPAN], 28, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.ROOT_CHORD], 18.61267549773935, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.83022, tol)

        # note: slightly different from GASP output value, likely truncation differences, this value is from Kenny
        assert_near_equal(prob[Aircraft.VerticalTail.MOMENT_ARM], 49.87526, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)


class TestEmpennageGroup(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('emp', EmpennageSize(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=12.615, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, val=0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, val=2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

    def test_large_sinle_aisle_1_defaults(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        assert_near_equal(self.prob[Aircraft.HorizontalTail.AREA], 375.9, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.SPAN], 42.25, tol)
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.16130387591471, tol
        )  # (potentially not actual GASP value, it is calculated twice in different places)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.57573, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.7, tol)

        assert_near_equal(self.prob[Aircraft.VerticalTail.AREA], 469.3, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.SPAN], 28, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.ROOT_CHORD], 18.61267549773935, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.83022, tol)
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.MOMENT_ARM], 49.87526, tol
        )  # note: slightly different from GASP output value, likely numerical diff.s, this value is from Kenny

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)

    def test_large_sinle_aisle_1_calc_volcoefs(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, val=True, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units='unitless')
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units='ft')
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')

        self.prob.run_model()

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.VOLUME_COEFFICIENT], 1.52233, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.VOLUME_COEFFICIENT], 0.11623, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)


class BWBTestEmpennageGroup(unittest.TestCase):
    """BWB model"""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('emp', EmpennageSize(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=0.000001, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=2142.85714286, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.5463, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=16.2200522, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=1.705, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.366, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, val=0.0150, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, val=5.2615, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=146.38501094, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.705, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.366, units='unitless'
        )

    def test_case1(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        assert_near_equal(self.prob[Aircraft.HorizontalTail.AREA], 0.00117064, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.SPAN], 0.04467601, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 0.03836448, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 0.02808445, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MOMENT_ARM], 29.69074172, tol)

        assert_near_equal(self.prob[Aircraft.VerticalTail.AREA], 169.11964286, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.SPAN], 16.98084188, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.ROOT_CHORD], 14.58190052, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 10.67457744, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MOMENT_ARM], 27.82191598, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)


if __name__ == '__main__':
    unittest.main()
