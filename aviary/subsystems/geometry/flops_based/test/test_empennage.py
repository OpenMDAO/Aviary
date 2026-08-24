import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.flops_based.empennage import EmpennageSize, TailSize
from aviary.subsystems.geometry.flops_based.prep_geom import PrepGeom
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft

tol = 1e-10
partial_tols = {'atol': 1e-10, 'rtol': 1e-10}


def group_output_names(group):
    """
    Promoted names of the variables a group actually computes.

    Read from the group rather than from the problem's model, because OpenMDAO's
    ``_auto_ivc`` contributes an output for every unconnected input and would otherwise
    make plain inputs look like computed outputs.
    """
    meta = group.get_io_metadata(iotypes=('output',), return_rel_names=False)

    return {item['prom_name'] for item in meta.values()}


# Reference wing, loosely based on the large single aisle model.
WING_AREA = 1370.3
WING_SPAN = 117.8054
WING_TAPER_RATIO = 0.352

HTAIL_VOLUME_COEFFICIENT = 1.189
HTAIL_MOMENT_ARM = 54.0
VTAIL_VOLUME_COEFFICIENT = 0.145
VTAIL_MOMENT_ARM = 49.0


def mean_aerodynamic_chord(area, span, taper_ratio):
    """
    Mean aerodynamic chord of a trapezoidal wing.

    Written here in terms of the root chord, which is a different arrangement of the
    expression used by the component, so that the test does not simply restate the
    implementation.
    """
    root_chord = 2.0 * area / (span * (1.0 + taper_ratio))

    return (2.0 / 3.0) * root_chord * (1.0 + taper_ratio + taper_ratio**2) / (1.0 + taper_ratio)


def build_tail(orientation):
    prob = om.Problem()
    prob.model.add_subsystem('tail', TailSize(orientation=orientation), promotes=['*'])
    prob.setup(check=False, force_alloc_complex=True)

    prob.set_val(Aircraft.Wing.AREA, WING_AREA, units='ft**2')
    prob.set_val(Aircraft.Wing.SPAN, WING_SPAN, units='ft')

    if orientation == 'horizontal':
        prob.set_val(Aircraft.Wing.TAPER_RATIO, WING_TAPER_RATIO, units='unitless')
        prob.set_val(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
            HTAIL_VOLUME_COEFFICIENT,
            units='unitless',
        )
        prob.set_val(Aircraft.HorizontalTail.MOMENT_ARM, HTAIL_MOMENT_ARM, units='ft')
    else:
        prob.set_val(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, VTAIL_VOLUME_COEFFICIENT, units='unitless'
        )
        prob.set_val(Aircraft.VerticalTail.MOMENT_ARM, VTAIL_MOMENT_ARM, units='ft')

    return prob


@use_tempdirs
class TestTailSize(unittest.TestCase):
    def test_horizontal_tail_area(self):
        prob = build_tail('horizontal')
        prob.run_model()

        chord = mean_aerodynamic_chord(WING_AREA, WING_SPAN, WING_TAPER_RATIO)
        expected = HTAIL_VOLUME_COEFFICIENT * WING_AREA * chord / HTAIL_MOMENT_ARM

        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], expected, tol)

    def test_vertical_tail_area(self):
        prob = build_tail('vertical')
        prob.run_model()

        expected = VTAIL_VOLUME_COEFFICIENT * WING_AREA * WING_SPAN / VTAIL_MOMENT_ARM

        assert_near_equal(prob[Aircraft.VerticalTail.AREA], expected, tol)

    def test_horizontal_volume_coefficient_recovered(self):
        # Feeding the computed area back through the definition of a tail volume
        # coefficient must return the coefficient that was asked for.
        prob = build_tail('horizontal')
        prob.run_model()

        area = prob[Aircraft.HorizontalTail.AREA]
        chord = mean_aerodynamic_chord(WING_AREA, WING_SPAN, WING_TAPER_RATIO)
        recovered = area * HTAIL_MOMENT_ARM / (WING_AREA * chord)

        assert_near_equal(recovered, HTAIL_VOLUME_COEFFICIENT, tol)

    def test_vertical_volume_coefficient_recovered(self):
        prob = build_tail('vertical')
        prob.run_model()

        area = prob[Aircraft.VerticalTail.AREA]
        recovered = area * VTAIL_MOMENT_ARM / (WING_AREA * WING_SPAN)

        assert_near_equal(recovered, VTAIL_VOLUME_COEFFICIENT, tol)

    def test_area_scales_with_wing(self):
        # The point of sizing the tails is that they track the wing instead of staying
        # fixed. Doubling wing area alone doubles the vertical tail area.
        prob = build_tail('vertical')
        prob.run_model()
        baseline = prob[Aircraft.VerticalTail.AREA].copy()

        prob.set_val(Aircraft.Wing.AREA, 2.0 * WING_AREA, units='ft**2')
        prob.run_model()

        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 2.0 * baseline, tol)

    def test_partials_horizontal(self):
        prob = build_tail('horizontal')
        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)

    def test_partials_vertical(self):
        prob = build_tail('vertical')
        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)

    def test_partials_untapered_wing(self):
        # A taper ratio of 1 sits at the root of the (tr - 1) factor in the taper
        # derivative, so check it explicitly.
        prob = build_tail('horizontal')
        prob.set_val(Aircraft.Wing.TAPER_RATIO, 1.0, units='unitless')
        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)

    def test_zero_moment_arm_raises(self):
        # The moment arm defaults to zero, which would otherwise divide by zero and
        # silently produce inf.
        prob = build_tail('horizontal')
        prob.set_val(Aircraft.HorizontalTail.MOMENT_ARM, 0.0, units='ft')

        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertIn(Aircraft.HorizontalTail.MOMENT_ARM, str(cm.exception))

    def test_zero_wing_span_raises(self):
        prob = build_tail('vertical')
        prob.set_val(Aircraft.Wing.SPAN, 0.0, units='ft')

        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertIn(Aircraft.Wing.SPAN, str(cm.exception))


@use_tempdirs
class TestEmpennageSize(unittest.TestCase):
    @staticmethod
    def _build(compute_htail, compute_vtail):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_AREA, val=compute_htail, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_AREA, val=compute_vtail, units='unitless')

        prob = om.Problem()
        group = EmpennageSize()
        prob.model.add_subsystem('empennage', group, promotes=['*'])
        setup_model_options(prob, options)
        prob.setup(check=False, force_alloc_complex=True)

        return prob, group

    def test_both_tails_sized(self):
        prob, _ = self._build(True, True)

        prob.set_val(Aircraft.Wing.AREA, WING_AREA, units='ft**2')
        prob.set_val(Aircraft.Wing.SPAN, WING_SPAN, units='ft')
        prob.set_val(Aircraft.Wing.TAPER_RATIO, WING_TAPER_RATIO, units='unitless')
        prob.set_val(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
            HTAIL_VOLUME_COEFFICIENT,
            units='unitless',
        )
        prob.set_val(Aircraft.HorizontalTail.MOMENT_ARM, HTAIL_MOMENT_ARM, units='ft')
        prob.set_val(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, VTAIL_VOLUME_COEFFICIENT, units='unitless'
        )
        prob.set_val(Aircraft.VerticalTail.MOMENT_ARM, VTAIL_MOMENT_ARM, units='ft')

        prob.run_model()

        chord = mean_aerodynamic_chord(WING_AREA, WING_SPAN, WING_TAPER_RATIO)
        expected_htail = HTAIL_VOLUME_COEFFICIENT * WING_AREA * chord / HTAIL_MOMENT_ARM
        expected_vtail = VTAIL_VOLUME_COEFFICIENT * WING_AREA * WING_SPAN / VTAIL_MOMENT_ARM

        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], expected_htail, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AREA], expected_vtail, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, **partial_tols)

    def test_no_tails_sized_by_default(self):
        _, group = self._build(False, False)
        outputs = group_output_names(group)

        self.assertNotIn(Aircraft.HorizontalTail.AREA, outputs)
        self.assertNotIn(Aircraft.VerticalTail.AREA, outputs)

    def test_horizontal_tail_only(self):
        _, group = self._build(True, False)
        outputs = group_output_names(group)

        self.assertIn(Aircraft.HorizontalTail.AREA, outputs)
        self.assertNotIn(Aircraft.VerticalTail.AREA, outputs)

    def test_vertical_tail_only(self):
        _, group = self._build(False, True)
        outputs = group_output_names(group)

        self.assertNotIn(Aircraft.HorizontalTail.AREA, outputs)
        self.assertIn(Aircraft.VerticalTail.AREA, outputs)


@use_tempdirs
class TestPrepGeomTailSizing(unittest.TestCase):
    """Check that PrepGeom picks the empennage group up only when it is asked to."""

    @staticmethod
    def _output_names(compute_htail, compute_vtail):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_AREA, val=compute_htail, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_AREA, val=compute_vtail, units='unitless')

        prob = om.Problem()
        group = PrepGeom()
        prob.model.add_subsystem('prep_geom', group, promotes=['*'])
        setup_model_options(prob, options)
        prob.setup(check=False)

        return group_output_names(group)

    def test_tail_areas_are_inputs_by_default(self):
        # Existing FLOPS models take both tail areas as user inputs; that must not change.
        outputs = self._output_names(False, False)

        self.assertNotIn(Aircraft.HorizontalTail.AREA, outputs)
        self.assertNotIn(Aircraft.VerticalTail.AREA, outputs)

    def test_tail_areas_become_outputs_when_requested(self):
        outputs = self._output_names(True, True)

        self.assertIn(Aircraft.HorizontalTail.AREA, outputs)
        self.assertIn(Aircraft.VerticalTail.AREA, outputs)


if __name__ == '__main__':
    unittest.main()
