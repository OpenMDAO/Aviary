import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.gasp_aero_coeffs import AeroFormfactors
from aviary.variable_info.variables import Aircraft, Mission

tol = 1e-8


class TestAeroCoeffs(unittest.TestCase):
    def test_aero_coeffs(self):
        aero_coeffs = AeroFormfactors()
        prob = om.Problem()
        prob.model.add_subsystem("comp", aero_coeffs, promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, .12)
        prob.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, .15)
        prob.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, .1)
        prob.set_val(Aircraft.Strut.THICKNESS_TO_CHORD, .1)
        prob.set_val(Aircraft.Wing.SWEEP, 30, units='deg')
        prob.set_val(Aircraft.VerticalTail.SWEEP, 15, units='deg')
        prob.set_val(Aircraft.HorizontalTail.SWEEP, 10, units='deg')
        prob.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0)
        prob.set_val(Mission.Design.MACH, .6)
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 6)
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH, 10)

        prob.run_model()

        assert_near_equal(prob.get_val(Aircraft.Wing.FORM_FACTOR), [2.4892359], tol)
        assert_near_equal(prob.get_val(
            Aircraft.VerticalTail.FORM_FACTOR), [2.73809734], tol)
        assert_near_equal(prob.get_val(
            Aircraft.HorizontalTail.FORM_FACTOR), [2.59223754], tol)
        assert_near_equal(prob.get_val(
            Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR), [2.4786], tol)
        assert_near_equal(prob.get_val(Aircraft.Nacelle.FORM_FACTOR), [1.815], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
