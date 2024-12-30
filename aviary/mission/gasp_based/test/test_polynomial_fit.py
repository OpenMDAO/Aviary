import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.polynomial_fit import PolynomialFit

X_cp = [45.0, 47.21906891, 50.28093109, 51.25, 51.25, 53.46906891, 56.53093109, 57.5,
        57.5, 59.71906891, 62.78093109, 63.75, 63.75, 65.96906891, 69.03093109, 70.0]
Y_cp = [0.0, 44.38137822, 105.61862178, 125.0, 125.0, 169.38137822, 230.61862178, 250.0,
        250.0, 294.38137822, 355.61862178, 375.0, 375.0, 419.38137822, 480.61862178, 500.0]


class PolynomialFitTest(unittest.TestCase):
    """
    Test computation of initial gear time and initial flap time.
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem("polyfit", PolynomialFit(N_cp=16), promotes=["*"])

        self.prob.model.set_input_defaults("time_cp", val=X_cp, units="s")
        self.prob.model.set_input_defaults("h_cp", val=Y_cp, units="ft")
        self.prob.model.set_input_defaults("t_init_gear", val=[15.0000001], units="s")
        self.prob.model.set_input_defaults("t_init_flaps", val=[32.5000001], units="s")

        newton = self.prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-6
        newton.options["rtol"] = 1e-6
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 15
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 10
        newton.options["err_on_non_converge"] = True
        newton.options["reraise_child_analysiserror"] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1
        newton.options["err_on_non_converge"] = False

        self.prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        # Checks output values only. It will not check partials
        # because cs is used to compute partials already.

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob["h_init_gear"], -600, tol)
        assert_near_equal(self.prob["h_init_flaps"], -250, tol)


if __name__ == "__main__":
    unittest.main()
