import numpy as np
from numpy.polynomial import Polynomial
import openmdao.api as om


class PolynomialFit(om.ImplicitComponent):
    """
    Using location data (control points) to build a polynomial fit function
    and to compute initial gear time and flap time.

    Methods
    -------
    initialize(self):
        declare number of control point "N_cp"
    setup(self):
        setup the polynomial fit inputs, outputs, and solver.
    solve_nonlinear(self, inputs, outputs):
        Compute the outputs, given the inputs using the numpy fitting function.
    apply_nonlinear(self, inputs, outputs, residuals):
        Compute the residuals
    """

    def initialize(self):

        self.options.declare("N_cp", types=int, desc="number of control point")

    def setup(self):

        # locations data up want to fit, which can
        # also be thought of as the "control points"
        # of the fit function
        self.add_input("h_cp", shape=self.options["N_cp"], units="ft")
        self.add_input("time_cp", shape=self.options["N_cp"], units="s")

        self.add_input("t_init_gear", 37.3, units="s")
        self.add_output("h_init_gear", shape=1, units="ft")

        self.add_input("t_init_flaps", 47.5, units="s")
        self.add_output("h_init_flaps", shape=1, units="ft")

        # these are the coefficients of the polynomial function you are fitting
        self.add_output("A", np.zeros(4))  # assuming a 5th order polynomial

        # using CS here will give accurate partials, but will miss the sparsity pattern
        # issue #497
        self.declare_partials("*", "*", method="cs")

        self.linear_solver = om.DirectSolver()

    def solve_nonlinear(self, inputs, outputs):
        """
        Compute the outputs, given the inputs using the numpy fitting function.
        """

        X_cp = inputs["time_cp"]
        Y_cp = inputs["h_cp"]

        # using the numpy fitting function because its faster and more stable
        polynomial = Polynomial.fit(X_cp, Y_cp, deg=3)

        coeffs = polynomial.convert().coef

        outputs["A"][:] = coeffs

        (
            a0,
            a1,
            a2,
            a3,
        ) = coeffs

        x_gear = inputs["t_init_gear"]
        outputs["h_init_gear"] = a0 + a1 * x_gear + a2 * x_gear**2 + a3 * x_gear**3

        x_flaps = inputs["t_init_flaps"]
        outputs["h_init_flaps"] = (
            a0 + a1 * x_flaps + a2 * x_flaps**2 + a3 * x_flaps**3
        )

    def apply_nonlinear(self, inputs, outputs, residuals):

        (
            a0,
            a1,
            a2,
            a3,
        ) = outputs["A"]
        """
        Compute the residuals
        """

        X_cp = inputs["time_cp"]
        Y_cp = inputs["h_cp"]

        Y_computed = a0 + a1 * X_cp + a2 * X_cp**2 + a3 * X_cp**3

        # note that derivatives are showing up in the apply_nonlinear method because
        # this is the formulation we use to form the residual.
        # We are minimizing the sum of the square of the error: np.sum((Y_computed-Y_cp)**2) w.r.t A
        # hence we differentiate the objective w.r.t A and set the resulting system of equations to 0
        d_error__d_Y_computed = 2 * (Y_computed - Y_cp)

        d_Y_computed__d_a0 = np.ones(self.options["N_cp"])
        d_Y_computed__d_a1 = X_cp
        d_Y_computed__d_a2 = X_cp**2
        d_Y_computed__d_a3 = X_cp**3

        residuals["A"][0] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a0)
        residuals["A"][1] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a1)
        residuals["A"][2] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a2)
        residuals["A"][3] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a3)

        x_gear = inputs["t_init_gear"]
        h_gear = a0 + a1 * x_gear + a2 * x_gear**2 + a3 * x_gear**3
        residuals["h_init_gear"] = h_gear - outputs["h_init_gear"]

        x_flaps = inputs["t_init_flaps"]
        h_flaps = a0 + a1 * x_flaps + a2 * x_flaps**2 + a3 * x_flaps**3
        residuals["h_init_flaps"] = h_flaps - outputs["h_init_flaps"]
