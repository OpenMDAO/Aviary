import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic


class GammaComp(om.ExplicitComponent):
    """
    Computes flight path angle and its curvature.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("dh_dr", shape=nn, units="m/distance_units",
                       desc="change in altitude wrt range")

        self.add_input("d2h_dr2", shape=nn, units="m/distance_units**2",
                       desc="second derivative of altitude wrt range")

        self.add_output(Dynamic.Mission.FLIGHT_PATH_ANGLE, shape=nn, units="rad",
                        desc="flight path angle")

        self.add_output("dgam_dr", shape=nn, units="rad/distance_units",
                        desc="change in flight path angle per unit range traversed")

    def setup_partials(self):
        nn = self.options["num_nodes"]
        ar = np.arange(nn, dtype=int)

        self.declare_partials(of=Dynamic.Mission.FLIGHT_PATH_ANGLE,
                              wrt="dh_dr", rows=ar, cols=ar)
        self.declare_partials(of="dgam_dr", wrt=["dh_dr", "d2h_dr2"], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        dh_dr = inputs["dh_dr"]
        d2h_dr2 = inputs["d2h_dr2"]

        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE] = np.arctan(dh_dr)
        outputs["dgam_dr"] = d2h_dr2 / (dh_dr**2 + 1)

    def compute_partials(self, inputs, partials):
        dh_dr = inputs["dh_dr"]
        d2h_dr2 = inputs["d2h_dr2"]

        partials[Dynamic.Mission.FLIGHT_PATH_ANGLE, "dh_dr"] = 1. / (dh_dr**2 + 1)
        partials["dgam_dr", "dh_dr"] = -d2h_dr2 * dh_dr * 2 / (dh_dr**2 + 1)**2
        partials["dgam_dr", "d2h_dr2"] = 1. / (dh_dr**2 + 1)
