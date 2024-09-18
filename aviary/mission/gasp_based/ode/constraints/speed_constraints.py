import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic


class SpeedConstraints(om.ExplicitComponent):
    """
    Compute speed constraint to be driven to zero in order to control speed
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            "EAS_target",
            default=0,
            desc="targeted equivalent airspeed in knots assuming mach constraint is"
            " satisfied",
        )
        self.options.declare(
            "mach_cruise", default=0, desc="targeted cruise mach number"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        arange = np.arange(nn)

        self.add_input(
            "EAS",
            val=np.ones(nn),
            units="kn",
            desc="equivalent airspeed",
        )
        self.add_input(
            Dynamic.Mission.MACH,
            val=np.ones(nn),
            units="unitless",
            desc="mach number",
        )

        self.add_output(
            "speed_constraint",
            val=np.ones((nn, 2)),
            units="unitless",
            desc="constraint to be driven to zero in order to control speed",
        )

        self.declare_partials(
            "speed_constraint", "EAS", rows=arange * 2, cols=arange, val=1.0
        )
        self.declare_partials(
            "speed_constraint",
            Dynamic.Mission.MACH,
            rows=arange * 2 + 1,
            cols=arange,
            val=self.options["EAS_target"],
        )

    def compute(self, inputs, outputs):

        EAS = inputs["EAS"]
        EAS_target = self.options["EAS_target"]
        mach = inputs[Dynamic.Mission.MACH]
        mach_cruise = self.options["mach_cruise"]

        EAS_constraint = EAS - EAS_target
        EAS_constraint = EAS_constraint[:, np.newaxis]
        mach_constraint = EAS_target * (mach - mach_cruise)
        mach_constraint = mach_constraint[:, np.newaxis]
        outputs["speed_constraint"] = np.hstack((EAS_constraint, mach_constraint))
