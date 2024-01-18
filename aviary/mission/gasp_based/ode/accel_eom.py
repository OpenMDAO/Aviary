import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Dynamic


class AccelerationRates(om.ExplicitComponent):

    """
    Compute the TAS rate, distance rate, and mass flow rate for a level flight acceleration phase.

    Equation comes from climb subroutine of GASP code.
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("analysis_scheme", types=AnalysisScheme, default=AnalysisScheme.COLLOCATION,
                             desc="The analysis method that will be used to close the trajectory; for example collocation or time integration")

    def setup(self):
        analysis_scheme = self.options["analysis_scheme"]
        nn = self.options["num_nodes"]
        arange = np.arange(nn)

        self.add_input(
            Dynamic.Mission.MASS,
            val=np.ones(nn) * 1e6,
            units="lbm",
            desc="total mass of the aircraft",
        )
        self.add_input(
            Dynamic.Mission.DRAG,
            val=np.zeros(nn),
            units="lbf",
            desc="drag on aircraft",
        )
        self.add_input(
            Dynamic.Mission.THRUST_TOTAL,
            val=np.zeros(nn),
            units="lbf",
            desc="total thrust",
        )
        self.add_input(
            "TAS",
            val=np.zeros(nn),
            units="ft/s",
            desc="true air speed",
        )

        self.add_output(
            "TAS_rate",
            val=np.zeros(nn),
            units="ft/s**2",
            desc="rate of change of true air speed",
        )
        self.add_output(
            Dynamic.Mission.DISTANCE_RATE,
            val=np.zeros(nn),
            units="ft/s",
            desc="rate of change of horizontal distance covered",
        )

        self.declare_partials(
            "TAS_rate", [
                Dynamic.Mission.MASS, Dynamic.Mission.DRAG, Dynamic.Mission.THRUST_TOTAL], rows=arange, cols=arange)
        self.declare_partials(Dynamic.Mission.DISTANCE_RATE, [
                              "TAS"], rows=arange, cols=arange, val=1.)

        if analysis_scheme is AnalysisScheme.SHOOTING:
            self.add_input("t_curr", val=np.ones(nn), desc="time", units="s")
            self.add_output(Dynamic.Mission.ALTITUDE_RATE, val=np.ones(nn),
                            desc="altitude rate", units="ft/s")
            self.add_input(
                Dynamic.Mission.DISTANCE, val=np.ones(nn), desc="distance traveled", units="ft")

    def compute(self, inputs, outputs):
        analysis_scheme = self.options["analysis_scheme"]

        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM
        drag = inputs[Dynamic.Mission.DRAG]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        TAS = inputs["TAS"]

        outputs["TAS_rate"] = (GRAV_ENGLISH_GASP / weight) * (thrust - drag)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS

        if analysis_scheme is AnalysisScheme.SHOOTING:
            outputs[Dynamic.Mission.ALTITUDE_RATE] = 0

    def compute_partials(self, inputs, J):
        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM
        drag = inputs[Dynamic.Mission.DRAG]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]

        J["TAS_rate", Dynamic.Mission.MASS] = \
            -(GRAV_ENGLISH_GASP / weight**2) * (thrust - drag) * GRAV_ENGLISH_LBM
        J["TAS_rate", Dynamic.Mission.DRAG] = -(GRAV_ENGLISH_GASP / weight)
        J["TAS_rate", Dynamic.Mission.THRUST_TOTAL] = GRAV_ENGLISH_GASP / weight
