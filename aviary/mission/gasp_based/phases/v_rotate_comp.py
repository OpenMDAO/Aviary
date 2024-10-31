import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class VRotateComp(om.ExplicitComponent):
    """
    Component that computes V_rotate based on vehicle properties and speed buffers.
    Note: This component is not used.
    """

    def setup(self):
        # Temporarily set this to shape (1, 1) to avoid OpenMDAO bug
        add_aviary_input(self, Dynamic.Mission.MASS, shape=(1, 1), units="lbm")
        add_aviary_input(
            self,
            Dynamic.Mission.DENSITY,
            shape=(1,),
            units="slug/ft**3",
            val=RHO_SEA_LEVEL_ENGLISH,
            desc="sea-level atmospheric density",
        )
        add_aviary_input(self, Aircraft.Wing.AREA, val=1.0)
        self.add_input("CL_max", shape=(1,), units="unitless",
                       desc="Maximum lift coefficient")
        self.add_input("dV1", shape=(1,), units="ft/s",
                       desc="Increment of engine failure decision speed above stall speed.")
        self.add_input("dVR", shape=(1,), units="ft/s",
                       desc="Increment of takeoff rotation speed above engine failure decision speed.")

        self.add_output("Vrot", shape=(1,), units="ft/s",
                        desc="Speed at which takeoff rotation should be initiated.")

        # Constant partials
        self.declare_partials(of="Vrot", wrt=["dV1", "dVR"], val=1.0)
        # Partials of nonlinear terms
        self.declare_partials(
            of="Vrot",
            wrt=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DENSITY,
                Aircraft.Wing.AREA,
                "CL_max",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mass, rho, wing_area, CL_max, dV1, dVR = inputs.values()
        outputs["Vrot"] = ((2 * mass * GRAV_ENGLISH_LBM) /
                           (rho * wing_area * CL_max))**0.5 + dV1 + dVR

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mass, rho, wing_area, CL_max, dV1, dVR = inputs.values()
        K = 0.5 * ((2 * mass * GRAV_ENGLISH_LBM) /
                   (rho * wing_area * CL_max)) ** 0.5

        partials["Vrot", Dynamic.Mission.MASS] = K / mass
        partials["Vrot", Dynamic.Mission.DENSITY] = -K / rho
        partials["Vrot", Aircraft.Wing.AREA] = -K / wing_area
        partials["Vrot", "CL_max"] = -K / CL_max
