"""Group containing all static aero calculations.

Flaps modeling occurs as a static calculation to provide CL max and CL/CD increments to
the dynamic aero.
"""

import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.subsystems.aerodynamics.gasp_based.flaps_model import FlapsGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

# TODO: add subsystems to compute CLMXFU, CLMXTO, CLMXLD using dynamic aero components
# with alpha > alpha_stall


class PreMissionAero(om.Group):
    """Takeoff and landing flaps modeling"""

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        aviary_options = self.options['aviary_options']

        self.add_subsystem(
            "atmos",
            USatm1976Comp(
                num_nodes=1),
            promotes_inputs=[
                ("h",
                 "alt_flaps")],
            promotes_outputs=[
                ("temp",
                 Dynamic.Mission.TEMPERATURE),
                ("pres",
                 Dynamic.Mission.STATIC_PRESSURE),
                ("sos",
                 Dynamic.Mission.SPEED_OF_SOUND),
                "rho",
                "viscosity"],
        )

        self.add_subsystem(
            "kin_visc",
            om.ExecComp(
                "kinematic_viscosity = viscosity / rho",
                viscosity={"units": "lbf*s/ft**2"},
                rho={"units": "slug/ft**3"},
                kinematic_viscosity={"units": "ft**2/s"},
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "flaps_up",
            FlapsGroup(aviary_options=aviary_options),
            promotes_inputs=[
                "*",
                ("flap_defl", "flap_defl_up"),
                ("slat_defl", "slat_defl_up"),
            ],
            promotes_outputs=[("CL_max", Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP)],
        )
        self.add_subsystem(
            "flaps_takeoff",
            FlapsGroup(aviary_options=aviary_options),
            # slat deflection same for takeoff and landing
            promotes_inputs=["*", ("flap_defl", Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF),
                             ("slat_defl", Aircraft.Wing.MAX_SLAT_DEFLECTION_TAKEOFF)],
            promotes_outputs=[
                ("CL_max", Mission.Takeoff.LIFT_COEFFICIENT_MAX),
                (
                    "delta_CL",
                    Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
                ),
                (
                    "delta_CD",
                    Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
                ),
            ],
        )
        self.add_subsystem(
            "flaps_landing",
            FlapsGroup(aviary_options=aviary_options),
            promotes_inputs=["*", ("flap_defl", Aircraft.Wing.FLAP_DEFLECTION_LANDING),
                             ("slat_defl", Aircraft.Wing.MAX_SLAT_DEFLECTION_LANDING)],
            promotes_outputs=[
                ("CL_max", Mission.Landing.LIFT_COEFFICIENT_MAX),
                (
                    "delta_CL",
                    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
                ),
                (
                    "delta_CD",
                    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
                ),
            ],
        )

        self.set_input_defaults("alt_flaps", 0)
        self.set_input_defaults("flap_defl_up", 0)
        self.set_input_defaults("slat_defl_up", 0)
