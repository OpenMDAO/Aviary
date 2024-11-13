"""Group containing all pre-mission aero calculations.

Flaps modeling occurs as a pre-mission calculation to provide CL max and CL/CD increments to
the dynamic aero.
"""

import openmdao.api as om
from aviary.subsystems.atmosphere.atmosphere import Atmosphere

from aviary.subsystems.aerodynamics.gasp_based.flaps_model import FlapsGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.enums import SpeedType
from aviary.subsystems.aerodynamics.gasp_based.gasp_aero_coeffs import AeroFormfactors
from aviary.subsystems.aerodynamics.gasp_based.interference import WingFuselageInterferencePremission

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

        self.add_subsystem("wing_fus_interference_premission",
                           WingFuselageInterferencePremission(),
                           promotes_inputs=["aircraft:*"],
                           promotes_outputs=[
                               "interference_independent_of_shielded_area",
                               "drag_loss_due_to_shielded_wing_area"],
                           )

        self.add_subsystem("aero_form_factors", AeroFormfactors(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"],
                           )

        # speeds weren't originally computed here, speedtype of Mach is intended
        # to avoid multiple sources for computed Mach (gets calculated somewhere upstream)
        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=1, input_speed_type=SpeedType.MACH),
            promotes=['*', (Dynamic.Mission.ALTITUDE, "alt_flaps")],
        )

        self.add_subsystem(
            "kin_visc",
            om.ExecComp(
                "kinematic_viscosity = viscosity / rho",
                viscosity={"units": "lbf*s/ft**2"},
                rho={"units": "slug/ft**3"},
                kinematic_viscosity={"units": "ft**2/s"},
            ),
            promotes=["viscosity",
                      ("kinematic_viscosity", Dynamic.Mission.KINEMATIC_VISCOSITY),
                      ("rho", Dynamic.Mission.DENSITY)],
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
