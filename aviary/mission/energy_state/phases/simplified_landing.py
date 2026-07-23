import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LandingCalc(om.ExplicitComponent):
    """Calculate the distance covered over the ground and approach velocity during landing."""

    def setup(self):
        add_aviary_input(self, Mission.FINAL_MASS, units='lbm')
        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, units='slug/ft**3')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Mission.Landing.LIFT_COEFFICIENT_MAX, units='unitless')

        add_aviary_output(self, Mission.Landing.GROUND_DISTANCE, units='ft')
        add_aviary_output(self, Mission.Landing.INITIAL_VELOCITY, units='ft/s')

        self.declare_partials(Mission.Landing.INITIAL_VELOCITY, '*')
        self.declare_partials(Mission.Landing.GROUND_DISTANCE, '*')

    def compute(self, inputs, outputs):
        rho_SL = RHO_SEA_LEVEL_ENGLISH
        landing_weight = inputs[Mission.FINAL_MASS] * GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        planform_area = inputs[Aircraft.Wing.AREA]
        Cl_ldg_max = inputs[Mission.Landing.LIFT_COEFFICIENT_MAX]

        rho_ratio = rho / rho_SL

        # TODO: This equation from FLOPS estimates the landing field length, not the actual ground
        # distance covered during landing, which should be less.

        Cl_app = Cl_ldg_max / 1.3**2

        V_app = ((2 * landing_weight) / (rho * planform_area * Cl_app)) ** 0.5

        landing_distance = 2500 + 105 * landing_weight / (planform_area * rho_ratio * Cl_app * 1.69)

        outputs[Mission.Landing.GROUND_DISTANCE] = landing_distance
        outputs[Mission.Landing.INITIAL_VELOCITY] = V_app

    def compute_partials(self, inputs, J):
        rho_SL = RHO_SEA_LEVEL_ENGLISH
        landing_weight = inputs[Mission.FINAL_MASS] * GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        planform_area = inputs[Aircraft.Wing.AREA]
        Cl_ldg_max = inputs[Mission.Landing.LIFT_COEFFICIENT_MAX]

        rho_ratio = rho / rho_SL

        Cl_app = Cl_ldg_max / 1.3**2

        # INITIAL_VELOCITY (V_app) Partials
        V_app = ((2 * landing_weight) / (rho * planform_area * Cl_app)) ** 0.5
        d_sqrt = 0.5 / V_app

        J[Mission.Landing.INITIAL_VELOCITY, Mission.FINAL_MASS] = (
            d_sqrt * (2 * GRAV_ENGLISH_LBM) / (rho * planform_area * Cl_app)
        )

        J[Mission.Landing.INITIAL_VELOCITY, Dynamic.Atmosphere.DENSITY] = (
            d_sqrt * (-2 * landing_weight) / (rho**2 * planform_area * Cl_app)
        )

        J[Mission.Landing.INITIAL_VELOCITY, Aircraft.Wing.AREA] = (
            d_sqrt * (-2 * landing_weight) / (rho * planform_area**2 * Cl_app)
        )

        J[Mission.Landing.INITIAL_VELOCITY, Mission.Landing.LIFT_COEFFICIENT_MAX] = (
            d_sqrt * (-2 * landing_weight) / (rho * planform_area * Cl_app**2) / 1.3**2
        )

        # GROUND DISTANCE Partials:
        J[Mission.Landing.GROUND_DISTANCE, Mission.FINAL_MASS] = (
            105 * GRAV_ENGLISH_LBM / (planform_area * rho_ratio * Cl_app * 1.69)
        )
        J[Mission.Landing.GROUND_DISTANCE, Aircraft.Wing.AREA] = (
            -105 * landing_weight / (planform_area**2 * rho_ratio * Cl_app * 1.69)
        )
        J[Mission.Landing.GROUND_DISTANCE, Mission.Landing.LIFT_COEFFICIENT_MAX] = (
            -105 * landing_weight / (planform_area * rho_ratio * Cl_app**2 * 1.69) / 1.3**2
        )
        J[Mission.Landing.GROUND_DISTANCE, Dynamic.Atmosphere.DENSITY] = (
            -105 * landing_weight / (planform_area * rho_ratio**2 * Cl_app * 1.69) / rho_SL
        )


class LandingGroup(om.Group):
    """
    Calculate the estimated Landing field length and velocity given the aircraft properties and atmospheric conditions.
    Note this is not the actual distance covered over the ground.
    """

    def setup(self):
        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=1),
            promotes=[
                '*',
                (Dynamic.Mission.ALTITUDE, Mission.Landing.INITIAL_ALTITUDE),
            ],
        )

        self.add_subsystem(
            'calcs',
            LandingCalc(),
            promotes_inputs=[
                Mission.FINAL_MASS,
                Dynamic.Atmosphere.DENSITY,
                Aircraft.Wing.AREA,
                Mission.Landing.LIFT_COEFFICIENT_MAX,
            ],
            promotes_outputs=[
                Mission.Landing.GROUND_DISTANCE,
                Mission.Landing.INITIAL_VELOCITY,
            ],
        )
