import openmdao.api as om
from aviary.subsystems.atmosphere.atmosphere import Atmosphere

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_METRIC
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission, Dynamic


class LandingCalc(om.ExplicitComponent):
    """
    Calculate the distance covered over the ground and approach velocity during landing
    """

    def setup(self):

        add_aviary_input(self, Mission.Landing.TOUCHDOWN_MASS, val=150_000)

        add_aviary_input(
            self,
            Dynamic.Mission.DENSITY,
            val=1.225,
            units="kg/m**3",
            desc="atmospheric density",
        )

        add_aviary_input(self, Aircraft.Wing.AREA, val=700)

        add_aviary_input(self, Mission.Landing.LIFT_COEFFICIENT_MAX, val=3)

        add_aviary_output(self, Mission.Landing.GROUND_DISTANCE, val=0)

        add_aviary_output(self, Mission.Landing.INITIAL_VELOCITY, val=0)

        self.declare_partials(Mission.Landing.INITIAL_VELOCITY, [
                              Mission.Landing.LIFT_COEFFICIENT_MAX, Aircraft.Wing.AREA, Mission.Landing.TOUCHDOWN_MASS])
        self.declare_partials(Mission.Landing.GROUND_DISTANCE, "*")

    def compute(self, inputs, outputs):

        rho_SL = RHO_SEA_LEVEL_METRIC
        landing_weight = inputs[Mission.Landing.TOUCHDOWN_MASS] * \
            GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Mission.DENSITY]
        planform_area = inputs[Aircraft.Wing.AREA]
        Cl_ldg_max = inputs[Mission.Landing.LIFT_COEFFICIENT_MAX]

        rho_ratio = rho / rho_SL

        # TODO: This equation from FLOPS estimates the landing field length, not the actual ground
        # distance covered during landing, which should be less.

        Cl_app = Cl_ldg_max / 1.3 ** 2
        V_app = 17.18644 * (landing_weight / (planform_area * Cl_app)) ** 0.5
        landing_distance = 2500 + 105 * landing_weight / (
            planform_area * rho_ratio * Cl_app * 1.69
        )

        outputs[Mission.Landing.GROUND_DISTANCE] = landing_distance
        outputs[Mission.Landing.INITIAL_VELOCITY] = V_app

    def compute_partials(self, inputs, J):

        rho_SL = RHO_SEA_LEVEL_METRIC
        landing_weight = inputs[Mission.Landing.TOUCHDOWN_MASS] * \
            GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Mission.DENSITY]
        planform_area = inputs[Aircraft.Wing.AREA]
        Cl_ldg_max = inputs[Mission.Landing.LIFT_COEFFICIENT_MAX]

        rho_ratio = rho / rho_SL

        Cl_app = Cl_ldg_max / 1.3 ** 2

        J[Mission.Landing.INITIAL_VELOCITY, Mission.Landing.LIFT_COEFFICIENT_MAX] = (
            17.18644
            * 0.5
            * (landing_weight / (planform_area * Cl_app)) ** (-0.5)
            * (-landing_weight)
            / (planform_area * Cl_app ** 2)
            / 1.3 ** 2
        )
        J[Mission.Landing.INITIAL_VELOCITY, Aircraft.Wing.AREA] = (
            17.18644
            * 0.5
            * (landing_weight / (planform_area * Cl_app)) ** (-0.5)
            * (-landing_weight)
            / (planform_area ** 2 * Cl_app)
        )
        J[Mission.Landing.INITIAL_VELOCITY, Mission.Landing.TOUCHDOWN_MASS] = (
            17.18644
            * 0.5
            * (landing_weight / (planform_area * Cl_app)) ** (-0.5)
            * GRAV_ENGLISH_LBM / (planform_area * Cl_app)
        )

        J[Mission.Landing.GROUND_DISTANCE, Mission.Landing.TOUCHDOWN_MASS] = \
            105 * GRAV_ENGLISH_LBM / (
            planform_area * rho_ratio * Cl_app * 1.69
        )
        J[Mission.Landing.GROUND_DISTANCE, Aircraft.Wing.AREA] = (
            -105 * landing_weight / (planform_area ** 2 * rho_ratio * Cl_app * 1.69)
        )
        J[Mission.Landing.GROUND_DISTANCE, Mission.Landing.LIFT_COEFFICIENT_MAX] = (
            -105
            * landing_weight
            / (planform_area * rho_ratio * Cl_app ** 2 * 1.69)
            / 1.3 ** 2
        )
        J[Mission.Landing.GROUND_DISTANCE, Dynamic.Mission.DENSITY] = (
            -105
            * landing_weight
            / (planform_area * rho_ratio**2 * Cl_app * 1.69)
            / rho_SL
        )


class LandingGroup(om.Group):
    """
    Calculate the distance covered over the ground and approach velocity
    during landing with atmosphere is included.
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
            "calcs",
            LandingCalc(),
            promotes_inputs=[
                Mission.Landing.TOUCHDOWN_MASS,
                Dynamic.Mission.DENSITY,
                Aircraft.Wing.AREA,
                Mission.Landing.LIFT_COEFFICIENT_MAX,
            ],
            promotes_outputs=[
                Mission.Landing.GROUND_DISTANCE,
                Mission.Landing.INITIAL_VELOCITY,
            ],
        )
