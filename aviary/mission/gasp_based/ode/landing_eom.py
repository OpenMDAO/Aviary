import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM, MU_LANDING, RHO_SEA_LEVEL_ENGLISH
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LandingAltitudeComponent(om.ExplicitComponent):
    """
    Compute the landing altitude.
    """

    def setup(self):
        add_aviary_input(self, Mission.Landing.OBSTACLE_HEIGHT, val=50.0)
        add_aviary_input(self, Mission.Landing.AIRPORT_ALTITUDE, val=0.0)

        add_aviary_output(self, Mission.Landing.INITIAL_ALTITUDE, val=0.0)

        self.declare_partials(Mission.Landing.INITIAL_ALTITUDE, "*", val=1)

    def compute(self, inputs, outputs):
        (
            approach_height,
            airport_alt,
        ) = inputs.values()
        outputs[Mission.Landing.INITIAL_ALTITUDE] = airport_alt + approach_height

    def compute_partials(self, inputs, J):
        pass


class GlideConditionComponent(om.ExplicitComponent):
    """
    Compute the initial conditions of the 2DOF glide phase.
    """

    def setup(self):
        self.add_input("rho_app", val=0.0, units="slug/ft**3", desc="air density")
        add_aviary_input(self, Mission.Landing.MAXIMUM_SINK_RATE, val=900.0)
        self.add_input(Dynamic.Mission.MASS, val=0.0, units="lbm",
                       desc="aircraft mass at start of landing")
        add_aviary_input(self, Aircraft.Wing.AREA, val=1.0)
        add_aviary_input(self, Mission.Landing.GLIDE_TO_STALL_RATIO, val=1.3)
        self.add_input("CL_max", val=0.0, units='unitless',
                       desc="CLMX: max CL at approach altitude")

        add_aviary_input(self, Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR, val=1.15)
        add_aviary_input(self, Mission.Landing.TOUCHDOWN_SINK_RATE, val=5.0)
        add_aviary_input(self, Mission.Landing.INITIAL_ALTITUDE, val=0.0)
        add_aviary_input(self, Mission.Landing.BRAKING_DELAY, val=1.0)

        add_aviary_output(self, Mission.Landing.INITIAL_VELOCITY, val=0.0,
                          desc="glide speed calculated using TAS_stall")
        add_aviary_output(self, Mission.Landing.STALL_VELOCITY, val=0.0)

        self.add_output(
            "TAS_touchdown", val=0.0, units="ft/s", desc="VTD: touchdown speed"
        )
        self.add_output(
            "density_ratio", val=0.0, units="unitless", desc="DRAT: density ratio for DLAND"
        )
        self.add_output(
            "wing_loading_land",
            val=0.0,
            units="lbf/ft**2",
            desc="WOS: wing loading at landing",
        )
        self.add_output(
            "theta",
            val=0.0,
            units="rad",
            desc="THETA: theta angle for approach to transition",
        )

        self.add_output(
            "glide_distance", val=0.0, units="ft", desc="DLGL: glide distance"
        )
        self.add_output(
            "tr_distance",
            val=0.0,
            units="ft",
            desc="DLTR: distance covered by the flare maneuver",
        )
        self.add_output(
            "delay_distance",
            val=0.0,
            units="ft",
            desc="DDELAY: delay distance - touchdown to brake application",
        )
        self.add_output(
            "flare_alt", val=0.0, units="ft", desc="HFLAR: altitude of flare maneuver"
        )

        self.declare_partials(
            Mission.Landing.INITIAL_VELOCITY,
            [Dynamic.Mission.MASS, Aircraft.Wing.AREA, "CL_max", "rho_app",
                Mission.Landing.GLIDE_TO_STALL_RATIO],
        )
        self.declare_partials(
            Mission.Landing.STALL_VELOCITY, [
                Dynamic.Mission.MASS, Aircraft.Wing.AREA, "CL_max", "rho_app"]
        )
        self.declare_partials(
            "TAS_touchdown",
            [Mission.Landing.GLIDE_TO_STALL_RATIO, Dynamic.Mission.MASS,
                Aircraft.Wing.AREA, "CL_max", "rho_app"],
        )
        self.declare_partials("density_ratio", ["rho_app"])
        self.declare_partials("wing_loading_land", [
                              Dynamic.Mission.MASS, Aircraft.Wing.AREA])
        self.declare_partials(
            "theta",
            [
                Mission.Landing.MAXIMUM_SINK_RATE,
                Dynamic.Mission.MASS,
                Aircraft.Wing.AREA,
                "CL_max",
                "rho_app",
                Mission.Landing.GLIDE_TO_STALL_RATIO,
            ],
        )
        self.declare_partials(
            "glide_distance",
            [
                Mission.Landing.INITIAL_ALTITUDE,
                Mission.Landing.MAXIMUM_SINK_RATE,
                Dynamic.Mission.MASS,
                Aircraft.Wing.AREA,
                "CL_max",
                "rho_app",
                Mission.Landing.GLIDE_TO_STALL_RATIO,
            ],
        )
        self.declare_partials(
            "tr_distance",
            [
                Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
                Mission.Landing.TOUCHDOWN_SINK_RATE,
                Dynamic.Mission.MASS,
                Aircraft.Wing.AREA,
                "CL_max",
                "rho_app",
                Mission.Landing.GLIDE_TO_STALL_RATIO,
                Mission.Landing.MAXIMUM_SINK_RATE,
            ],
        )
        self.declare_partials(
            "delay_distance",
            [
                Mission.Landing.GLIDE_TO_STALL_RATIO,
                Dynamic.Mission.MASS,
                Aircraft.Wing.AREA,
                "CL_max",
                "rho_app",
                Mission.Landing.BRAKING_DELAY,
            ],
        )
        self.declare_partials(
            "flare_alt",
            [
                Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
                Mission.Landing.TOUCHDOWN_SINK_RATE,
                Mission.Landing.MAXIMUM_SINK_RATE,
                Dynamic.Mission.MASS,
                Aircraft.Wing.AREA,
                "CL_max",
                "rho_app",
                Mission.Landing.GLIDE_TO_STALL_RATIO,
            ],
        )

    def compute(self, inputs, outputs):
        (
            rho_app,
            rate_of_sink_max,
            mass,
            wing_area,
            glide_to_stall_ratio,
            CL_max,
            landing_flare_load_factor,
            rate_of_sink_td,
            approach_alt,
            time_delay,
        ) = inputs.values()
        weight = mass * GRAV_ENGLISH_LBM
        G = GRAV_ENGLISH_GASP

        wing_loading_land = weight / wing_area
        TAS_stall = np.sqrt(2 * wing_loading_land / (CL_max * rho_app))
        TAS_glide = TAS_stall * glide_to_stall_ratio
        theta = np.arcsin(rate_of_sink_max / (60.0 * TAS_glide))
        glide_distance = approach_alt / np.tan(theta)

        # TODO: I didn't understand what is being iterated for glide gamma and it looks like the GASP input/output just uses the maximum rate of sink
        gamma_touchdown = rate_of_sink_td / TAS_glide
        touchdown_velocity_ratio = (glide_to_stall_ratio + 1.0) * 0.5
        RZ = TAS_glide * TAS_glide / G / (landing_flare_load_factor - 1.0)

        flare_alt = (
            TAS_glide
            * TAS_glide
            * (theta * theta - gamma_touchdown * gamma_touchdown)
            / (2.0 * G * (landing_flare_load_factor - 1.0))
        )
        TAS_touchdown = touchdown_velocity_ratio * TAS_stall
        tr_distance = ((RZ * theta) / 2.0) * ((1.0 - gamma_touchdown / theta) ** 2)
        delay_distance = TAS_touchdown * time_delay

        outputs["flare_alt"] = flare_alt
        outputs[Mission.Landing.INITIAL_VELOCITY] = TAS_glide
        outputs[Mission.Landing.STALL_VELOCITY] = TAS_stall
        outputs["TAS_touchdown"] = TAS_touchdown
        outputs["density_ratio"] = rho_app / RHO_SEA_LEVEL_ENGLISH
        outputs["wing_loading_land"] = wing_loading_land
        outputs["glide_distance"] = glide_distance
        outputs["tr_distance"] = tr_distance
        outputs["delay_distance"] = delay_distance
        outputs["theta"] = theta

    def compute_partials(self, inputs, J):
        (
            rho_app,
            rate_of_sink_max,
            mass,
            wing_area,
            glide_to_stall_ratio,
            CL_max,
            landing_flare_load_factor,
            rate_of_sink_td,
            approach_alt,
            time_delay,
        ) = inputs.values()
        weight = mass * GRAV_ENGLISH_LBM
        G = GRAV_ENGLISH_GASP

        wing_loading_land = weight / wing_area
        TAS_stall = np.sqrt(2 * wing_loading_land / (CL_max * rho_app))
        TAS_glide = TAS_stall * glide_to_stall_ratio
        gamma_touchdown = rate_of_sink_td / TAS_glide
        touchdown_velocity_ratio = (glide_to_stall_ratio + 1.0) * 0.5
        theta = np.arcsin(rate_of_sink_max / (60.0 * TAS_glide))
        TAS_touchdown = touchdown_velocity_ratio * TAS_stall

        dTasStall_dWeight = (
            (2 * (1 / wing_area) / (CL_max * rho_app)) ** 0.5 * 0.5 * weight ** (-0.5)
        )
        dTasStall_dWingArea = (
            (2 * weight / (CL_max * rho_app)) ** 0.5 * (-0.5) * wing_area ** (-1.5)
        )
        dTasStall_dClMax = (
            (2 * wing_loading_land / (rho_app)) ** 0.5 * (-0.5) * CL_max ** (-1.5)
        )
        dTasStall_dRhoApp = (
            (2 * wing_loading_land / (CL_max)) ** 0.5 * (-0.5) * rho_app ** (-1.5)
        )
        dTasGlide_dWeight = (
            dTasStall_dWeight * glide_to_stall_ratio
        )
        dTasTd_dWeight = (
            touchdown_velocity_ratio * dTasStall_dWeight
        )
        dTheta_dWeight = (
            (1 - (rate_of_sink_max / (60.0 * TAS_glide)) ** 2) ** (-0.5)
            * (-rate_of_sink_max / (60.0 * TAS_glide**2))
            * dTasGlide_dWeight
        )

        J[Mission.Landing.INITIAL_VELOCITY, Dynamic.Mission.MASS] = \
            dTasGlide_dWeight * GRAV_ENGLISH_LBM
        J[Mission.Landing.INITIAL_VELOCITY, Aircraft.Wing.AREA] = dTasGlide_dWingArea = (
            dTasStall_dWingArea * glide_to_stall_ratio
        )
        J[Mission.Landing.INITIAL_VELOCITY, "CL_max"] = dTasGlide_dClMax = (
            dTasStall_dClMax * glide_to_stall_ratio
        )
        J[Mission.Landing.INITIAL_VELOCITY, "rho_app"] = dTasGlide_dRhoApp = (
            dTasStall_dRhoApp * glide_to_stall_ratio
        )
        J[Mission.Landing.INITIAL_VELOCITY,
            Mission.Landing.GLIDE_TO_STALL_RATIO] = TAS_stall

        J[Mission.Landing.STALL_VELOCITY, Dynamic.Mission.MASS] = \
            dTasStall_dWeight * GRAV_ENGLISH_LBM
        J[Mission.Landing.STALL_VELOCITY, Aircraft.Wing.AREA] = dTasStall_dWingArea
        J[Mission.Landing.STALL_VELOCITY, "CL_max"] = dTasStall_dClMax
        J[Mission.Landing.STALL_VELOCITY, "rho_app"] = dTasStall_dRhoApp

        J["TAS_touchdown", Mission.Landing.GLIDE_TO_STALL_RATIO] = dTasTd_dGlideToStallRatio = (
            0.5 * TAS_stall
        )
        J["TAS_touchdown", Dynamic.Mission.MASS] = dTasTd_dWeight * GRAV_ENGLISH_LBM
        J["TAS_touchdown", Aircraft.Wing.AREA] = dTasTd_dWingArea = (
            touchdown_velocity_ratio * dTasStall_dWingArea
        )
        J["TAS_touchdown", "CL_max"] = dTasTd_dClMax = (
            touchdown_velocity_ratio * dTasStall_dClMax
        )
        J["TAS_touchdown", "rho_app"] = dTasTd_dRhoApp = (
            touchdown_velocity_ratio * dTasStall_dRhoApp
        )

        J["density_ratio", "rho_app"] = 1 / RHO_SEA_LEVEL_ENGLISH

        J["wing_loading_land", Dynamic.Mission.MASS] = GRAV_ENGLISH_LBM / wing_area
        J["wing_loading_land", Aircraft.Wing.AREA] = -weight / wing_area**2

        np.arcsin(rate_of_sink_max / (60.0 * TAS_glide))

        J["theta", Mission.Landing.MAXIMUM_SINK_RATE] = dTheta_dRateOfSinkMax = (
            (1 - (rate_of_sink_max / (60.0 * TAS_glide)) ** 2) ** (-0.5)
            * 1
            / (60.0 * TAS_glide)
        )
        J["theta", Dynamic.Mission.MASS] = dTheta_dWeight * GRAV_ENGLISH_LBM
        J["theta", Aircraft.Wing.AREA] = dTheta_dWingArea = (
            (1 - (rate_of_sink_max / (60.0 * TAS_glide)) ** 2) ** (-0.5)
            * (-rate_of_sink_max / (60.0 * TAS_glide**2))
            * dTasGlide_dWingArea
        )
        J["theta", "CL_max"] = dTheta_dClMax = (
            (1 - (rate_of_sink_max / (60.0 * TAS_glide)) ** 2) ** (-0.5)
            * (-rate_of_sink_max / (60.0 * TAS_glide**2))
            * dTasGlide_dClMax
        )
        J["theta", "rho_app"] = dTheta_dRhoApp = (
            (1 - (rate_of_sink_max / (60.0 * TAS_glide)) ** 2) ** (-0.5)
            * (-rate_of_sink_max / (60.0 * TAS_glide**2))
            * dTasGlide_dRhoApp
        )
        J["theta", Mission.Landing.GLIDE_TO_STALL_RATIO] = dTheta_dGlideToStallRatio = (
            (1 - (rate_of_sink_max / (60.0 * TAS_glide)) ** 2) ** (-0.5)
            * (-rate_of_sink_max / (60.0 * TAS_glide**2))
            * TAS_stall
        )

        approach_alt / np.tan(theta)

        J["glide_distance", Mission.Landing.INITIAL_ALTITUDE] = 1 / np.tan(theta)
        J["glide_distance", Mission.Landing.MAXIMUM_SINK_RATE] = (
            -approach_alt
            / (np.tan(theta)) ** 2
            * (1 / np.cos(theta)) ** 2
            * dTheta_dRateOfSinkMax
        )
        J["glide_distance", Dynamic.Mission.MASS] = (
            -approach_alt
            / (np.tan(theta)) ** 2
            * (1 / np.cos(theta)) ** 2
            * dTheta_dWeight * GRAV_ENGLISH_LBM
        )
        J["glide_distance", Aircraft.Wing.AREA] = (
            -approach_alt
            / (np.tan(theta)) ** 2
            * (1 / np.cos(theta)) ** 2
            * dTheta_dWingArea
        )
        J["glide_distance", "CL_max"] = (
            -approach_alt
            / (np.tan(theta)) ** 2
            * (1 / np.cos(theta)) ** 2
            * dTheta_dClMax
        )
        J["glide_distance", "rho_app"] = (
            -approach_alt
            / (np.tan(theta)) ** 2
            * (1 / np.cos(theta)) ** 2
            * dTheta_dRhoApp
        )
        J["glide_distance", Mission.Landing.GLIDE_TO_STALL_RATIO] = (
            -approach_alt
            / (np.tan(theta)) ** 2
            * (1 / np.cos(theta)) ** 2
            * dTheta_dGlideToStallRatio
        )

        RZ = TAS_glide**2 / G / (landing_flare_load_factor - 1.0)

        dRZ_dWeight = (
            2 * TAS_glide / G / (landing_flare_load_factor - 1.0) * dTasGlide_dWeight
        )
        dRZ_dWingArea = (
            2 * TAS_glide / G / (landing_flare_load_factor - 1.0) * dTasGlide_dWingArea
        )
        dRZ_dClMax = (
            2 * TAS_glide / G / (landing_flare_load_factor - 1.0) * dTasGlide_dClMax
        )
        dRZ_dRhoApp = (
            2 * TAS_glide / G / (landing_flare_load_factor - 1.0) * dTasGlide_dRhoApp
        )
        dRZ_dGlideToStallRatio = (
            2 * TAS_glide / G / (landing_flare_load_factor - 1.0) * TAS_stall
        )
        dRZ_dLandingFlareLoadFactor = (
            -(TAS_glide**2) / G / (landing_flare_load_factor - 1.0) ** 2
        )

        gamma_touchdown = rate_of_sink_td / TAS_glide

        dGammaTd_dRateOfSinkTd = 1 / TAS_glide
        dGammaTd_dWeight = -rate_of_sink_td / TAS_glide**2 * dTasGlide_dWeight
        dGammaTd_dWingArea = -rate_of_sink_td / TAS_glide**2 * dTasGlide_dWingArea
        dGammaTd_dClMax = -rate_of_sink_td / TAS_glide**2 * dTasGlide_dClMax
        dGammaTd_dRhoApp = -rate_of_sink_td / TAS_glide**2 * dTasGlide_dRhoApp
        dGammaTd_dGlideToStallRatio = -rate_of_sink_td / TAS_glide**2 * TAS_stall

        inter1 = (RZ * theta) / 2.0
        dInter1_dLandingFlareLoadFactor = 0.5 * (dRZ_dLandingFlareLoadFactor * theta)
        dInter1_dRateOfSinkMax = 0.5 * (RZ * dTheta_dRateOfSinkMax)
        dInter1_dWeight = 0.5 * (dRZ_dWeight * theta + RZ * dTheta_dWeight)
        dInter1_dWingArea = 0.5 * (dRZ_dWingArea * theta + RZ * dTheta_dWingArea)
        dInter1_dClMax = 0.5 * (dRZ_dClMax * theta + RZ * dTheta_dClMax)
        dInter1_dRhoApp = 0.5 * (dRZ_dRhoApp * theta + RZ * dTheta_dRhoApp)
        dInter1_dGlideToStallRatio = 0.5 * (
            dRZ_dGlideToStallRatio * theta + RZ * dTheta_dGlideToStallRatio
        )

        inter2 = (1.0 - gamma_touchdown / theta) ** 2
        dInter2_dRateOfSinkMax = (
            2
            * (1.0 - gamma_touchdown / theta)
            * gamma_touchdown
            / theta**2
            * dTheta_dRateOfSinkMax
        )
        dInter2_dWeight = (
            -2
            * (1.0 - gamma_touchdown / theta)
            * (theta * dGammaTd_dWeight - gamma_touchdown * dTheta_dWeight)
            / theta**2
        )
        dInter2_dWingArea = (
            -2
            * (1.0 - gamma_touchdown / theta)
            * (theta * dGammaTd_dWingArea - gamma_touchdown * dTheta_dWingArea)
            / theta**2
        )
        dInter2_dClMax = (
            -2
            * (1.0 - gamma_touchdown / theta)
            * (theta * dGammaTd_dClMax - gamma_touchdown * dTheta_dClMax)
            / theta**2
        )
        dInter2_dRhoApp = (
            -2
            * (1.0 - gamma_touchdown / theta)
            * (theta * dGammaTd_dRhoApp - gamma_touchdown * dTheta_dRhoApp)
            / theta**2
        )
        dInter2_dGlideToStallRatio = (
            -2
            * (1.0 - gamma_touchdown / theta)
            * (
                theta * dGammaTd_dGlideToStallRatio
                - gamma_touchdown * dTheta_dGlideToStallRatio
            )
            / theta**2
        )
        dInter2_dRateOfSinkTd = (
            -2 * (1.0 - gamma_touchdown / theta) * dGammaTd_dRateOfSinkTd / theta
        )

        tr_distance = inter1 * inter2

        J["tr_distance", Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR] = (
            dInter1_dLandingFlareLoadFactor * inter2
        )
        J["tr_distance", Mission.Landing.MAXIMUM_SINK_RATE] = (
            dInter1_dRateOfSinkMax * inter2 + inter1 * dInter2_dRateOfSinkMax
        )
        J["tr_distance", Dynamic.Mission.MASS] = (
            dInter1_dWeight * inter2 + inter1 * dInter2_dWeight
        ) * GRAV_ENGLISH_LBM
        J["tr_distance", Aircraft.Wing.AREA] = (
            dInter1_dWingArea * inter2 + inter1 * dInter2_dWingArea
        )
        J["tr_distance", "CL_max"] = dInter1_dClMax * inter2 + inter1 * dInter2_dClMax
        J["tr_distance", "rho_app"] = (
            dInter1_dRhoApp * inter2 + inter1 * dInter2_dRhoApp
        )
        J["tr_distance", Mission.Landing.GLIDE_TO_STALL_RATIO] = (
            dInter1_dGlideToStallRatio * inter2 + inter1 * dInter2_dGlideToStallRatio
        )
        J["tr_distance", Mission.Landing.TOUCHDOWN_SINK_RATE] = inter1 * dInter2_dRateOfSinkTd

        J["delay_distance", Mission.Landing.GLIDE_TO_STALL_RATIO] = (
            time_delay * dTasTd_dGlideToStallRatio
        )
        J["delay_distance", Dynamic.Mission.MASS] = \
            time_delay * dTasTd_dWeight * GRAV_ENGLISH_LBM
        J["delay_distance", Aircraft.Wing.AREA] = time_delay * dTasTd_dWingArea
        J["delay_distance", "CL_max"] = time_delay * dTasTd_dClMax
        J["delay_distance", "rho_app"] = time_delay * dTasTd_dRhoApp
        J["delay_distance", Mission.Landing.BRAKING_DELAY] = TAS_touchdown

        flare_alt = (
            TAS_glide**2
            * (theta**2 - gamma_touchdown**2)
            / (2.0 * G * (landing_flare_load_factor - 1.0))
        )

        J["flare_alt", Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR] = (
            -(TAS_glide**2)
            * (theta**2 - gamma_touchdown**2)
            / (2.0 * G * (landing_flare_load_factor - 1.0)) ** 2
            * 2
            * G
        )
        J["flare_alt", Mission.Landing.TOUCHDOWN_SINK_RATE] = (
            -2
            * TAS_glide**2
            * gamma_touchdown
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * dGammaTd_dRateOfSinkTd
        )
        J["flare_alt", Mission.Landing.MAXIMUM_SINK_RATE] = (
            2
            * TAS_glide**2
            * theta
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * dTheta_dRateOfSinkMax
        )
        J["flare_alt", Dynamic.Mission.MASS] = (
            1
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * (
                2 * TAS_glide * dTasGlide_dWeight * (theta**2 - gamma_touchdown**2)
                + TAS_glide**2
                * (2 * theta * dTheta_dWeight - 2 * gamma_touchdown * dGammaTd_dWeight)
            ) * GRAV_ENGLISH_LBM
        )
        J["flare_alt", Aircraft.Wing.AREA] = (
            1
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * (
                2
                * TAS_glide
                * dTasGlide_dWingArea
                * (theta**2 - gamma_touchdown**2)
                + TAS_glide**2
                * (
                    2 * theta * dTheta_dWingArea
                    - 2 * gamma_touchdown * dGammaTd_dWingArea
                )
            )
        )
        J["flare_alt", "CL_max"] = (
            1
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * (
                2 * TAS_glide * dTasGlide_dClMax * (theta**2 - gamma_touchdown**2)
                + TAS_glide**2
                * (2 * theta * dTheta_dClMax - 2 * gamma_touchdown * dGammaTd_dClMax)
            )
        )
        J["flare_alt", "rho_app"] = (
            1
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * (
                2 * TAS_glide * dTasGlide_dRhoApp * (theta**2 - gamma_touchdown**2)
                + TAS_glide**2
                * (2 * theta * dTheta_dRhoApp - 2 * gamma_touchdown * dGammaTd_dRhoApp)
            )
        )
        J["flare_alt", Mission.Landing.GLIDE_TO_STALL_RATIO] = (
            1
            / (2.0 * G * (landing_flare_load_factor - 1.0))
            * (
                2 * TAS_glide * TAS_stall * (theta**2 - gamma_touchdown**2)
                + TAS_glide**2
                * (
                    2 * theta * dTheta_dGlideToStallRatio
                    - 2 * gamma_touchdown * dGammaTd_dGlideToStallRatio
                )
            )
        )


class LandingGroundRollComponent(om.ExplicitComponent):
    """
    Compute the groundroll distance and average acceleration/deceleration
    """

    def setup(self):

        self.add_input("touchdown_CD", val=0.0, units='unitless',
                       desc="CDRL: CD at touchdown")
        self.add_input("touchdown_CL", val=0.0, units='unitless',
                       desc="CLRL: CL at touchdown")

        add_aviary_input(self, Mission.Landing.STALL_VELOCITY, val=0.0)
        self.add_input(
            "TAS_touchdown", val=0.0, units="ft/s", desc="VTD: velocity at touchdown"
        )
        self.add_input(
            "thrust_idle",
            val=0.0,
            units="lbf",
            desc="TIDLE: idle thrust at start of landing",
        )
        self.add_input(
            "density_ratio", val=0.0, units="unitless", desc="DRAT: density ratio for DLAND"
        )
        self.add_input(
            "wing_loading_land",
            val=0.0,
            units="lbf/ft**2",
            desc="WOS: wing loading at landing",
        )
        self.add_input(
            "glide_distance", val=0.0, units="ft", desc="DLGL: glide distance"
        )
        self.add_input(
            "tr_distance",
            val=0.0,
            units="ft",
            desc="DLTR: distance from glide to touchdown",
        )
        self.add_input(
            "delay_distance",
            val=0.0,
            units="ft",
            desc="DDELAY: touchdown to brake application",
        )
        self.add_input(
            "CL_max", val=0.0, units="unitless", desc="CLMX: max CL at approach altitude"
        )
        self.add_input(
            Dynamic.Mission.MASS,
            val=0.0,
            units="lbm",
            desc="WL: aircraft mass at start of landing",
        )

        self.add_output(
            "ground_roll_distance",
            val=0.0,
            units="ft",
            desc="DLG: distance during braked ground roll",
        )
        add_aviary_output(self, Mission.Landing.GROUND_DISTANCE, val=0.0)
        self.add_output("average_acceleration", val=0.0, units="unitless",  # renamed from GASP var ABAR
                        desc="average acceleration/decelleration based on starting speed (TAS) and rollout distance")

        self.declare_partials(
            "ground_roll_distance",
            [
                "wing_loading_land",
                "density_ratio",
                "touchdown_CD",
                "touchdown_CL",
                "thrust_idle",
                Dynamic.Mission.MASS,
                "CL_max",
                Mission.Landing.STALL_VELOCITY,
                "TAS_touchdown",
            ],
        )
        self.declare_partials(
            Mission.Landing.GROUND_DISTANCE,
            [
                "wing_loading_land",
                "density_ratio",
                "touchdown_CD",
                "touchdown_CL",
                "thrust_idle",
                Dynamic.Mission.MASS,
                "CL_max",
                Mission.Landing.STALL_VELOCITY,
                "TAS_touchdown",
                "tr_distance",
                "delay_distance",
                "glide_distance",
            ],
        )
        self.declare_partials(
            "average_acceleration",
            [
                "TAS_touchdown",
                "wing_loading_land",
                "density_ratio",
                "touchdown_CD",
                "touchdown_CL",
                "thrust_idle",
                Dynamic.Mission.MASS,
                "CL_max",
                Mission.Landing.STALL_VELOCITY,
            ],
        )

    def compute(self, inputs, outputs):
        (
            touchdown_CD,
            touchdown_CL,
            TAS_stall,
            TAS_touchdown,
            thrust_idle,
            density_ratio,
            wing_loading_land,
            glide_distance,
            tr_distance,
            delay_distance,
            CL_max,
            mass,
        ) = inputs.values()

        weight = mass * GRAV_ENGLISH_LBM
        G = GRAV_ENGLISH_GASP
        MUB = MU_LANDING

        DLRL = touchdown_CD - (MUB * touchdown_CL)
        ARAT = DLRL / (CL_max * (TAS_stall / TAS_touchdown) ** 2)
        thrust_to_landing_weight_ratio = thrust_idle / weight
        FRAT = MUB - thrust_to_landing_weight_ratio
        ALN = np.log(FRAT / (FRAT + ARAT))

        ground_roll_distance = (
            -13.0287 * wing_loading_land * ALN / (density_ratio * DLRL)
        )
        total_distance = (
            ground_roll_distance + tr_distance + delay_distance + glide_distance
        )
        average_acceleration = TAS_touchdown**2.0 / (ground_roll_distance * 2.0 * G)

        outputs["ground_roll_distance"] = ground_roll_distance
        outputs[Mission.Landing.GROUND_DISTANCE] = total_distance
        outputs["average_acceleration"] = average_acceleration

    def compute_partials(self, inputs, J):
        (
            touchdown_CD,
            touchdown_CL,
            TAS_stall,
            TAS_touchdown,
            thrust_idle,
            density_ratio,
            wing_loading_land,
            glide_distance,
            tr_distance,
            delay_distance,
            CL_max,
            mass,
        ) = inputs.values()
        weight = mass * GRAV_ENGLISH_LBM
        G = GRAV_ENGLISH_GASP
        MUB = MU_LANDING

        DLRL = touchdown_CD - (MUB * touchdown_CL)
        ARAT = DLRL / (CL_max * (TAS_stall / TAS_touchdown) ** 2)
        thrust_to_landing_weight_ratio = thrust_idle / weight
        FRAT = MUB - thrust_to_landing_weight_ratio
        ALN = np.log(FRAT / (FRAT + ARAT))

        dDLRL_dTouchdownCD = 1
        dDLRL_dTouchdownCL = -MUB

        dARAT_dTouchdownCD = dDLRL_dTouchdownCD / (
            CL_max * (TAS_stall / TAS_touchdown) ** 2
        )
        dARAT_dTouchdownCL = dDLRL_dTouchdownCL / (
            CL_max * (TAS_stall / TAS_touchdown) ** 2
        )
        dARAT_dClMax = -DLRL / (CL_max**2 * (TAS_stall / TAS_touchdown) ** 2)
        dARAT_dTasStall = -2 * DLRL / (CL_max * (TAS_stall**3 / TAS_touchdown**2))
        dARAT_dTasTouchdown = 2 * TAS_touchdown * DLRL / (CL_max * (TAS_stall) ** 2)

        dFRAT_dThrustIdle = -1 / weight
        dFRAT_dWeight = thrust_idle / weight**2

        dALN_dTouchdownCD = (
            1
            / (FRAT / (FRAT + ARAT))
            * (-FRAT / (FRAT + ARAT) ** 2)
            * dARAT_dTouchdownCD
        )
        dALN_dTouchdownCL = (
            1
            / (FRAT / (FRAT + ARAT))
            * (-FRAT / (FRAT + ARAT) ** 2)
            * dARAT_dTouchdownCL
        )
        dALN_dThrustIdle = (
            1
            / (FRAT / (FRAT + ARAT))
            * ((FRAT + ARAT) - FRAT)
            / (FRAT + ARAT) ** 2
            * dFRAT_dThrustIdle
        )
        dALN_dWeight = (
            1
            / (FRAT / (FRAT + ARAT))
            * ((FRAT + ARAT) - FRAT)
            / (FRAT + ARAT) ** 2
            * dFRAT_dWeight
        )
        dALN_dClMax = (
            1 / (FRAT / (FRAT + ARAT)) * (-FRAT / (FRAT + ARAT) ** 2 * dARAT_dClMax)
        )
        dALN_dTasStall = (
            1 / (FRAT / (FRAT + ARAT)) * (-FRAT / (FRAT + ARAT) ** 2 * dARAT_dTasStall)
        )
        dALN_dTasTouchdown = (
            1
            / (FRAT / (FRAT + ARAT))
            * (-FRAT / (FRAT + ARAT) ** 2 * dARAT_dTasTouchdown)
        )

        ground_roll_distance = (
            -13.0287 * wing_loading_land * ALN / (density_ratio * DLRL)
        )
        dGRD_dWeight = (
            -13.0287 * wing_loading_land * dALN_dWeight / (density_ratio * DLRL)
        )

        J["ground_roll_distance", "wing_loading_land"] = dGRD_dWingLoadingLand = (
            -13.0287 * ALN / (density_ratio * DLRL)
        )
        J["ground_roll_distance", "density_ratio"] = dGRD_dDensityRatio = (
            13.0287 * wing_loading_land * ALN / (density_ratio**2 * DLRL)
        )
        J["ground_roll_distance", "touchdown_CD"] = dGRD_dTouchdownCD = (
            -13.0287
            * (wing_loading_land / density_ratio)
            * (DLRL * dALN_dTouchdownCD - ALN * dDLRL_dTouchdownCD)
            / DLRL**2
        )
        J["ground_roll_distance", "touchdown_CL"] = dGRD_dTouchdownCL = (
            -13.0287
            * (wing_loading_land / density_ratio)
            * (DLRL * dALN_dTouchdownCL - ALN * dDLRL_dTouchdownCL)
            / DLRL**2
        )
        J["ground_roll_distance", "thrust_idle"] = dGRD_dThrustIdle = (
            -13.0287 * wing_loading_land * dALN_dThrustIdle / (density_ratio * DLRL)
        )
        J["ground_roll_distance", Dynamic.Mission.MASS] = dGRD_dWeight * GRAV_ENGLISH_LBM
        J["ground_roll_distance", "CL_max"] = dGRD_dClMax = (
            -13.0287 * wing_loading_land * dALN_dClMax / (density_ratio * DLRL)
        )
        J["ground_roll_distance", Mission.Landing.STALL_VELOCITY] = dGRD_dTasStall = (
            -13.0287 * wing_loading_land * dALN_dTasStall / (density_ratio * DLRL)
        )
        J["ground_roll_distance", "TAS_touchdown"] = dGRD_dTasTouchdown = (
            -13.0287 * wing_loading_land * dALN_dTasTouchdown / (density_ratio * DLRL)
        )

        J[Mission.Landing.GROUND_DISTANCE,
            "wing_loading_land"] = dGRD_dWingLoadingLand
        J[Mission.Landing.GROUND_DISTANCE, "density_ratio"] = dGRD_dDensityRatio
        J[Mission.Landing.GROUND_DISTANCE, "touchdown_CD"] = dGRD_dTouchdownCD
        J[Mission.Landing.GROUND_DISTANCE, "touchdown_CL"] = dGRD_dTouchdownCL
        J[Mission.Landing.GROUND_DISTANCE, "thrust_idle"] = dGRD_dThrustIdle
        J[Mission.Landing.GROUND_DISTANCE, Dynamic.Mission.MASS] = \
            dGRD_dWeight * GRAV_ENGLISH_LBM
        J[Mission.Landing.GROUND_DISTANCE, "CL_max"] = dGRD_dClMax
        J[Mission.Landing.GROUND_DISTANCE,
            Mission.Landing.STALL_VELOCITY] = dGRD_dTasStall
        J[Mission.Landing.GROUND_DISTANCE, "TAS_touchdown"] = dGRD_dTasTouchdown
        J[Mission.Landing.GROUND_DISTANCE, "tr_distance"] = 1
        J[Mission.Landing.GROUND_DISTANCE, "delay_distance"] = 1
        J[Mission.Landing.GROUND_DISTANCE, "glide_distance"] = 1

        J["average_acceleration", "wing_loading_land"] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dWingLoadingLand
        )
        J["average_acceleration", "density_ratio"] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dDensityRatio
        )
        J["average_acceleration", "touchdown_CD"] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dTouchdownCD
        )
        J["average_acceleration", "touchdown_CL"] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dTouchdownCL
        )
        J["average_acceleration", "thrust_idle"] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dThrustIdle
        )
        J["average_acceleration", Dynamic.Mission.MASS] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dWeight * GRAV_ENGLISH_LBM
        )
        J["average_acceleration", "CL_max"] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dClMax
        )
        J["average_acceleration", Mission.Landing.STALL_VELOCITY] = (
            -(TAS_touchdown**2.0)
            / (ground_roll_distance**2 * 2.0 * G)
            * dGRD_dTasStall
        )
        J["average_acceleration", "TAS_touchdown"] = (
            ground_roll_distance * 2 * G * 2 * TAS_touchdown
            - TAS_touchdown**2 * 2 * G * dGRD_dTasTouchdown
        ) / (ground_roll_distance * 2.0 * G) ** 2
