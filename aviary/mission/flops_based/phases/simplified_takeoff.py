import openmdao.api as om
from aviary.subsystems.atmosphere.atmosphere import Atmosphere

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_METRIC
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class StallSpeed(om.ExplicitComponent):
    """
    Calculates the stall speed of the aircraft using
    v_stall = (2 * weight / (density * planform_area * Cl_max)) ** 0.5
    """

    def setup(self):
        """
        Setup the inputs and output to calculate the stall speed of the aircraft.
        """

        self.add_input(
            "mass",
            val=150_000,
            units="lbm",
            desc="mass of the aircraft",
        )

        self.add_input(
            Dynamic.Mission.DENSITY,
            val=1.225,
            units="kg/m**3",
            desc="atmospheric density",
        )

        self.add_input("planform_area", val=7, units="m**2", desc="area of the wings")

        self.add_input(
            "Cl_max",
            val=2,
            units='unitless',
            desc="maximum lift coefficient",
        )

        self.add_output(
            "v_stall",
            val=0.1,
            units="m/s",
            desc="stall velocity",
        )

        self.declare_partials("v_stall", "*")

    def compute(self, inputs, outputs):

        weight = inputs["mass"] * GRAV_ENGLISH_LBM
        # # convert from pounds to newtons.
        # This is only necessary because the equation expects newtons,
        # but the mission expects pounds mass instead of pounds force.
        weight = weight*4.44822
        rho = inputs[Dynamic.Mission.DENSITY]
        S = inputs["planform_area"]
        Cl_max = inputs["Cl_max"]

        v_stall = (2 * weight / (rho * S * Cl_max)) ** 0.5

        outputs["v_stall"] = v_stall

    def compute_partials(self, inputs, J):

        weight = inputs["mass"] * GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Mission.DENSITY]
        S = inputs["planform_area"]
        Cl_max = inputs["Cl_max"]

        rad = 2 * weight / (rho * S * Cl_max)

        J["v_stall", "mass"] = 0.5 * 4.44822**.5 * \
            rad ** (-0.5) * 2 * GRAV_ENGLISH_LBM / (rho * S * Cl_max)
        J["v_stall", Dynamic.Mission.DENSITY] = (
            0.5 * 4.44822**0.5 * rad ** (-0.5) * (-2 * weight) / (rho**2 * S * Cl_max)
        )
        J["v_stall", "planform_area"] = (
            0.5 * 4.44822**.5 * rad ** (-0.5) * (-2 * weight) / (rho * S ** 2 * Cl_max)
        )
        J["v_stall", "Cl_max"] = (
            0.5 * 4.44822**.5 * rad ** (-0.5) * (-2 * weight) / (rho * S * Cl_max ** 2)
        )


class FinalTakeoffConditions(om.ExplicitComponent):
    """
    Calculate the final takeoff condition including ground distance, final velocity,
    final mass, and final altitude.
    """

    def initialize(self):
        self.options.declare("num_engines", desc="number of engines on aircraft")

    def setup(self):

        self.add_input(
            "v_stall",
            val=0.1,
            units="m/s",
            desc="stall speed of the aircraft",
        )

        add_aviary_input(self, Mission.Summary.GROSS_MASS, val=150_000)

        add_aviary_input(self, Mission.Takeoff.FUEL_SIMPLE, val=10.e3)

        self.add_input(
            Dynamic.Mission.DENSITY,
            val=1.225,
            units="kg/m**3",
            desc="atmospheric density",
        )

        add_aviary_input(self, Aircraft.Wing.AREA, val=7)

        add_aviary_input(self, Mission.Takeoff.LIFT_COEFFICIENT_MAX, val=2)

        add_aviary_input(self, Mission.Design.THRUST_TAKEOFF_PER_ENG, val=100_000)

        add_aviary_input(self, Mission.Takeoff.LIFT_OVER_DRAG, val=2)

        add_aviary_output(self, Mission.Takeoff.GROUND_DISTANCE, val=0)

        add_aviary_output(self, Mission.Takeoff.FINAL_VELOCITY,
                          val=0, units="m/s")

        add_aviary_output(self, Mission.Takeoff.FINAL_MASS, val=0)

        add_aviary_output(self, Mission.Takeoff.FINAL_ALTITUDE, val=0)

    def setup_partials(self):

        self.declare_partials(
            Mission.Takeoff.FINAL_VELOCITY,
            "v_stall",
            val=1.2309,
        )
        self.declare_partials(
            Mission.Takeoff.GROUND_DISTANCE,
            [
                Mission.Summary.GROSS_MASS,
                Dynamic.Mission.DENSITY,
                Aircraft.Wing.AREA,
                Mission.Takeoff.LIFT_COEFFICIENT_MAX,
                Mission.Design.THRUST_TAKEOFF_PER_ENG,
                Mission.Takeoff.LIFT_OVER_DRAG,
            ],
        )
        self.declare_partials(
            Mission.Takeoff.FINAL_MASS,
            Mission.Summary.GROSS_MASS,
            val=1.0,
        )
        self.declare_partials(
            Mission.Takeoff.FINAL_MASS,
            Mission.Takeoff.FUEL_SIMPLE,
            val=-1.0,
        )

    def compute(self, inputs, outputs):

        rho_SL = RHO_SEA_LEVEL_METRIC

        v_stall = inputs["v_stall"]
        gross_mass = inputs[Mission.Summary.GROSS_MASS]
        ramp_weight = gross_mass * GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Mission.DENSITY]
        S = inputs[Aircraft.Wing.AREA]
        Cl_max = inputs[Mission.Takeoff.LIFT_COEFFICIENT_MAX]
        thrust = inputs[Mission.Design.THRUST_TAKEOFF_PER_ENG]
        L_over_D = inputs[Mission.Takeoff.LIFT_OVER_DRAG]
        num_engines = self.options['num_engines']
        rho_ratio = rho / rho_SL

        # note: this is different from the paper, not entirely clear why other than rho
        # itself can be eliminated
        rolling_distance = (
            17.0
            * ramp_weight
            / (
                S
                * Cl_max
                * (
                    thrust * num_engines / ramp_weight
                    - (0.20 + 0.00550 * ramp_weight / S) / L_over_D
                )
            )
        )

        rotation_distance = 140.0 * (ramp_weight / (S * Cl_max * rho_ratio)) ** 0.5

        # The calculation below uses all engines operating. The original FLOPS calculation uses one
        # engine inoperative to calculate the field length (not the all-engines performance).
        climbout_distance = (
            140.0
            * (ramp_weight / S) ** 0.5
            / (
                1.0
                + thrust * num_engines / ramp_weight
                - 0.90 / L_over_D
            )
        )

        # The FLOPS methodology does not calculate V2. However, it assumes CL @ V2 = 0.66 * CL_max.
        # Therefore, V2 = V_stall / sqrt(0.66).
        V2 = v_stall * 1.2309

        outputs[Mission.Takeoff.GROUND_DISTANCE] = rolling_distance + \
            rotation_distance + climbout_distance
        outputs[Mission.Takeoff.FINAL_VELOCITY] = V2
        outputs[Mission.Takeoff.FINAL_MASS] = \
            gross_mass - inputs[Mission.Takeoff.FUEL_SIMPLE]
        outputs[Mission.Takeoff.FINAL_ALTITUDE] = 35

    def compute_partials(self, inputs, J):
        rho_SL = RHO_SEA_LEVEL_METRIC

        ramp_weight = inputs[Mission.Summary.GROSS_MASS] * GRAV_ENGLISH_LBM
        rho = inputs[Dynamic.Mission.DENSITY]
        S = inputs[Aircraft.Wing.AREA]
        Cl_max = inputs[Mission.Takeoff.LIFT_COEFFICIENT_MAX]
        thrust = inputs[Mission.Design.THRUST_TAKEOFF_PER_ENG]
        L_over_D = inputs[Mission.Takeoff.LIFT_OVER_DRAG]
        num_engines = self.options['num_engines']
        rho_ratio = rho / rho_SL

        den_RD = (
            S
            * Cl_max
            * (
                thrust * num_engines / ramp_weight
                - (0.20 + 0.00550 * ramp_weight / S) / L_over_D
            )
        )
        rad_Rot = ramp_weight / (S * Cl_max * rho_ratio)
        den_Cout = (
            1.0 + thrust * num_engines / ramp_weight - 0.90 / L_over_D
        )

        S * Cl_max * (
            thrust * num_engines / ramp_weight
            - (0.20 + 0.00550 * ramp_weight / S) / L_over_D
        )
        S * Cl_max * (
            -thrust * num_engines / ramp_weight ** 2
            - (0.00550 / S) / L_over_D
        )

        # dRD_dWt = (den_RD*17 - 17*ramp_weight*S*Cl_max*(-thrust*num_engines/ramp_weight**2 - (.00550/S)/L_over_D))/den_RD**2
        dRD_dM = (17 / den_RD - 17 * ramp_weight / den_RD ** 2 * S * Cl_max * (
            -thrust * num_engines / ramp_weight ** 2
            - (0.00550 / S) / L_over_D
        )) * GRAV_ENGLISH_LBM
        dRD_dS = (
            den_RD * 0
            - 17
            * ramp_weight
            * (
                Cl_max
                * (
                    thrust * num_engines / ramp_weight
                    - (0.2 + 0.0055 * ramp_weight / S) / L_over_D
                )
                + S * Cl_max * (0.0055 * ramp_weight / S ** 2 / L_over_D)
            )
        ) / den_RD ** 2
        dRD_dClMax = (
            -17
            * ramp_weight
            / den_RD ** 2
            * (
                S
                * (
                    thrust * num_engines / ramp_weight
                    - (0.20 + 0.00550 * ramp_weight / S) / L_over_D
                )
            )
        )
        dRD_dThrust = (
            -17
            * ramp_weight
            / (
                S
                * Cl_max
                * (
                    thrust * num_engines / ramp_weight
                    - (0.2 + 0.0055 * ramp_weight / S) / L_over_D
                )
            )
            ** 2
            * S
            * Cl_max
            * num_engines
            / ramp_weight
        )
        dRD_dLoD = (
            -17
            * ramp_weight
            / (
                S
                * Cl_max
                * (
                    thrust * num_engines / ramp_weight
                    - (0.2 + 0.0055 * ramp_weight / S) / L_over_D
                )
            )
            ** 2
            * S
            * Cl_max
            * (0.2 + 0.0055 * ramp_weight / S)
            / L_over_D ** 2
        )
        dRD_dRho = 0

        dRot_dM = 140 * 0.5 * \
            rad_Rot ** (-0.5) / (S * Cl_max * rho_ratio) * GRAV_ENGLISH_LBM
        dRot_dS = (
            140
            * 0.5
            * rad_Rot ** (-0.5)
            * (-ramp_weight / (S ** 2 * Cl_max * rho_ratio))
        )
        dRot_dClMax = (
            140
            * 0.5
            * rad_Rot ** (-0.5)
            * (-ramp_weight / (S * Cl_max ** 2 * rho_ratio))
        )
        dRot_dThrust = 0
        dRot_dLoD = 0
        dRot_dRho = (
            140
            * 0.5
            * rad_Rot ** (-0.5)
            * (-ramp_weight / (S * Cl_max * rho ** 2 / rho_SL))
        )

        dCout_dM = (140 * 0.5 * (ramp_weight / S) ** (-0.5) / S / den_Cout - 140 * (
            ramp_weight / S
        ) ** 0.5 / (den_Cout) ** 2 * (
            -thrust * num_engines / ramp_weight ** 2
        )) * GRAV_ENGLISH_LBM
        dCout_dS = (
            140 * 0.5 * (ramp_weight / S) ** (-0.5) * (-ramp_weight) / S ** 2 / den_Cout
        )
        dCout_dClMax = 0
        dCout_dThrust = (
            -140
            * (ramp_weight / S) ** 0.5
            / den_Cout ** 2
            * num_engines
            / ramp_weight
        )
        dCout_dLoD = (
            -140 * (ramp_weight / S) ** 0.5 / den_Cout ** 2 * 0.9 / L_over_D ** 2
        )
        dCout_dRho = 0

        J[Mission.Takeoff.GROUND_DISTANCE,
            Mission.Summary.GROSS_MASS] = dRD_dM + dRot_dM + dCout_dM
        J[Mission.Takeoff.GROUND_DISTANCE, Dynamic.Mission.DENSITY] = (
            dRD_dRho + dRot_dRho + dCout_dRho
        )
        J[Mission.Takeoff.GROUND_DISTANCE,
            Aircraft.Wing.AREA] = dRD_dS + dRot_dS + dCout_dS
        J[
            Mission.Takeoff.GROUND_DISTANCE,
            Mission.Takeoff.LIFT_COEFFICIENT_MAX
        ] = dRD_dClMax + dRot_dClMax + dCout_dClMax
        J[
            Mission.Takeoff.GROUND_DISTANCE,
            Mission.Design.THRUST_TAKEOFF_PER_ENG
        ] = dRD_dThrust + dRot_dThrust + dCout_dThrust
        J[Mission.Takeoff.GROUND_DISTANCE,
            Mission.Takeoff.LIFT_OVER_DRAG] = dRD_dLoD + dRot_dLoD + dCout_dLoD


class TakeoffGroup(om.Group):
    """
    Calculate the final takeoff condition including ground distance, final velocity,
    final mass, and final altitude with atmosphere is included. 
    """

    def initialize(self):
        self.options.declare("num_engines", desc="number of engines on aircraft")

    def setup(self):

        self.add_subsystem(
            name='atmosphere', subsys=Atmosphere(num_nodes=1), promotes=['*']
        )

        self.add_subsystem(
            "stall_speed",
            StallSpeed(),
            promotes_outputs=[
                "v_stall",
            ],
            promotes_inputs=[
                ("mass", Mission.Summary.GROSS_MASS),
                Dynamic.Mission.DENSITY,
                ('planform_area', Aircraft.Wing.AREA),
                ("Cl_max", Mission.Takeoff.LIFT_COEFFICIENT_MAX),
            ],
        )

        self.add_subsystem(
            "final_conditions",
            FinalTakeoffConditions(num_engines=self.options["num_engines"]),
            promotes_inputs=[
                "v_stall",
                Mission.Summary.GROSS_MASS,
                Dynamic.Mission.DENSITY,
                Aircraft.Wing.AREA,
                Mission.Takeoff.FUEL_SIMPLE,
                Mission.Takeoff.LIFT_COEFFICIENT_MAX,
                Mission.Design.THRUST_TAKEOFF_PER_ENG,
                Mission.Takeoff.LIFT_OVER_DRAG,
            ],
            promotes_outputs=[
                Mission.Takeoff.GROUND_DISTANCE,
                Mission.Takeoff.FINAL_VELOCITY,
                Mission.Takeoff.FINAL_MASS,
                Mission.Takeoff.FINAL_ALTITUDE,
            ],
        )

        self.add_subsystem('compute_mach', om.ExecComp('final_mach = final_velocity / speed_of_sound',
                                                       final_mach={'units': 'unitless'},
                                                       final_velocity={'units': 'm/s'},
                                                       speed_of_sound={'units': 'm/s'}),
                           promotes_inputs=[('speed_of_sound', 'speed_of_sound'),
                                            ('final_velocity', Mission.Takeoff.FINAL_VELOCITY)],
                           promotes_outputs=[('final_mach', Mission.Takeoff.FINAL_MACH)])
