import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class CLmaxCalculation(om.ExplicitComponent):
    """
    CL_max calculation for GASP-based aerodynamics
    """

    def setup(self):

        # inputs

        # from component 1 outputs
        self.add_input(
            "VLAM8",
            val=0.74444322,
            units='unitless',
            desc="VLAM8: sensitivity of flap clean wing maximum lift coefficient to wing sweep angle",
        )
        self.add_input(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=1118.21948771,
            units="ft/s",
            desc="SA: speed of sound at sea level",
        )

        # from component 3 outputs
        add_aviary_input(self, Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, val=1.5)
        self.add_input(
            "VLAM1",
            val=0.97217,
            units='unitless',
            desc="VLAM1: sensitivity of clean wing maximum lift coefficient to wing aspect ratio",
        )
        self.add_input(
            "VLAM2",
            val=1.09948,
            units='unitless',
            desc="VLAM2: sensitivity of clean wing maximum lift coefficient to wing thickness to chord ratio",
        )
        self.add_input(
            "VLAM3",
            val=0.97217,
            units='unitless',
            desc="VLAM3: sensitivity of flap clean wing maximum lift coefficient to aspect ratio",
        )
        self.add_input(
            "VLAM4",
            val=1.25725,
            units='unitless',
            desc="VLAM4: sensitivity of flap clean wing maximum lift coefficient slope to wing thickness",
        )
        self.add_input(
            "VLAM5",
            val=1.0,
            units='unitless',
            desc="VLAM5: sensitivity of flap clean wing maximum lift coefficient to wing flap to chord ratio",
        )
        self.add_input(
            "VLAM6",
            val=1.0,
            units='unitless',
            desc="VLAM6: sensitivity of flap clean wing maximum lift coefficient to wing flap deflection",
        )
        self.add_input(
            "VLAM7",
            val=0.735,
            units='unitless',
            desc="VLAM7: sensitivity of flap clean wing maximum lift coefficient to wing flap span",
        )
        self.add_input(
            "VLAM13", val=1.03512, units='unitless', desc="VLAM13: reynolds number correction factor"
        )
        self.add_input(
            "VLAM14", val=0.99124, units='unitless', desc="VLAM14: mach number correction factor "
        )

        # other inputs

        add_aviary_input(self, Aircraft.Wing.LOADING, val=128)

        self.add_input(
            Dynamic.Mission.STATIC_PRESSURE, val=(14.696 * 144), units="lbf/ft**2", desc="P0: static pressure"
        )

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, val=12.61)

        add_aviary_input(self, Aircraft.Wing.MAX_LIFT_REF, val=1.15)
        add_aviary_input(self, Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, val=0.93)
        self.add_input(
            "VLAM9",
            val=0.9975,
            units='unitless',
            desc="VLAM9: sensitivity of slat clean wing maximum lift coefficient to slat chord",
        )
        self.add_input(
            "VLAM10",
            val=0.74,
            units='unitless',
            desc="VLAM10: sensitivity of slat clean wing maximum lift coefficient to slat deflection angle",
        )
        self.add_input(
            "VLAM11",
            val=0.84232,
            units='unitless',
            desc="VLAM11: sensitivity of slat clean wing mazimum lift coefficient to slat span",
        )
        self.add_input(
            "VLAM12",
            val=0.79208,
            units='unitless',
            desc="VLAM12: sensitivity of slat clean wing maximum lift coefficient to leading edge sweepback",
        )
        self.add_input("fus_lift", val=0.05498, units='unitless',
                       desc="DELCLF: fuselage lift increment")
        self.add_input(
            Dynamic.Mission.KINEMATIC_VISCOSITY,
            val=0.15723e-03,
            units="ft**2/s",
            desc="XKV: kinematic viscosity",
        )
        self.add_input(
            Dynamic.Mission.TEMPERATURE,
            val=518.67,
            units="degR",
            desc="T0: static temperature of air cross wing",
        )

        # outputs

        self.add_output("CL_max", val=2.8155,
                        desc="CLMAX: maximum lift coefficient", units="unitless")
        self.add_output(Dynamic.Mission.MACH, val=0.17522,
                        units='unitless', desc="SMN: mach number")
        self.add_output("reynolds", val=157.1111, units='unitless',
                        desc="RNW: reynolds number")

    def setup_partials(self):

        self.declare_partials(
            "CL_max",
            [
                Aircraft.Wing.MAX_LIFT_REF,
                "VLAM1",
                "VLAM2",
                Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM8",
                Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM,
                "VLAM9",
                "VLAM10",
                "VLAM11",
                "VLAM12",
                "VLAM13",
                "VLAM14",
                "fus_lift",
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            Dynamic.Mission.MACH,
            [
                Aircraft.Wing.LOADING,
                Dynamic.Mission.STATIC_PRESSURE,
                Aircraft.Wing.MAX_LIFT_REF,
                "VLAM1",
                "VLAM2",
                Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM8",
                Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM,
                "VLAM9",
                "VLAM10",
                "VLAM11",
                "VLAM12",
                "VLAM13",
                "VLAM14",
                "fus_lift",
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "reynolds",
            [
                Dynamic.Mission.KINEMATIC_VISCOSITY,
                Dynamic.Mission.SPEED_OF_SOUND,
                Aircraft.Wing.AVERAGE_CHORD,
                Dynamic.Mission.STATIC_PRESSURE,
                Aircraft.Wing.LOADING,
                Aircraft.Wing.MAX_LIFT_REF,
                "VLAM1",
                "VLAM2",
                Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM8",
                Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM,
                "VLAM9",
                "VLAM10",
                "VLAM11",
                "VLAM12",
                "VLAM13",
                "VLAM14",
                "fus_lift",
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )

    def compute(self, inputs, outputs):

        VLAM1 = inputs["VLAM1"]
        VLAM2 = inputs["VLAM2"]
        VLAM3 = inputs["VLAM3"]
        VLAM4 = inputs["VLAM4"]
        VLAM5 = inputs["VLAM5"]
        VLAM6 = inputs["VLAM6"]
        VLAM7 = inputs["VLAM7"]
        VLAM8 = inputs["VLAM8"]
        VLAM9 = inputs["VLAM9"]
        VLAM10 = inputs["VLAM10"]
        VLAM11 = inputs["VLAM11"]
        VLAM12 = inputs["VLAM12"]
        VLAM13 = inputs["VLAM13"]
        VLAM14 = inputs["VLAM14"]

        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]
        wing_loading = inputs[Aircraft.Wing.LOADING]
        P = inputs[Dynamic.Mission.STATIC_PRESSURE]
        avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
        kinematic_viscosity = inputs[Dynamic.Mission.KINEMATIC_VISCOSITY]
        max_lift_reference = inputs[Aircraft.Wing.MAX_LIFT_REF]
        leading_lift_increment = inputs[Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM]
        fus_lift = inputs["fus_lift"]
        trailing_lift_increment = inputs[Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM]

        # outputs

        outputs["CL_max"] = CL_max = (
            max_lift_reference * VLAM1 * VLAM2
            + trailing_lift_increment * VLAM3 * VLAM4 * VLAM5 * VLAM6 * VLAM7 * VLAM8
            + leading_lift_increment * VLAM9 * VLAM10 * VLAM11 * VLAM12
        ) * VLAM13 * VLAM14 + fus_lift

        Q1 = wing_loading / CL_max

        outputs[Dynamic.Mission.MACH] = mach = (Q1 / 0.7 / P) ** 0.5

        VK = mach * sos
        outputs["reynolds"] = reynolds = (avg_chord * VK / kinematic_viscosity) / 100000
