import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft


class LiftAndDragIncrements(om.ExplicitComponent):
    """
    Compute lift and drag increments from flaps for GASP-based aerodynamics
    """

    def setup(self):

        # Inputs

        self.add_input(
            "VDEL5",
            val=0.90761,
            units='unitless',
            desc="VDEL5: sensitivity of flap minimum drag coefficient to flap hinge line sweep",
        )
        self.add_input(
            "VLAM8",
            val=0.74444,
            units='unitless',
            desc="VLAM8: sensitivity of flap clean wing maximum lift coefficient to wing sweep angle",
        )
        self.add_input(
            "VDEL4",
            val=0.93578,
            units='unitless',
            desc="VDEL4: sensitivity of minimum drag coefficient to fuselage width to span ratio",
        )
        add_aviary_input(self, Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, val=0.1)
        self.add_input(
            "VDEL1",
            val=1.0,
            units='unitless',
            desc="VDEL1: sensitivity of flap minimum drag coefficient to flap chord ratio",
        )
        self.add_input(
            "VDEL2",
            val=0.62455,
            units='unitless',
            desc="VDEL2: sensitivity of flap minimum drag coefficient to flap angle",
        )
        self.add_input(
            "VDEL3",
            val=0.765,
            units='unitless',
            desc="VDEL3: sensitivity of flap minimum drag coefficient to partial flap span",
        )
        add_aviary_input(self, Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, val=1.5)
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
            val=1,
            units='unitless',
            desc="VLAM5: sensitivity of flap clean wing maximum lift coefficient to wing flap to chord ratio",
        )
        self.add_input(
            "VLAM6",
            val=1,
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

        # outputs

        self.add_output("delta_CD", val=0.0,
                        desc="DCD: increment on drag coefficient", units="unitless")
        self.add_output("delta_CL", val=0.0,
                        desc="DCL: increment on lift coefficient", units="unitless")

    def setup_partials(self):

        self.declare_partials(
            "delta_CD",
            [Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
                "VDEL1", "VDEL2", "VDEL3", "VDEL4", "VDEL5"],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "delta_CL",
            [
                Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM8",
                "VLAM13",
                "VLAM14",
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )

    def compute(self, inputs, outputs):

        delta_drag_trailing = inputs[Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM]
        trailing_lift_increment = inputs[Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM]
        VDEL1 = inputs["VDEL1"]
        VDEL2 = inputs["VDEL2"]
        VDEL3 = inputs["VDEL3"]
        VDEL4 = inputs["VDEL4"]
        VDEL5 = inputs["VDEL5"]
        VLAM3 = inputs["VLAM3"]
        VLAM4 = inputs["VLAM4"]
        VLAM5 = inputs["VLAM5"]
        VLAM6 = inputs["VLAM6"]
        VLAM7 = inputs["VLAM7"]
        VLAM8 = inputs["VLAM8"]
        VLAM13 = inputs["VLAM13"]
        VLAM14 = inputs["VLAM14"]

        outputs["delta_CD"] = delta_CD = (
            delta_drag_trailing * VDEL1 * VDEL2 * VDEL3 * VDEL4 * VDEL5
        )
        outputs["delta_CL"] = delta_CL = (
            trailing_lift_increment
            * VLAM3
            * VLAM4
            * VLAM5
            * VLAM6
            * VLAM7
            * VLAM8
            * VLAM13
            * VLAM14
        )
