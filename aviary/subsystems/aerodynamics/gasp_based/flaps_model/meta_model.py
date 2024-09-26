import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import FlapType
from aviary.variable_info.variables import Aircraft, Dynamic


class MetaModelGroup(om.Group):
    """
    Group of metamodel components to interpolate intermediate calculation values for flaps model in GASP-based
    aerodynamics
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        flap_type = self.options["aviary_options"].get_val(
            Aircraft.Wing.FLAP_TYPE, units='unitless')

        # VDEL1
        VDEL1_interp = self.add_subsystem(
            "VDEL1_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "aircraft:*",
            ],
            promotes_outputs=[
                "VDEL1",
            ],
        )

        VDEL1_interp.add_input(
            Aircraft.Wing.FLAP_CHORD_RATIO,
            0.3,
            training_data=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            units="unitless",
            desc="ratio of flap chord to wing chord",
        )

        if flap_type is FlapType.PLAIN or flap_type is FlapType.SPLIT:

            VDEL1_interp.add_output(
                "VDEL1",
                1.0,
                training_data=[0.0, 0.32, 0.66, 1.0, 1.32, 1.70],
                units="unitless",
                desc="sensitivity of flap minimum drag coefficient to flap chord ratio",
            )

        else:

            VDEL1_interp.add_output(
                "VDEL1",
                1.0,
                training_data=[0.0, 0.24, 0.55, 1.00, 1.60, 2.20],
                units="unitless",
                desc="sensitivity of flap minimum drag coefficient to flap chord ratio",
            )

        # VDEL2
        VDEL2_interp = self.add_subsystem(
            "VDEL2_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "flap_defl_ratio",
            ],
            promotes_outputs=[
                "VDEL2",
            ],
        )

        VDEL2_interp.add_input(
            "flap_defl_ratio",
            0.727273,
            training_data=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0],
            units="unitless",
            desc="ratio of flap deflection to optimum flap deflection angle",
        )

        VDEL2_interp.add_output(
            "VDEL2",
            0.62455,
            training_data=[
                0.0,
                0.18,
                0.37,
                0.65,
                1.00,
                1.97,
                3.44,
                4.15,
                4.55,
                4.82,
                5.00,
            ],
            units="unitless",
            desc="sensitivity of flap minimum drag coefficient to flap angle",
        )

        # VDEL3

        VDEL3_interp = self.add_subsystem(
            "VDEL3_interp",
            om.MetaModelStructuredComp(method="scipy_slinear", extrapolate=True),
            promotes_inputs=["aircraft:*"],
            promotes_outputs=[
                "VDEL3",
            ],
        )

        VDEL3_interp.add_input(
            Aircraft.Wing.FLAP_SPAN_RATIO,
            0.65,
            training_data=[0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
            units="unitless",
            desc="BTEOB: trailing edge flap span divided by wing span",
        )

        VDEL3_interp.add_input(
            Aircraft.Wing.TAPER_RATIO,
            0.33,
            training_data=[0.0, 0.33, 1.0],
            units="unitless",
            desc="taper ratio of wing",
        )

        VDEL3_interp.add_output(
            "VDEL3",
            0.765,
            units="unitless",
            desc="sensitivity of flap minimum drag coefficient to partial flap span",
            training_data=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.28, 0.2],
                    [0.67, 0.52, 0.4],
                    [0.86, 0.72, 0.6],
                    [0.92, 0.81, 0.7],
                    [0.96, 0.88, 0.8],
                    [0.99, 0.95, 0.9],
                    [1.0, 1.0, 1.0],
                ]
            ),
        )

        # [0.,.4,.67,.86,.92,.96,.99,1.0,0.,.28,.52,.72,.81,.88,.95,1.0,0.,.2,.4,.6,.7,.8,.9,1.0]

        # VLAM1
        VLAM1_interp = self.add_subsystem(
            "VLAM1_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "aircraft:*",
            ],
            promotes_outputs=[
                "VLAM1",
            ],
        )

        VLAM1_interp.add_input(
            Aircraft.Wing.ASPECT_RATIO,
            10.13,
            training_data=[
                0.0,
                0.2,
                0.6,
                1.0,
                1.4,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.3,
                5.0,
                7.0,
                9.0,
                10.0,
                11.2,
                12.0,
                20.0,
            ],
            units="unitless",
            desc="aspect ratio",
        )

        VLAM1_interp.add_output(
            "VLAM1",
            0.97217,
            training_data=[
                0.0,
                1.36,
                1.47,
                1.49,
                1.47,
                1.24,
                0.97,
                0.91,
                0.88,
                0.87,
                0.86,
                0.87,
                0.92,
                0.96,
                0.97,
                0.99,
                1.0,
                1.0,
            ],
            units="unitless",
            desc="sensitivity of clean wing maximum lift coefficient to wing aspect ratio",
        )

        # VLAM2
        VLAM2_interp = self.add_subsystem(
            "VLAM2_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            ],
            promotes_outputs=["VLAM2"],
        )

        VLAM2_interp.add_input(
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            0.13966,
            training_data=[
                0.0,
                0.04,
                0.06,
                0.07,
                0.08,
                0.10,
                0.11,
                0.12,
                0.14,
                0.15,
                0.16,
                0.18,
                0.20,
                0.22,
                0.24,
                0.28,
            ],
            units="unitless",
            desc="average wing thickness to chord ratio",
        )

        VLAM2_interp.add_output(
            "VLAM2",
            1.09948,
            training_data=[
                0.8,
                0.82,
                0.84,
                0.85,
                0.88,
                1.00,
                1.05,
                1.07,
                1.10,
                1.11,
                1.11,
                1.10,
                1.07,
                1.02,
                0.96,
                0.80,
            ],
            units="unitless",
            desc="sensitivity of clean wing maximum lift coefficient to wing thickness to chord ratio",
        )

        # VLAM3
        VLAM3_interp = self.add_subsystem(
            "VLAM3_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "aircraft:*",
            ],
            promotes_outputs=["VLAM3"],
        )

        VLAM3_interp.add_input(
            Aircraft.Wing.ASPECT_RATIO,
            10.13,
            training_data=[
                0.0,
                0.2,
                0.6,
                1.0,
                1.4,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.3,
                5.0,
                7.0,
                9.0,
                10.0,
                11.2,
                12.0,
                20.0,
            ],
            units="unitless",
            desc="aspect ratio",
        )

        VLAM3_interp.add_output(
            "VLAM3",
            0.97217,
            training_data=[
                0.0,
                0.1,
                0.24,
                0.33,
                0.41,
                0.50,
                0.56,
                0.61,
                0.66,
                0.70,
                0.72,
                0.77,
                0.88,
                0.95,
                0.97,
                0.99,
                1.0,
                1.0,
            ],
            units="unitless",
            desc="sensitivity of flap clean wing maximum lift coefficient to wing aspect ratio",
        )

        # VLAM4
        VLAM4_interp = self.add_subsystem(
            "VLAM4_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            ],
            promotes_outputs=[
                "VLAM4",
            ],
        )

        VLAM4_interp.add_input(
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            0.13966,
            training_data=[
                0.0,
                0.04,
                0.06,
                0.07,
                0.08,
                0.10,
                0.11,
                0.12,
                0.14,
                0.15,
                0.16,
                0.18,
                0.20,
                0.22,
                0.24,
                0.28,
            ],
            units="unitless",
            desc="average wing thickness to chord ratio",
        )

        if flap_type is FlapType.PLAIN or flap_type is FlapType.SPLIT:

            VLAM4_interp.add_output(
                "VLAM4",
                1.19742,
                training_data=[
                    1.25,
                    1.17,
                    1.08,
                    1.05,
                    1.02,
                    1.00,
                    1.02,
                    1.05,
                    1.20,
                    1.36,
                    1.60,
                    1.87,
                    2.02,
                    2.12,
                    2.18,
                    2.20,
                ],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient slope to wing thickness",
            )

        else:

            VLAM4_interp.add_output(
                "VLAM4",
                1.25725,
                training_data=[
                    0.84,
                    0.86,
                    0.89,
                    0.91,
                    0.94,
                    1.00,
                    1.04,
                    1.10,
                    1.26,
                    1.33,
                    1.39,
                    1.49,
                    1.55,
                    1.58,
                    1.59,
                    1.60,
                ],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient slope to wing thickness",
            )

        # VLAM5
        VLAM5_interp = self.add_subsystem(
            "VLAM5_intep",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "aircraft:*",
            ],
            promotes_outputs=["VLAM5"],
        )

        VLAM5_interp.add_input(
            Aircraft.Wing.FLAP_CHORD_RATIO,
            0.3,
            training_data=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            units="unitless",
            desc="ratio of flap chord to wing chord",
        )

        if flap_type is FlapType.PLAIN or flap_type is FlapType.SPLIT:

            VLAM5_interp.add_output(
                "VLAM5",
                1.0,
                training_data=[0.0, 0.72, 0.94, 1.00, 0.95, 0.73],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient to wing flap to chord ratio",
            )

        elif (
            flap_type is FlapType.SINGLE_SLOTTED
            or flap_type is FlapType.DOUBLE_SLOTTED
            or flap_type is FlapType.TRIPLE_SLOTTED
        ):

            VLAM5_interp.add_output(
                "VLAM5",
                1.0,
                training_data=[0.0, 0.575, 0.83, 1.00, 1.065, 1.09],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient to wing flap to chord ratio",
            )

        else:

            VLAM5_interp.add_output(
                "VLAM5",
                1.0,
                training_data=[0.0, 0.41, 0.73, 1.00, 1.22, 1.40],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient to wing flap to chord ratio",
            )

        # VLAM6
        VLAM6_interp = self.add_subsystem(
            "VLAM6_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "flap_defl",
            ],
            promotes_outputs=[
                "VLAM6",
            ],
        )

        VLAM6_interp.add_input(
            "flap_defl",
            10.0,
            training_data=[
                0.0,
                5.0,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                38.0,
                40.0,
                42.0,
                44.0,
                50.0,
                55.0,
                60.0,
            ],
            units="deg",
            desc="flap deflection",
        )

        if flap_type is FlapType.PLAIN or flap_type is FlapType.SPLIT:

            VLAM6_interp.add_output(
                "VLAM6",
                0.8,
                training_data=[
                    0.0,
                    0.12,
                    0.23,
                    0.34,
                    0.43,
                    0.53,
                    0.62,
                    0.71,
                    0.76,
                    0.80,
                    0.82,
                    0.86,
                    0.94,
                    0.98,
                    1.0,
                ],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient to wing flap deflection",
            )

        elif (
            flap_type is FlapType.SINGLE_SLOTTED
            or flap_type is FlapType.DOUBLE_SLOTTED
            or flap_type is FlapType.TRIPLE_SLOTTED
        ):

            VLAM6_interp.add_output(
                "VLAM6",
                1.0,
                training_data=[
                    0.0,
                    0.22,
                    0.41,
                    0.57,
                    0.71,
                    0.83,
                    0.91,
                    0.975,
                    0.995,
                    1.0,
                    0.997,
                    0.992,
                    0.945,
                    0.85,
                    0.75,
                ],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient to wing flap deflection",
            )

        elif (flap_type is FlapType.FOWLER or flap_type is FlapType.DOUBLE_SLOTTED_FOWLER):

            VLAM6_interp.add_output(
                "VLAM6",
                1.11,
                training_data=[
                    0.0,
                    0.25,
                    0.46,
                    0.65,
                    0.80,
                    0.92,
                    1.00,
                    1.07,
                    1.10,
                    1.11,
                    1.10,
                    1.07,
                    0.85,
                    0.56,
                    0.20,
                ],
                units="unitless",
                desc="sensitivity of flap clean wing maximum lift coefficient to wing flap deflection",
            )

        else:
            raise ValueError(flap_type + ' is not a valid flap type')

        # VLAM7
        VLAM7_interp = self.add_subsystem(
            "VLAM7_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                Aircraft.Wing.FLAP_SPAN_RATIO,
            ],
            promotes_outputs=[
                "VLAM7",
            ],
        )

        VLAM7_interp.add_input(
            Aircraft.Wing.FLAP_SPAN_RATIO,
            0.65,
            training_data=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
            units="unitless",
            desc="BTEOB: trailing edge flap span divided by wing span",
        )

        VLAM7_interp.add_output(
            "VLAM7",
            0.735,
            training_data=[0.0, 0.25, 0.47, 0.69, 0.87, 0.94, 1.00],
            units="unitless",
            desc="sensitivity of flap clean wing maximum lift coefficient to wing flap span",
        )

        # VLAM10
        VLAM10_interp = self.add_subsystem(
            "VLAM10_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "slat_defl_ratio",
            ],
            promotes_outputs=[
                "VLAM10",
            ],
        )

        VLAM10_interp.add_input(
            "slat_defl_ratio",
            0.5,
            training_data=[
                0.0,
                0.2,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.4,
                1.6,
                1.7,
            ],
            units="unitless",
            desc="Ratio of leading edge slat deflection to optimum deflection angle",
        )

        VLAM10_interp.add_output(
            "VLAM10",
            0.74,
            training_data=[
                0.0,
                0.34,
                0.62,
                0.74,
                0.83,
                0.90,
                0.96,
                0.99,
                1.00,
                0.99,
                0.96,
                0.81,
                0.49,
                0.22,
            ],
            units="unitless",
            desc="sensitivity of clean wing maximum lift coefficient to slat deflection angle",
        )

        # VLAM11
        VLAM11_interp = self.add_subsystem(
            "VLAM11_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                Aircraft.Wing.SLAT_SPAN_RATIO,
            ],
            promotes_outputs=[
                "VLAM11",
            ],
        )

        VLAM11_interp.add_input(
            Aircraft.Wing.SLAT_SPAN_RATIO,
            0.89759553,
            training_data=[0.0, 0.2, 0.3, 0.4, 0.47, 0.5, 1.0],
            units="unitless",
            desc="ratio of leading edge slat span to wing span",
        )

        VLAM11_interp.add_output(
            "VLAM11",
            0.84232,
            training_data=[0.0, 0.05, 0.09, 0.15, 0.20, 0.23, 1.00],
            units="unitless",
            desc="sensitivity of slat clean wing maximum lift coefficient to slat span",
        )

        # VLAM13
        VLAM13_interp = self.add_subsystem(
            "VLAM13_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                "reynolds",
            ],
            promotes_outputs=[
                "VLAM13",
            ],
        )

        VLAM13_interp.add_input(
            "reynolds",
            val=157.1111,
            training_data=[
                1.0,
                2.0,
                5.0,
                10.0,
                30.0,
                60.0,
                90.0,
                120.0,
                170.0,
                250.0,
                300.0,
                500.0,
                1000.0,
                10000.0,
            ],
            units="unitless",
            desc="reynolds number",
        )

        VLAM13_interp.add_output(
            "VLAM13",
            1.03512,
            training_data=[
                0.70,
                0.70,
                0.75,
                0.81,
                0.925,
                1.0,
                1.04,
                1.05,
                1.03,
                1.00,
                0.98,
                0.93,
                0.90,
                0.90,
            ],
            units="unitless",
            desc="reynolds number correction factor",
        )

        # VLAM14
        VLAM14_interp = self.add_subsystem(
            "VLAM14_interp",
            om.MetaModelStructuredComp(method="1D-slinear", extrapolate=True),
            promotes_inputs=[
                Dynamic.Mission.MACH,
            ],
            promotes_outputs=[
                "VLAM14",
            ],
        )

        VLAM14_interp.add_input(
            Dynamic.Mission.MACH,
            0.17522,
            training_data=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            units="unitless",
            desc="mach number",
        )

        VLAM14_interp.add_output(
            "VLAM14",
            0.99124,
            training_data=[1.0, 0.99, 0.94, 0.87, 0.78, 0.66],
            units="unitless",
            desc="mach number correction factor",
            ref=100,
        )

        # fus_lift
        fus_lift_interp = self.add_subsystem(
            "fus_lift_interp",
            om.MetaModelStructuredComp(method="scipy_slinear", extrapolate=True),
            promotes_inputs=["body_to_span_ratio", "chord_to_body_ratio"],
            promotes_outputs=[
                "fus_lift",
            ],
        )

        fus_lift_interp.add_input(
            "body_to_span_ratio",
            0.09240447,
            training_data=[0.0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            units="unitless",
            desc="trailing edge flap span divided by wing span",
        )

        fus_lift_interp.add_input(
            "chord_to_body_ratio",
            0.12679,
            training_data=[0.1, 0.2, 0.3, 0.4, 0.5],
            units="unitless",
            desc="taper ratio of wing",
        )

        fus_lift_interp.add_output(
            "fus_lift",
            0.05498,
            units="unitless",
            desc="sensitivity of flap minimum drag coefficient to partial flap span",
            training_data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.046, 0.018, -0.002, -0.009, -0.025],
                    [0.070, 0.025, -0.007, -0.030, -0.048],
                    [0.076, 0.026, -0.010, -0.038, -0.057],
                    [0.080, 0.023, -0.018, -0.051, -0.070],
                    [0.073, 0.004, -0.035, -0.073, -0.090],
                    [0.053, -0.022, -0.060, -0.094, -0.109],
                    [0.030, -0.047, -0.084, -0.112, -0.126],
                    [-0.018, -0.094, -0.126, -0.145, -0.155],
                    [-0.068, -0.130, -0.160, -0.172, -0.180],
                ]
            ),
        )
