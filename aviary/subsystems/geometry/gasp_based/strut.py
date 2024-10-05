import warnings

import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class StrutGeom(om.ExplicitComponent):
    """
    Computation of strut length, strut area, and strut chord for GASP-based geometry.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Aircraft.Wing.AREA, val=150)
        add_aviary_input(self, Aircraft.Strut.AREA_RATIO, val=.2)

        add_aviary_input(self, Aircraft.Strut.ATTACHMENT_LOCATION, val=0, units="ft")

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=10.0)

        self.add_output(
            "strut_y", val=30, units="ft", desc="YSTRUT: attachment location of strut"
        )
        add_aviary_output(self, Aircraft.Strut.LENGTH, val=1.0)
        add_aviary_output(self, Aircraft.Strut.AREA, val=30)
        add_aviary_output(self, Aircraft.Strut.CHORD, val=1.0)

    def setup_partials(self):

        self.declare_partials(
            "strut_y", [Aircraft.Strut.ATTACHMENT_LOCATION],
        )
        self.declare_partials(
            Aircraft.Strut.LENGTH,
            [
                Aircraft.Strut.ATTACHMENT_LOCATION,
                Aircraft.Fuselage.AVG_DIAMETER,
            ],
        )
        self.declare_partials(
            Aircraft.Strut.CHORD,
            [
                Aircraft.Wing.AREA,
                Aircraft.Strut.AREA_RATIO,
                Aircraft.Strut.ATTACHMENT_LOCATION,
                Aircraft.Fuselage.AVG_DIAMETER,
            ],
        )
        self.declare_partials(
            Aircraft.Strut.AREA,
            [
                Aircraft.Wing.AREA,
                Aircraft.Strut.AREA_RATIO
            ],
        )

    def compute(self, inputs, outputs):

        strut_x = inputs[Aircraft.Strut.ATTACHMENT_LOCATION]
        wing_area = inputs[Aircraft.Wing.AREA]
        strut_wing_area_ratio = inputs[Aircraft.Strut.AREA_RATIO]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        outputs["strut_y"] = strut_x / 2

        if outputs["strut_y"] < cabin_width / 2:
            warnings.warn(
                "Strut attachment location on wing is inside fuselage, did you mean to do that?")

        outputs[Aircraft.Strut.LENGTH] = np.sqrt(
            (outputs["strut_y"] - cabin_width / 2) ** 2 + cabin_width**2
        )
        outputs[Aircraft.Strut.AREA] = strut_area = wing_area * strut_wing_area_ratio
        outputs[Aircraft.Strut.CHORD] = 0.5 * strut_area / outputs[Aircraft.Strut.LENGTH]

    def compute_partials(self, inputs, partials):

        strut_x = inputs[Aircraft.Strut.ATTACHMENT_LOCATION]
        wing_area = inputs[Aircraft.Wing.AREA]
        strut_wing_area_ratio = inputs[Aircraft.Strut.AREA_RATIO]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        dstruty_dstrutx = partials[
            "strut_y", Aircraft.Strut.ATTACHMENT_LOCATION
        ] = (1 / 2)
        strut_y = strut_x / 2
        strut_length = np.sqrt((strut_y - cabin_width / 2) ** 2 + cabin_width**2)

        dlength_dstrutx = partials[
            Aircraft.Strut.LENGTH, Aircraft.Strut.ATTACHMENT_LOCATION
        ] = (
            0.5
            * ((strut_y - cabin_width / 2) ** 2 + cabin_width**2) ** -0.5
            * 2
            * (strut_y - cabin_width / 2)
            * dstruty_dstrutx
        )
        dlength_dwidth = partials[Aircraft.Strut.LENGTH, Aircraft.Fuselage.AVG_DIAMETER] = (
            5 * cabin_width - 2 * strut_y
        ) / (
            2
            * (
                np.sqrt(
                    5 * cabin_width**2 - 4 * cabin_width * strut_y + 4 * strut_y**2
                )
            )
        )

        partials[Aircraft.Strut.CHORD, Aircraft.Strut.ATTACHMENT_LOCATION] = (
            -1 * 0.5 * wing_area * strut_wing_area_ratio / strut_length**2 * dlength_dstrutx
        )
        partials[Aircraft.Strut.CHORD, Aircraft.Fuselage.AVG_DIAMETER] = (
            -1 * 0.5 * wing_area * strut_wing_area_ratio / strut_length**2 * dlength_dwidth
        )
        partials[Aircraft.Strut.CHORD, Aircraft.Wing.AREA] = 0.5 / \
            strut_length * strut_wing_area_ratio
        partials[Aircraft.Strut.CHORD, Aircraft.Strut.AREA_RATIO] = 0.5 / \
            strut_length * wing_area

        partials[Aircraft.Strut.AREA, Aircraft.Wing.AREA] = strut_wing_area_ratio
        partials[Aircraft.Strut.AREA, Aircraft.Strut.AREA_RATIO] = wing_area
