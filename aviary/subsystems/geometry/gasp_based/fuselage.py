import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


def sigX(x):
    sig = 1 / (1 + np.exp(-x))

    return sig


def dSigXdX(x):
    derivative = -1 / (1 + np.exp(-x)) ** 2 * (-1 * np.exp(-x))

    return derivative


class FuselageParameters(om.ExplicitComponent):
    """
    Computation of average fuselage diameter, cabin height, cabin length and nose height.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Aircraft.Fuselage.DELTA_DIAMETER, val=4.5)
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, val=9.5)

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, val=0, units='inch')
        self.add_output("cabin_height", val=0, units="ft", desc="HC: height of cabin")
        self.add_output("cabin_len", val=0, units="ft", desc="LC: length of cabin")
        self.add_output("nose_height", val=0, units="ft", desc="HN: height of nose")

        self.declare_partials(
            "cabin_height",
            [
                Aircraft.Fuselage.DELTA_DIAMETER,
            ],
        )
        self.declare_partials(
            "nose_height",
            [
                Aircraft.Fuselage.DELTA_DIAMETER,
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options['aviary_options'].get_val(Settings.VERBOSITY)
        aviary_options: AviaryValues = self.options['aviary_options']
        seats_abreast = aviary_options.get_val(Aircraft.Fuselage.NUM_SEATS_ABREAST)
        seat_width = aviary_options.get_val(Aircraft.Fuselage.SEAT_WIDTH, units='inch')
        num_aisle = aviary_options.get_val(Aircraft.Fuselage.NUM_AISLES)
        aisle_width = aviary_options.get_val(Aircraft.Fuselage.AISLE_WIDTH, units='inch')
        PAX = self.options['aviary_options'].get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')
        seat_pitch = aviary_options.get_val(Aircraft.Fuselage.SEAT_PITCH, units='inch')
        delta_diameter = inputs[Aircraft.Fuselage.DELTA_DIAMETER]

        cabin_width = seats_abreast * seat_width + num_aisle * aisle_width + 12

        if PAX < 1:
            if verbosity >= Verbosity.BRIEF:
                print("Warning: you have not specified at least one passenger")

        # single seat across
        cabin_len_a = PAX * seat_pitch / 12
        nose_height_a = cabin_width / 12
        cabin_height_a = nose_height_a + delta_diameter

        # multiple seats across
        cabin_len_b = (PAX - 1) * seat_pitch / (seats_abreast * 12)
        cabin_height_b = cabin_width / 12
        nose_height_b = cabin_height_b - delta_diameter

        outputs[Aircraft.Fuselage.AVG_DIAMETER] = cabin_width
        # There are separate equations for aircraft with a single seat per row vs. multiple seats per row.
        # Here and in compute_partials, these equations are smoothed using a sigmoid fnuction centered at
        # 1.5 seats, the sigmoid function is steep enough that there should be no noticable difference
        # between the smoothed function and the stepwise function at 1 and 2 seats.
        outputs["cabin_height"] = cabin_height_a * sigX(100*(1.5-seats_abreast)) + \
            cabin_height_b*sigX(100*(seats_abreast-1.5))
        outputs["cabin_len"] = cabin_len_a * sigX(100*(1.5-seats_abreast)) + \
            cabin_len_b*sigX(100*(seats_abreast-1.5))
        outputs["nose_height"] = nose_height_a * sigX(100*(1.5-seats_abreast)) + \
            nose_height_b*sigX(100*(seats_abreast-1.5))

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        seats_abreast = aviary_options.get_val(Aircraft.Fuselage.NUM_SEATS_ABREAST)

        J["nose_height", Aircraft.Fuselage.DELTA_DIAMETER] = sigX(
            100*(seats_abreast-1.5))*(-1)
        J["cabin_height", Aircraft.Fuselage.DELTA_DIAMETER] = sigX(
            100*(1.5-seats_abreast))*1


class FuselageSize(om.ExplicitComponent):
    """
    Computation of fuselage length, fuselage wetted area, and cabin length
    for the tail boom fuselage. 
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Aircraft.Fuselage.NOSE_FINENESS, val=1)
        self.add_input("nose_height", val=0, units="ft", desc="HN: height of nose")
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, val=9.5)
        self.add_input("cabin_len", val=0, units="ft", desc="LC: length of cabin")
        add_aviary_input(self, Aircraft.Fuselage.TAIL_FINENESS, val=3)
        self.add_input("cabin_height", val=0, units="ft", desc="HC: height of cabin")
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA_SCALER,
                         val=1, units="unitless")

        add_aviary_output(self, Aircraft.Fuselage.LENGTH, val=0)
        add_aviary_output(self, Aircraft.Fuselage.WETTED_AREA, val=0)
        add_aviary_output(self, Aircraft.TailBoom.LENGTH, val=0)

        self.declare_partials(
            Aircraft.Fuselage.LENGTH,
            [
                Aircraft.Fuselage.NOSE_FINENESS,
                "nose_height",
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                "cabin_len",
                Aircraft.Fuselage.TAIL_FINENESS,
                "cabin_height",
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.WETTED_AREA,
            [
                Aircraft.Fuselage.WETTED_AREA_SCALER,
                "cabin_height",
                Aircraft.Fuselage.NOSE_FINENESS,
                "nose_height",
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                "cabin_len",
                Aircraft.Fuselage.TAIL_FINENESS,
            ],
        )

        self.declare_partials(
            Aircraft.TailBoom.LENGTH,
            [
                Aircraft.Fuselage.NOSE_FINENESS,
                "nose_height",
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                "cabin_len",
                Aircraft.Fuselage.TAIL_FINENESS,
                "cabin_height",
            ],
        )

    def compute(self, inputs, outputs):
        LoverD_nose = inputs[Aircraft.Fuselage.NOSE_FINENESS]
        LoverD_tail = inputs[Aircraft.Fuselage.TAIL_FINENESS]
        cockpit_len = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH]
        fus_SA_scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
        nose_height = inputs["nose_height"]
        cabin_len = inputs["cabin_len"]
        cabin_height = inputs["cabin_height"]

        fus_len = (
            LoverD_nose * nose_height
            + cockpit_len
            + cabin_len
            + LoverD_tail * cabin_height
        )

        fus_SA = cabin_height * (
            2.5 * (LoverD_nose * nose_height + cockpit_len)
            + 3.14 * cabin_len
            + 2.1 * LoverD_tail * cabin_height
        )

        fus_SA = fus_SA * fus_SA_scaler

        cabin_len_tailboom = fus_len

        outputs[Aircraft.Fuselage.LENGTH] = fus_len
        outputs[Aircraft.Fuselage.WETTED_AREA] = fus_SA
        outputs[Aircraft.TailBoom.LENGTH] = cabin_len_tailboom

    def compute_partials(self, inputs, J):
        LoverD_nose = inputs[Aircraft.Fuselage.NOSE_FINENESS]
        LoverD_tail = inputs[Aircraft.Fuselage.TAIL_FINENESS]
        nose_height = inputs["nose_height"]
        cabin_height = inputs["cabin_height"]
        fus_SA_scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
        cockpit_len = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH]
        cabin_len = inputs["cabin_len"]

        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.NOSE_FINENESS] = nose_height
        J[Aircraft.Fuselage.LENGTH, "nose_height"] = LoverD_nose
        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = 1
        J[Aircraft.Fuselage.LENGTH, "cabin_len"] = 1
        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.TAIL_FINENESS] = cabin_height
        J[Aircraft.Fuselage.LENGTH, "cabin_height"] = LoverD_tail

        J[Aircraft.Fuselage.WETTED_AREA, "cabin_height"] = fus_SA_scaler * (
            2.5 * (LoverD_nose * nose_height + cockpit_len)
            + 3.14 * cabin_len
            + 2.1 * LoverD_tail * cabin_height
            + cabin_height * 2.1 * LoverD_tail
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.NOSE_FINENESS] = (
            fus_SA_scaler * cabin_height * 2.5 * nose_height
        )
        J[Aircraft.Fuselage.WETTED_AREA, "nose_height"] = (
            fus_SA_scaler * cabin_height * 2.5 * LoverD_nose
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = (
            fus_SA_scaler * cabin_height * 2.5
        )
        J[Aircraft.Fuselage.WETTED_AREA, "cabin_len"] = (
            fus_SA_scaler * 3.14 * cabin_height
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.TAIL_FINENESS] = (
            fus_SA_scaler * 2.1 * cabin_height**2
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.WETTED_AREA_SCALER] = cabin_height * (
            2.5 * (LoverD_nose * nose_height + cockpit_len)
            + 3.14 * cabin_len
            + 2.1 * LoverD_tail * cabin_height
        )

        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.NOSE_FINENESS] = nose_height
        J[Aircraft.TailBoom.LENGTH, "nose_height"] = LoverD_nose
        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = 1
        J[Aircraft.TailBoom.LENGTH, "cabin_len"] = 1
        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.TAIL_FINENESS] = cabin_height
        J[Aircraft.TailBoom.LENGTH, "cabin_height"] = LoverD_tail


class FuselageGroup(om.Group):
    """
    Group to pull together FuselageParameters and FuselageSize.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        aviary_options = self.options['aviary_options']

        # outputs from parameters that are used in size but not outside of this group
        connected_input_outputs = ["cabin_height", "cabin_len", "nose_height"]

        parameters = self.add_subsystem(
            "parameters",
            FuselageParameters(
                aviary_options=aviary_options,
            ),
            promotes_inputs=["aircraft:*"],
            promotes_outputs=["aircraft:*"] + connected_input_outputs,
        )

        size = self.add_subsystem(
            "size",
            FuselageSize(aviary_options=aviary_options,),
            promotes_inputs=connected_input_outputs + ["aircraft:*"],
            promotes_outputs=["aircraft:*"],
        )

        self.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, val=9.5, units="ft")
