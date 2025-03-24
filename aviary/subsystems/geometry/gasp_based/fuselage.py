import numpy as np
import openmdao.api as om

from aviary.utils.functions import sigmoidX
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft, Settings


class FuselageParameters(om.ExplicitComponent):
    """
    Computation of average fuselage diameter, cabin height, cabin length and nose height.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Fuselage.AISLE_WIDTH, units='inch')
        add_aviary_option(self, Aircraft.Fuselage.NUM_AISLES)
        add_aviary_option(self, Aircraft.Fuselage.NUM_SEATS_ABREAST)
        add_aviary_option(self, Aircraft.Fuselage.SEAT_PITCH, units='inch')
        add_aviary_option(self, Aircraft.Fuselage.SEAT_WIDTH, units='inch')
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):

        add_aviary_input(self, Aircraft.Fuselage.DELTA_DIAMETER, units='ft')
        # add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, units='inch')
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
        options = self.options
        verbosity = options[Settings.VERBOSITY]
        seats_abreast = options[Aircraft.Fuselage.NUM_SEATS_ABREAST]
        seat_width, _ = options[Aircraft.Fuselage.SEAT_WIDTH]
        num_aisle = options[Aircraft.Fuselage.NUM_AISLES]
        aisle_width, _ = options[Aircraft.Fuselage.AISLE_WIDTH]
        PAX = options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        seat_pitch, _ = options[Aircraft.Fuselage.SEAT_PITCH]

        delta_diameter = inputs[Aircraft.Fuselage.DELTA_DIAMETER]

        cabin_width = seats_abreast * seat_width + num_aisle * aisle_width + 12

        if PAX < 1:
            if verbosity >= Verbosity.BRIEF:
                print("Warning: you have not specified at least one passenger")

        # single seat across
        cabin_len_a = PAX * seat_pitch / 12
        nose_width_a = cabin_width / 12
        nose_height_a = nose_width_a
        cabin_height_a = nose_height_a + delta_diameter

        # multiple seats across, assuming no first class seats
        cabin_len_b = (PAX - 1) * seat_pitch / (seats_abreast * 12)
        cabin_width_b = cabin_width / 12
        cabin_height_b = cabin_width_b
        nose_height_b = cabin_height_b - delta_diameter

        outputs[Aircraft.Fuselage.AVG_DIAMETER] = cabin_width
        # There are separate equations for aircraft with a single seat per row vs. multiple seats per row.
        # Here and in compute_partials, these equations are smoothed using a sigmoid fnuction centered at
        # 1.5 seats, the sigmoid function is steep enough that there should be no noticable difference
        # between the smoothed function and the stepwise function at 1 and 2 seats.
        sig1 = sigmoidX(seats_abreast, 1.5, -0.01)
        sig2 = sigmoidX(seats_abreast, 1.5, 0.01)
        outputs["cabin_height"] = cabin_height_a * sig1 + cabin_height_b*sig2
        outputs["cabin_len"] = cabin_len_a * sig1 + cabin_len_b*sig2
        outputs["nose_height"] = nose_height_a * sig1 + nose_height_b*sig2

    def compute_partials(self, inputs, J):
        options = self.options
        seats_abreast = options[Aircraft.Fuselage.NUM_SEATS_ABREAST]

        J["nose_height", Aircraft.Fuselage.DELTA_DIAMETER] = -sigmoidX(
            seats_abreast, 1.5, 0.01)
        J["cabin_height", Aircraft.Fuselage.DELTA_DIAMETER] = sigmoidX(
            seats_abreast, 1.5, -0.01)


class FuselageSize(om.ExplicitComponent):
    """
    Computation of fuselage length, fuselage wetted area, and cabin length
    for the tail boom fuselage.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.NOSE_FINENESS, units='unitless')
        self.add_input("nose_height", val=0, units="ft", desc="HN: height of nose")
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, units='ft')
        self.add_input("cabin_len", val=0, units="ft", desc="LC: length of cabin")
        add_aviary_input(self, Aircraft.Fuselage.TAIL_FINENESS, units='unitless')
        self.add_input("cabin_height", val=0, units="ft", desc="HC: height of cabin")
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_output(self, Aircraft.TailBoom.LENGTH, units='ft')

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
        # length to diameter ratio of nose cone of fuselage
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

    def setup(self):

        # outputs from parameters that are used in size but not outside of this group
        connected_input_outputs = ["cabin_height", "cabin_len", "nose_height"]

        parameters = self.add_subsystem(
            "parameters",
            FuselageParameters(),
            promotes_inputs=["aircraft:*"],
            promotes_outputs=["aircraft:*"] + connected_input_outputs,
        )

        size = self.add_subsystem(
            "size",
            FuselageSize(),
            promotes_inputs=connected_input_outputs + ["aircraft:*"],
            promotes_outputs=["aircraft:*"],
        )


class BWBFuselageParameters(om.ExplicitComponent):
    """
    Computation of average fuselage diameter, cabin height, cabin length and nose height for BWB.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Fuselage.AISLE_WIDTH, units='inch')
        add_aviary_option(self, Aircraft.Fuselage.NUM_AISLES)
        add_aviary_option(self, Aircraft.Fuselage.NUM_SEATS_ABREAST)
        add_aviary_option(self, Aircraft.Fuselage.SEAT_PITCH, units='inch')
        add_aviary_option(self, Aircraft.Fuselage.SEAT_WIDTH, units='inch')
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        pass


class BWBCabinLayout(om.ExplicitComponent):
    """
    layout of passenger cabin for BWB
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.NUM_SEATS_ABREAST,
                          units='unitless', desc='INGASP.SAB')
        add_aviary_option(self, Aircraft.Fuselage.SEAT_WIDTH,
                          units='inch', desc='INGASP.WS')
        add_aviary_option(self, Aircraft.Fuselage.NUM_AISLES,
                          units='unitless', desc='INGASP.AS')
        add_aviary_option(self, Aircraft.Fuselage.AISLE_WIDTH,
                          units='inch', desc='INGASP.WAS')
        add_aviary_option(self, Aircraft.Fuselage.SEAT_PITCH,
                          units='inch', desc='INGASP.PS')
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS,
                          units='unitless', desc='INGASP.PAX')
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
                          units='unitless', desc='equiv INGASP.PCT_FC')
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                         units='deg')
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                         units='ft')
        self.add_input('nose_length', units='ft')

        self.add_output('fuselage_station_aft', units='ft',
                        desc='EL_AFT: fuselage station of aft pressure bulkhead')

        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        options = self.options
        verbosity = options[Settings.VERBOSITY]
        rad2deg = 180. / np.pi

        # Hard code variables in GASP:
        FC_lav_galley_length = 8.0  # EL_FLGC: length of first class lav, galley & closet, ft
        FC_seat_width = 28.0  # WS_FC: first class seat width, inch
        FC_seat_pitch = 36.0  # PS_FC: first class seat pitch, inch
        FC_num_aisles = 2  # AS_FC: num of aisles in first class
        FC_aisle_width = 24.0  # WAS_FC: First class aisle width, inch
        length_FC_to_TC = 5.0  # Length of first class/tourist class aisle
        TC_num_pax_per_lav = 78  # NLAVTC: tourist class passengers per lav
        TC_lav_width = 42.0  # WIDLAV: Lav width, inches
        TC_galley_area_per_pax = 0.15  # AGAL_TC: tourist class galley area per passenger, ft**2
        # If there is no first class cabin, please set NUM_FIRST_CLASS = 0.

        num_seat_abreast = options[Aircraft.Fuselage.NUM_SEATS_ABREAST]
        TC_seat_pitch, _ = options[Aircraft.Fuselage.SEAT_PITCH]
        seat_width, _ = options[Aircraft.Fuselage.SEAT_WIDTH]
        if seat_width <= 0.0:
            raise ValueError('fuselage seat width must be positive.')
        aisle_width, _ = options[Aircraft.Fuselage.AISLE_WIDTH]
        num_aisles = options[Aircraft.Fuselage.NUM_AISLES]

        cabin_width = (num_seat_abreast * seat_width +
                       num_aisles * aisle_width) / 12.0 + 1.

        sweep_FB = inputs[Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP][0]
        pax = options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        pax_FC = options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        if pax_FC <= 0:
            if verbosity > Verbosity.BRIEF:
                print(
                    "Warning: There are no first class passengers. First class cabin is not included.")
        if pax_FC > pax:
            raise ValueError(
                'Number of first class passengers must not exceed the total number of passengers.')
        pax_TC = pax - pax_FC

        nose_length = inputs['nose_length'][0]
        pilot_com_length = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH][0]
        if pax_FC > 0:
            fwd_pax_fuselage_station = nose_length + pilot_com_length + FC_lav_galley_length
        else:
            fwd_pax_fuselage_station = nose_length + pilot_com_length

        # First Class
        length_FC_by_row = []  # length in first class, ft
        width_FC_by_row = []  # width in first class, ft
        num_seats_FC_by_row = []  # num of seats in first class
        sum_num_seats_FC = 0
        EL_FC_last_row = 0.0
        if pax_FC > 0:
            EL_FC_ptr = fwd_pax_fuselage_station - FC_seat_pitch / 12.0

            Idx_row_FC = 0
            while sum_num_seats_FC < pax_FC:
                len = EL_FC_ptr + FC_seat_pitch / 12.0
                length_FC_by_row.append(len)
                wid = 2.0 * length_FC_by_row[Idx_row_FC] / np.tan(sweep_FB / rad2deg)
                width_FC_by_row.append(wid)
                if width_FC_by_row[Idx_row_FC] > cabin_width:
                    width_FC_by_row[Idx_row_FC] = cabin_width
                wid_aisle = FC_num_aisles * FC_aisle_width / 12.0
                num = int(
                    (width_FC_by_row[Idx_row_FC] - wid_aisle) / FC_seat_width * 12.0
                )
                num_seats_FC_by_row.append(num)
                prev_sum_num_seats_FC = sum_num_seats_FC
                sum_num_seats_FC = sum_num_seats_FC + num_seats_FC_by_row[Idx_row_FC]
                if sum_num_seats_FC > pax_FC:
                    break
                EL_FC_ptr = length_FC_by_row[Idx_row_FC]
                Idx_row_FC = Idx_row_FC + 1

            # Last row of first class
            EL_FC_last_row = length_FC_by_row[Idx_row_FC]
            num_seats_last_row = pax_FC - prev_sum_num_seats_FC
            # If only one seat in last row, delete last row & assume seat located forward
            if num_seats_last_row < 2:
                EL_FC_last_row = length_FC_by_row[Idx_row_FC - 1]
                EL_FC_ptr = length_FC_by_row[Idx_row_FC - 1]

            sum_num_seats_FC = pax_FC
        else:
            # If not first class
            EL_FC_last_row = 0

        # First Class/Tourist Class Aisle
        if pax_FC > 0:
            EL_TC_ptr = EL_FC_last_row + FC_seat_pitch / 12.0
        else:
            EL_TC_ptr = fwd_pax_fuselage_station

        # Tourist Class
        length_TC_by_row = []  # length in tourist class, ft
        width_TC_by_row = []  # width in tourist class, ft
        num_seats_TC_by_row = []  # num of seats in tourist class
        sum_num_seats_TC = 0
        if pax_FC > 0:
            EL_TC_ptr = EL_TC_ptr + length_FC_to_TC - TC_seat_pitch / 12.0
        else:
            EL_TC_ptr = EL_TC_ptr - TC_seat_pitch / 12.0
        Idx_row_TC = 0
        while sum_num_seats_TC < pax_TC:
            len = EL_TC_ptr + TC_seat_pitch / 12.0
            length_TC_by_row.append(len)
            wid = 2.0 * length_TC_by_row[Idx_row_TC] / np.tan(sweep_FB / rad2deg)
            width_TC_by_row.append(wid)
            if width_TC_by_row[Idx_row_TC] > cabin_width:
                width_TC_by_row[Idx_row_TC] = cabin_width
            width_aisle = num_aisles * aisle_width / 12.0
            num = int(
                (width_TC_by_row[Idx_row_TC] - width_aisle) / (seat_width / 12.0)
            )
            num_seats_TC_by_row.append(num)
            prev_num_seats_TC = sum_num_seats_TC
            sum_num_seats_TC = sum_num_seats_TC + num_seats_TC_by_row[Idx_row_TC]
            if sum_num_seats_TC >= pax_TC:
                break
            EL_TC_ptr = length_TC_by_row[Idx_row_TC]
            Idx_row_TC = Idx_row_TC + 1

        sum_num_seats_TC = pax_TC
        # last row in tourist class: find number of seats in last row
        num_seats_last_row = sum_num_seats_TC - prev_num_seats_TC
        # find width available for last row for lavs/galleys (Assumes Steward's seat in TC aisle)
        width_last_row = num_seats_last_row * seat_width / 12.0
        width_aisle = num_aisles * aisle_width / 12.0
        wid_last_row_avail = cabin_width - width_last_row - width_aisle
        # find number of tourist class lavs and aft galley (galley width = lav width)
        num_lav_TC = int(sum_num_seats_TC / TC_num_pax_per_lav) + 1
        wid_galley = 144.0 * sum_num_seats_TC * TC_galley_area_per_pax / TC_lav_width

        # find EL_AFT = fuselage station of aft pressure bulkhead. ft
        if wid_last_row_avail >= num_lav_TC * TC_lav_width / 12.0 + wid_galley / 12.0:
            # Add lavs/gallley to last row
            EL_AFT = length_TC_by_row[Idx_row_TC] + TC_lav_width / 12.0
        else:
            # add additional row for lavs & galleys to last row
            EL_AFT = length_TC_by_row[Idx_row_TC] + \
                TC_seat_pitch / 12.0 + TC_lav_width / 12.0

        outputs['fuselage_station_aft'] = EL_AFT
