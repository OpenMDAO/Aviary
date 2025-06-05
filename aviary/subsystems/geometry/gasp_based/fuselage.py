import numpy as np
import openmdao.api as om

from aviary.utils.functions import sigmoidX
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class FuselageParameters(om.ExplicitComponent):
    """Computation of average fuselage diameter, cabin height, cabin length and nose height."""

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

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, units='inch')
        self.add_output('cabin_height', val=0, units='ft', desc='HC: height of cabin')
        self.add_output('cabin_len', val=0, units='ft', desc='LC: length of cabin')
        self.add_output('nose_height', val=0, units='ft', desc='HN: height of nose')

        self.declare_partials(
            'cabin_height',
            [
                Aircraft.Fuselage.DELTA_DIAMETER,
            ],
        )
        self.declare_partials(
            'nose_height',
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
                print('Warning: you have not specified at least one passenger')

        # single seat across
        cabin_len_a = PAX * seat_pitch / 12
        nose_height_a = cabin_width / 12
        cabin_height_a = nose_height_a + delta_diameter

        # multiple seats across, assuming no first class seats
        cabin_len_b = (PAX - 1) * seat_pitch / (seats_abreast * 12)
        cabin_height_b = cabin_width / 12
        nose_height_b = cabin_height_b - delta_diameter

        outputs[Aircraft.Fuselage.AVG_DIAMETER] = cabin_width
        # There are separate equations for aircraft with a single seat per row vs. multiple seats per row.
        # Here and in compute_partials, these equations are smoothed using a sigmoid fnuction centered at
        # 1.5 seats, the sigmoid function is steep enough that there should be no noticeable difference
        # between the smoothed function and the stepwise function at 1 and 2 seats.
        sig1 = sigmoidX(seats_abreast, 1.5, -0.01)
        sig2 = sigmoidX(seats_abreast, 1.5, 0.01)
        outputs['cabin_height'] = cabin_height_a * sig1 + cabin_height_b * sig2
        outputs['cabin_len'] = cabin_len_a * sig1 + cabin_len_b * sig2
        outputs['nose_height'] = nose_height_a * sig1 + nose_height_b * sig2

    def compute_partials(self, inputs, J):
        options = self.options
        seats_abreast = options[Aircraft.Fuselage.NUM_SEATS_ABREAST]

        J['nose_height', Aircraft.Fuselage.DELTA_DIAMETER] = -sigmoidX(seats_abreast, 1.5, 0.01)
        J['cabin_height', Aircraft.Fuselage.DELTA_DIAMETER] = sigmoidX(seats_abreast, 1.5, -0.01)


class FuselageSize(om.ExplicitComponent):
    """
    Computation of fuselage length, fuselage wetted area, and cabin length
    for the tail boom fuselage.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.NOSE_FINENESS, units='unitless')
        self.add_input('nose_height', val=0, units='ft', desc='HN: height of nose')
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, units='ft')
        self.add_input('cabin_len', val=0, units='ft', desc='LC: length of cabin')
        add_aviary_input(self, Aircraft.Fuselage.TAIL_FINENESS, units='unitless')
        self.add_input('cabin_height', val=0, units='ft', desc='HC: height of cabin')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Fuselage.LENGTH, units='ft', desc='ELF')
        add_aviary_output(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_output(self, Aircraft.TailBoom.LENGTH, units='ft', desc='ELFFC')
        add_aviary_output(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2', desc='ACABIN')

        self.declare_partials(
            Aircraft.Fuselage.LENGTH,
            [
                Aircraft.Fuselage.NOSE_FINENESS,
                'nose_height',
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                'cabin_len',
                Aircraft.Fuselage.TAIL_FINENESS,
                'cabin_height',
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.WETTED_AREA,
            [
                Aircraft.Fuselage.WETTED_AREA_SCALER,
                'cabin_height',
                Aircraft.Fuselage.NOSE_FINENESS,
                'nose_height',
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                'cabin_len',
                Aircraft.Fuselage.TAIL_FINENESS,
            ],
        )

        self.declare_partials(
            Aircraft.TailBoom.LENGTH,
            [
                Aircraft.Fuselage.NOSE_FINENESS,
                'nose_height',
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                'cabin_len',
                Aircraft.Fuselage.TAIL_FINENESS,
                'cabin_height',
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.CABIN_AREA,
            ['cabin_len', 'cabin_height', Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH],
        )

    def compute(self, inputs, outputs):
        # length to diameter ratio of nose cone of fuselage
        LoverD_nose = inputs[Aircraft.Fuselage.NOSE_FINENESS]
        LoverD_tail = inputs[Aircraft.Fuselage.TAIL_FINENESS]
        cockpit_len = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH]
        fus_SA_scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
        nose_height = inputs['nose_height']
        cabin_len = inputs['cabin_len']
        cabin_height = inputs['cabin_height']

        fus_len = LoverD_nose * nose_height + cockpit_len + cabin_len + LoverD_tail * cabin_height

        fus_SA = cabin_height * (
            2.5 * (LoverD_nose * nose_height + cockpit_len)
            + 3.14 * cabin_len
            + 2.1 * LoverD_tail * cabin_height
        )

        fus_SA = fus_SA * fus_SA_scaler

        cabin_len_tailboom = fus_len

        cabin_width = cabin_height  # assume tube shape
        cabin_area = cabin_width * (cockpit_len + cabin_len)

        outputs[Aircraft.Fuselage.LENGTH] = fus_len
        outputs[Aircraft.Fuselage.WETTED_AREA] = fus_SA
        outputs[Aircraft.TailBoom.LENGTH] = cabin_len_tailboom
        outputs[Aircraft.Fuselage.CABIN_AREA] = cabin_area

    def compute_partials(self, inputs, J):
        LoverD_nose = inputs[Aircraft.Fuselage.NOSE_FINENESS]
        LoverD_tail = inputs[Aircraft.Fuselage.TAIL_FINENESS]
        nose_height = inputs['nose_height']
        cabin_height = inputs['cabin_height']
        fus_SA_scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
        cockpit_len = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH]
        cabin_len = inputs['cabin_len']

        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.NOSE_FINENESS] = nose_height
        J[Aircraft.Fuselage.LENGTH, 'nose_height'] = LoverD_nose
        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = 1
        J[Aircraft.Fuselage.LENGTH, 'cabin_len'] = 1
        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.TAIL_FINENESS] = cabin_height
        J[Aircraft.Fuselage.LENGTH, 'cabin_height'] = LoverD_tail

        J[Aircraft.Fuselage.WETTED_AREA, 'cabin_height'] = fus_SA_scaler * (
            2.5 * (LoverD_nose * nose_height + cockpit_len)
            + 3.14 * cabin_len
            + 2.1 * LoverD_tail * cabin_height
            + cabin_height * 2.1 * LoverD_tail
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.NOSE_FINENESS] = (
            fus_SA_scaler * cabin_height * 2.5 * nose_height
        )
        J[Aircraft.Fuselage.WETTED_AREA, 'nose_height'] = (
            fus_SA_scaler * cabin_height * 2.5 * LoverD_nose
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = (
            fus_SA_scaler * cabin_height * 2.5
        )
        J[Aircraft.Fuselage.WETTED_AREA, 'cabin_len'] = fus_SA_scaler * 3.14 * cabin_height
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.TAIL_FINENESS] = (
            fus_SA_scaler * 2.1 * cabin_height**2
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.WETTED_AREA_SCALER] = cabin_height * (
            2.5 * (LoverD_nose * nose_height + cockpit_len)
            + 3.14 * cabin_len
            + 2.1 * LoverD_tail * cabin_height
        )

        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.NOSE_FINENESS] = nose_height
        J[Aircraft.TailBoom.LENGTH, 'nose_height'] = LoverD_nose
        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = 1
        J[Aircraft.TailBoom.LENGTH, 'cabin_len'] = 1
        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.TAIL_FINENESS] = cabin_height
        J[Aircraft.TailBoom.LENGTH, 'cabin_height'] = LoverD_tail

        J[Aircraft.Fuselage.CABIN_AREA, 'cabin_len'] = cabin_height
        J[Aircraft.Fuselage.CABIN_AREA, 'cabin_height'] = cabin_len + cockpit_len
        J[Aircraft.Fuselage.CABIN_AREA, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = cabin_height


class FuselageGroup(om.Group):
    """Group to pull together FuselageParameters and FuselageSize."""

    def setup(self):
        # outputs from parameters that are used in size but not outside of this group
        connected_input_outputs = ['cabin_height', 'cabin_len', 'nose_height']

        self.add_subsystem(
            'parameters',
            FuselageParameters(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'] + connected_input_outputs,
        )

        self.add_subsystem(
            'size',
            FuselageSize(),
            promotes_inputs=connected_input_outputs + ['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )


class BWBFuselageParameters1(om.ExplicitComponent):
    """Computation of average fuselage diameter, cabin height, cabin length and nose height for BWB."""

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Fuselage.AISLE_WIDTH, units='inch')
        add_aviary_option(self, Aircraft.Fuselage.NUM_AISLES)
        add_aviary_option(self, Aircraft.Fuselage.NUM_SEATS_ABREAST)
        add_aviary_option(self, Aircraft.Fuselage.SEAT_WIDTH, units='inch')
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.DELTA_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.NOSE_FINENESS, units='unitless')

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.HYDRAULIC_DIAMETER, units='ft', desc='DHYDRAL')
        self.add_output('cabin_height', units='ft', desc='HC: height of cabin')
        self.add_output('nose_height', units='ft', desc='HN: height of nose')
        self.add_output('nose_length', units='ft', desc='L_NOSE: length of nose')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.AVG_DIAMETER,
            [
                Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
            ],
        )
        self.declare_partials(
            Aircraft.Fuselage.HYDRAULIC_DIAMETER,
            [
                Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
                Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
            ],
        )
        self.declare_partials(
            'cabin_height',
            [
                Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
            ],
        )
        self.declare_partials(
            'nose_height',
            [
                Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
                Aircraft.Fuselage.DELTA_DIAMETER,
            ],
        )
        self.declare_partials(
            'nose_length',
            [
                Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
                Aircraft.Fuselage.DELTA_DIAMETER,
                Aircraft.Fuselage.NOSE_FINENESS,
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
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL]
        cabin_width = (seats_abreast * seat_width + num_aisle * aisle_width) / 12.0 + 1.0
        body_width = cabin_width + additional_width

        cabin_height = cabin_width * inputs[Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO]
        nose_height = cabin_height - inputs[Aircraft.Fuselage.DELTA_DIAMETER]
        nose_length = nose_height * inputs[Aircraft.Fuselage.NOSE_FINENESS]

        if PAX < 1:
            if verbosity >= Verbosity.BRIEF:
                print('Warning: you have not specified at least one passenger')

        fuselage_cross_area = np.pi * body_width * cabin_height / 4.0
        hydraulic_diameter = np.sqrt(4.0 * fuselage_cross_area / np.pi)

        outputs[Aircraft.Fuselage.AVG_DIAMETER] = body_width
        outputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER] = hydraulic_diameter
        outputs['cabin_height'] = cabin_height
        outputs['nose_height'] = nose_height
        outputs['nose_length'] = nose_length

    def compute_partials(self, inputs, J):
        options = self.options

        seats_abreast = options[Aircraft.Fuselage.NUM_SEATS_ABREAST]
        seat_width, _ = options[Aircraft.Fuselage.SEAT_WIDTH]
        num_aisle = options[Aircraft.Fuselage.NUM_AISLES]
        aisle_width, _ = options[Aircraft.Fuselage.AISLE_WIDTH]
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL]
        cabin_width = (seats_abreast * seat_width + num_aisle * aisle_width) / 12.0 + 1.0
        body_width = cabin_width + additional_width

        nose_height_to_length = inputs[Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO]
        delta_diameter = inputs[Aircraft.Fuselage.DELTA_DIAMETER]
        nose_fineness = inputs[Aircraft.Fuselage.NOSE_FINENESS]
        hydraulic_diameter = np.sqrt(body_width * cabin_width * nose_height_to_length)

        J[Aircraft.Fuselage.AVG_DIAMETER, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL] = 1.0

        J[Aircraft.Fuselage.HYDRAULIC_DIAMETER, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL] = (
            0.5 * cabin_width * nose_height_to_length / hydraulic_diameter
        )
        J[Aircraft.Fuselage.HYDRAULIC_DIAMETER, Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO] = (
            0.5 * body_width * cabin_width / hydraulic_diameter
        )

        J['cabin_height', Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO] = cabin_width

        J['nose_height', Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO] = cabin_width
        J['nose_height', Aircraft.Fuselage.DELTA_DIAMETER] = -1.0

        J['nose_length', Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO] = cabin_width * nose_fineness
        J['nose_length', Aircraft.Fuselage.DELTA_DIAMETER] = -nose_fineness
        J['nose_length', Aircraft.Fuselage.NOSE_FINENESS] = (
            cabin_width * nose_height_to_length - delta_diameter
        )


class BWBCabinLayout(om.ExplicitComponent):
    """layout of passenger cabin for BWB."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.SEAT_WIDTH, units='inch', desc='INGASP.WS')
        add_aviary_option(self, Aircraft.Fuselage.NUM_AISLES, units='unitless', desc='INGASP.AS')
        add_aviary_option(self, Aircraft.Fuselage.AISLE_WIDTH, units='inch', desc='INGASP.WAS')
        add_aviary_option(self, Aircraft.Fuselage.SEAT_PITCH, units='inch', desc='INGASP.PS')
        add_aviary_option(
            self, Aircraft.CrewPayload.Design.NUM_PASSENGERS, units='unitless', desc='INGASP.PAX'
        )
        add_aviary_option(
            self,
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
            units='unitless',
            desc='equiv INGASP.PCT_FC',
        )
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, units='deg')
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, units='ft')
        self.add_input('nose_length', units='ft', desc='L_NOSE: nose length')

        self.add_output(
            'fuselage_station_aft',
            units='ft',
            desc='EL_AFT: fuselage station of aft pressure bulkhead',
        )

        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        options = self.options
        verbosity = options[Settings.VERBOSITY]
        rad2deg = 180.0 / np.pi

        # Hard code variables in GASP:
        FC_lav_galley_length = 8.0  # EL_FLGC: length of first class lav, galley & closet, ft
        FC_seat_width = 28.0  # WS_FC: first class seat width, inch
        FC_seat_pitch = 36.0  # PS_FC: first class seat pitch, inch
        FC_num_aisles = 2  # AS_FC: num of aisles in first class
        FC_aisle_width = 24.0  # WAS_FC: First class aisle width, inch
        length_FC_to_TC = 5.0  # Length of first class/tourist class aisle, ft
        TC_num_pax_per_lav = 78  # NLAVTC: tourist class passengers per lav
        TC_lav_width = 42.0  # WIDLAV: Lav width, inches
        TC_galley_area_per_pax = 0.15  # AGAL_TC: tourist class galley area per passenger, ft**2
        # If there is no first class cabin, please set NUM_FIRST_CLASS = 0.

        TC_seat_pitch, _ = options[Aircraft.Fuselage.SEAT_PITCH]
        seat_width, _ = options[Aircraft.Fuselage.SEAT_WIDTH]
        if seat_width <= 0.0:
            raise ValueError('fuselage seat width must be positive.')
        aisle_width, _ = options[Aircraft.Fuselage.AISLE_WIDTH]
        num_aisles = options[Aircraft.Fuselage.NUM_AISLES]

        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER][0]
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL][0]
        cabin_width = body_width - additional_width

        sweep_FB = inputs[Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP][0]
        pax = options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        pax_FC = options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        if pax_FC <= 0:
            if verbosity > Verbosity.BRIEF:
                print('Warning: No first class passengers or cabins are included.')
        if pax_FC > pax:
            raise ValueError(
                'Number of first class passengers must not exceed the total number of passengers.'
            )
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

            Idx_row_FC = -1
            while sum_num_seats_FC < pax_FC:
                Idx_row_FC = Idx_row_FC + 1
                len = EL_FC_ptr + FC_seat_pitch / 12.0
                length_FC_by_row.append(len)
                wid = 2.0 * length_FC_by_row[Idx_row_FC] / np.tan(sweep_FB / rad2deg)
                wid = np.minimum(wid, cabin_width)
                width_FC_by_row.append(wid)
                wid_aisle = FC_num_aisles * FC_aisle_width / 12.0
                num = int((width_FC_by_row[Idx_row_FC] - wid_aisle) / FC_seat_width * 12.0)
                num_seats_FC_by_row.append(num)
                prev_sum_num_seats_FC = sum_num_seats_FC
                sum_num_seats_FC = sum_num_seats_FC + num_seats_FC_by_row[Idx_row_FC]
                EL_FC_ptr = length_FC_by_row[Idx_row_FC]

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
        if pax_FC > 0:
            EL_TC_ptr = EL_TC_ptr + length_FC_to_TC - TC_seat_pitch / 12.0
        else:
            EL_TC_ptr = EL_TC_ptr - TC_seat_pitch / 12.0
        length_TC_by_row = []  # length in tourist class, ft
        width_TC_by_row = []  # width in tourist class, ft
        num_seats_TC_by_row = []  # num of seats in tourist class
        sum_num_seats_TC = 0
        Idx_row_TC = -1
        while sum_num_seats_TC < pax_TC:
            Idx_row_TC = Idx_row_TC + 1
            len = EL_TC_ptr + TC_seat_pitch / 12.0
            length_TC_by_row.append(len)
            wid = 2.0 * length_TC_by_row[Idx_row_TC] / np.tan(sweep_FB / rad2deg)
            wid = np.minimum(wid, cabin_width)
            width_TC_by_row.append(wid)
            width_aisle = num_aisles * aisle_width / 12.0
            num = int((width_TC_by_row[Idx_row_TC] - width_aisle) / (seat_width / 12.0))
            num_seats_TC_by_row.append(num)
            prev_num_seats_TC = sum_num_seats_TC
            sum_num_seats_TC = sum_num_seats_TC + num_seats_TC_by_row[Idx_row_TC]
            EL_TC_ptr = length_TC_by_row[Idx_row_TC]

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
            EL_AFT = length_TC_by_row[Idx_row_TC] + TC_seat_pitch / 12.0 + TC_lav_width / 12.0

        outputs['fuselage_station_aft'] = EL_AFT


class BWBFuselageParameters2(om.ExplicitComponent):
    """
    Computation of average fuselage diameter, cabin floor area, cabin length and
    fuselage planform area.
    """

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, units='deg')
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.TAIL_FINENESS, units='unitless', desc='ELODT')
        self.add_input(
            'fuselage_station_aft',
            val=0.0,
            units='ft',
            desc='EL_AFT: fuselage station of aft pressure bulkhead',
        )
        self.add_input('nose_length', val=0.0, units='ft', desc='L_NOSE: length of nose')
        self.add_input('cabin_height', val=0.0, units='ft', desc='HC: height of cabin')

        add_aviary_output(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2', desc='ACABIN')
        add_aviary_output(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2', desc='SPF_BODY')
        self.add_output('cabin_len', units='ft', desc='LC: length of cabin')
        self.add_output('forebody_len', units='ft', desc='L_FBODY: length of forebody')
        self.add_output('aftbody_len', units='ft', desc='L_ABODY: length of aftbody')
        self.add_output('nose_area', units='ft**2', desc='A_NOSE: nose area')

    def setup_partials(self):
        self.declare_partials(
            'cabin_len',
            [
                'fuselage_station_aft',
                'nose_length',
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
            ],
        )

        self.declare_partials(
            'forebody_len',
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
            ],
        )

        self.declare_partials(
            'aftbody_len',
            [
                Aircraft.Fuselage.TAIL_FINENESS,
                'cabin_height',
            ],
        )

        self.declare_partials(
            'nose_area',
            [
                'nose_length',
                Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.CABIN_AREA,
            [
                Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
                'fuselage_station_aft',
                'nose_length',
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.PLANFORM_AREA,
            [
                Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
                'fuselage_station_aft',
                'nose_length',
                'cabin_height',
                Aircraft.Fuselage.TAIL_FINENESS,
            ],
        )

    def compute(self, inputs, outputs):
        rad2deg = 180.0 / np.pi

        PASSENGER_LEADING_EDGE_SWEEP = inputs[Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP]
        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL]
        pilot_comp_len = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH]
        len_to_diam_tail_cone = inputs[Aircraft.Fuselage.TAIL_FINENESS]
        nose_len = inputs['nose_length']
        cabin_height = inputs['cabin_height']
        fuselage_station_aft = inputs['fuselage_station_aft']

        cabin_width = body_width - additional_width
        forebody_len = 0.5 * cabin_width * np.tan(PASSENGER_LEADING_EDGE_SWEEP / rad2deg)
        nose_width = 2.0 * nose_len / np.tan(PASSENGER_LEADING_EDGE_SWEEP / rad2deg)
        aftbody_len = len_to_diam_tail_cone * cabin_height
        cabin_len = fuselage_station_aft - nose_len - pilot_comp_len

        area_nose_planform = 0.5 * nose_len * nose_width
        area_aftbody_planform = aftbody_len * (cabin_width + body_width) / 2.0
        area_forebody = (forebody_len - nose_len) * (body_width + nose_width) / 2.0
        area_aftbody = body_width * (fuselage_station_aft - forebody_len)
        # Cabin Floor Area  ("Home Plate" - without the nose)
        area_cabin = area_forebody + area_aftbody

        # Planform Area of Body
        area_body_planform = area_nose_planform + area_cabin + area_aftbody_planform

        outputs['cabin_len'] = cabin_len
        outputs['forebody_len'] = forebody_len
        outputs['aftbody_len'] = aftbody_len
        outputs['nose_area'] = area_nose_planform
        outputs[Aircraft.Fuselage.CABIN_AREA] = area_cabin
        outputs[Aircraft.Fuselage.PLANFORM_AREA] = area_body_planform

    def compute_partials(self, inputs, J):
        options = self.options
        verbosity = options[Settings.VERBOSITY]
        rad2deg = 180.0 / np.pi

        PASSENGER_LEADING_EDGE_SWEEP = inputs[Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP]
        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL]
        len_to_diam_tail_cone = inputs[Aircraft.Fuselage.TAIL_FINENESS]
        nose_len = inputs['nose_length']
        cabin_height = inputs['cabin_height']
        fuselage_station_aft = inputs['fuselage_station_aft']

        if PASSENGER_LEADING_EDGE_SWEEP <= 0.0 or PASSENGER_LEADING_EDGE_SWEEP >= 90.0:
            if verbosity > Verbosity.BRIEF:
                print('Forebody sweep angle must be between 0 and 90 degrees.')
        fb_sweep_rad = PASSENGER_LEADING_EDGE_SWEEP / rad2deg
        fb_tan = np.tan(fb_sweep_rad)
        fb_cos = np.cos(fb_sweep_rad)
        fb_dtan = 1.0 / rad2deg / (fb_cos * fb_cos)

        cabin_width = body_width - additional_width

        J['cabin_len', 'fuselage_station_aft'] = 1.0
        J['cabin_len', 'nose_length'] = -1.0
        J['cabin_len', Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = -1.0

        J['forebody_len', Aircraft.Fuselage.AVG_DIAMETER] = 0.5 * fb_tan
        J['forebody_len', Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP] = 0.5 * cabin_width * fb_dtan
        J['forebody_len', Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL] = -0.5 * fb_tan

        J['aftbody_len', Aircraft.Fuselage.TAIL_FINENESS] = cabin_height
        J['aftbody_len', 'cabin_height'] = len_to_diam_tail_cone

        d_forebody_area_d_bd_width = 0.5 * fb_tan * (0.5 * body_width + nose_len / fb_tan) + (
            0.5 * cabin_width * fb_tan - nose_len
        ) * (0.5)
        d_forebody_area_d_ns_length = (
            -(0.5 * body_width + nose_len / fb_tan)
            + (0.5 * cabin_width * fb_tan - nose_len) / fb_tan
        )
        d_forebody_area_d_add_length = -0.5 * fb_tan * (0.5 * body_width + nose_len / fb_tan)
        d_forebody_area_d_sweep = (0.5 * cabin_width * fb_dtan) * (
            0.5 * body_width + nose_len / fb_tan
        ) - (0.5 * cabin_width * fb_tan - nose_len) * nose_len / fb_tan**2 * fb_dtan
        d_aftbody_area_d_bd_width = (
            fuselage_station_aft - 0.5 * cabin_width * fb_tan - 0.5 * body_width * fb_tan
        )
        d_aftbody_area_d_fuselage_aft = body_width
        d_aftbody_area_d_additional = 0.5 * body_width * fb_tan
        d_aftbody_area_d_sweep = -0.5 * body_width * cabin_width * fb_dtan

        #
        # cabin_area = forebody + aftbody
        #
        d_cabin_area_d_body_len = d_forebody_area_d_bd_width + d_aftbody_area_d_bd_width
        J[Aircraft.Fuselage.CABIN_AREA, Aircraft.Fuselage.AVG_DIAMETER] = d_cabin_area_d_body_len

        d_cabin_area_d_nose_len = d_forebody_area_d_ns_length
        J[Aircraft.Fuselage.CABIN_AREA, 'nose_length'] = d_cabin_area_d_nose_len

        d_cabin_area_d_fuselage_aft = d_aftbody_area_d_fuselage_aft
        J[Aircraft.Fuselage.CABIN_AREA, 'fuselage_station_aft'] = d_cabin_area_d_fuselage_aft

        d_cabin_area_d_additional = d_forebody_area_d_add_length + d_aftbody_area_d_additional
        J[Aircraft.Fuselage.CABIN_AREA, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL] = (
            d_cabin_area_d_additional
        )

        d_cabin_area_d_sweep = d_forebody_area_d_sweep + d_aftbody_area_d_sweep
        J[Aircraft.Fuselage.CABIN_AREA, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP] = (
            d_cabin_area_d_sweep
        )

        d_nose_pf_area_d_nose_len = 2 * nose_len / fb_tan
        J['nose_area', 'nose_length'] = d_nose_pf_area_d_nose_len
        d_nose_pf_area_d_sweep = -nose_len * nose_len / fb_tan / fb_tan * fb_dtan
        J['nose_area', Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP] = d_nose_pf_area_d_sweep

        d_tail_pf_area_d_tail_fineness = cabin_height * (body_width - 0.5 * additional_width)
        d_tail_pf_area_d_both_width = len_to_diam_tail_cone * cabin_height
        d_tail_pf_area_d_additional = -0.5 * len_to_diam_tail_cone * cabin_height
        d_tail_pf_area_d_cabin_height = len_to_diam_tail_cone * (
            body_width - 0.5 * additional_width
        )

        #
        # fuselage_planform_area = nose_area + cabin_area + tail_area
        #
        J[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP] = (
            d_nose_pf_area_d_sweep + d_cabin_area_d_sweep
        )
        J[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.AVG_DIAMETER] = (
            d_cabin_area_d_body_len + d_tail_pf_area_d_both_width
        )
        J[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL] = (
            d_cabin_area_d_additional + d_tail_pf_area_d_additional
        )
        J[Aircraft.Fuselage.PLANFORM_AREA, 'fuselage_station_aft'] = d_cabin_area_d_fuselage_aft
        J[Aircraft.Fuselage.PLANFORM_AREA, 'nose_length'] = (
            d_nose_pf_area_d_nose_len + d_cabin_area_d_nose_len
        )
        J[Aircraft.Fuselage.PLANFORM_AREA, 'cabin_height'] = d_tail_pf_area_d_cabin_height
        J[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.TAIL_FINENESS] = (
            d_tail_pf_area_d_tail_fineness
        )


class BWBFuselageSize(om.ExplicitComponent):
    """Computation of fuselage length and wetted area."""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, units='ft', desc='ELPC')
        self.add_input('cabin_height', val=0.0, units='ft', desc='HC: height of cabin')
        self.add_input('forebody_len', val=0.0, units='ft', desc='L_FBODY: forebody length')
        self.add_input(
            'fuselage_station_aft',
            val=0.0,
            units='ft',
            desc='EL_AFT: fuselage station of aft pressure bulkhead',
        )
        self.add_input('nose_area', val=0.0, units='ft**2', desc='A_NOSE: nose area')
        self.add_input('aftbody_len', val=0.0, units='ft', desc='L_ABODY: aftbody length')
        self.add_input('nose_length', val=0.0, units='ft', desc='L_NOSE: nose length')
        self.add_input('cabin_len', val=0.0, units='ft', desc='LC: cabin length')

        add_aviary_output(self, Aircraft.Fuselage.LENGTH, units='ft', desc='ELF')
        add_aviary_output(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2', desc='SF')
        # tail boom support is not implemented in Aviary
        add_aviary_output(self, Aircraft.TailBoom.LENGTH, units='ft', desc='ELFFC')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.LENGTH,
            [
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                'nose_length',
                'aftbody_len',
                'cabin_len',
            ],
        )

        self.declare_partials(
            Aircraft.TailBoom.LENGTH,
            [
                Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
                'nose_length',
                'aftbody_len',
                'cabin_len',
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.WETTED_AREA,
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
                Aircraft.Fuselage.WETTED_AREA_SCALER,
                'cabin_height',
                'forebody_len',
                'nose_area',
                'aftbody_len',
                'fuselage_station_aft',
            ],
        )

    def compute(self, inputs, outputs):
        cockpit_len = inputs[Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH]
        cabin_len = inputs['cabin_len']
        nose_length = inputs['nose_length']
        aftbody_len = inputs['aftbody_len']

        fuselage_length = nose_length + cockpit_len + cabin_len + aftbody_len
        outputs[Aircraft.Fuselage.LENGTH] = fuselage_length

        cabin_len_tailboom = fuselage_length
        outputs[Aircraft.TailBoom.LENGTH] = cabin_len_tailboom

        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL]
        fus_SA_scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
        fuselage_station_aft = inputs['fuselage_station_aft']
        cabin_height = inputs['cabin_height']
        forebody_len = inputs['forebody_len']
        nose_area = inputs['nose_area']

        diag_a_c = np.sqrt(aftbody_len**2 + (cabin_height / 2.0) ** 2)
        diag_a_a = np.sqrt(aftbody_len**2 + (additional_width / 2.0) ** 2)
        diag_f_b = np.sqrt(forebody_len**2 + (body_width / 2.0) ** 2)
        diag_f_c = np.sqrt(forebody_len**2 + (cabin_height / 2.0) ** 2)

        # use a simple prismatic surface area estimation
        forebody_surface_area = cabin_height * diag_f_b + body_width * diag_f_c

        # just top and bottom surface area for cabin
        cabin_surface_area = 2.0 * (body_width * (fuselage_station_aft - forebody_len) + nose_area)

        body_width_pass = body_width - additional_width
        aftbody_surface_area = (body_width + body_width_pass) * diag_a_c + cabin_height * diag_a_a

        fuselage_wetted_area = forebody_surface_area + cabin_surface_area + aftbody_surface_area
        # Adjust Fuselage Wetted Area by WETTED_AREA_SCALER
        fuselage_wetted_area = fuselage_wetted_area * fus_SA_scaler

        outputs[Aircraft.Fuselage.WETTED_AREA] = fuselage_wetted_area

    def compute_partials(self, inputs, J):
        J[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = 1
        J[Aircraft.Fuselage.LENGTH, 'nose_length'] = 1
        J[Aircraft.Fuselage.LENGTH, 'aftbody_len'] = 1
        J[Aircraft.Fuselage.LENGTH, 'cabin_len'] = 1

        J[Aircraft.TailBoom.LENGTH, Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH] = 1
        J[Aircraft.TailBoom.LENGTH, 'nose_length'] = 1
        J[Aircraft.TailBoom.LENGTH, 'aftbody_len'] = 1
        J[Aircraft.TailBoom.LENGTH, 'cabin_len'] = 1

        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        additional_width = inputs[Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL]
        fus_SA_scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
        fuselage_station_aft = inputs['fuselage_station_aft']
        cabin_height = inputs['cabin_height']
        forebody_len = inputs['forebody_len']
        nose_area = inputs['nose_area']
        aftbody_len = inputs['aftbody_len']
        body_width_pass = body_width - additional_width

        diag_a_c = np.sqrt(aftbody_len**2 + (cabin_height / 2.0) ** 2)
        diag_a_a = np.sqrt(aftbody_len**2 + (additional_width / 2.0) ** 2)
        diag_f_b = np.sqrt(forebody_len**2 + (body_width / 2.0) ** 2)
        diag_f_c = np.sqrt(forebody_len**2 + (cabin_height / 2.0) ** 2)

        forebody_surface_area = cabin_height * diag_f_b + body_width * diag_f_c
        cabin_surface_area = 2.0 * (body_width * (fuselage_station_aft - forebody_len) + nose_area)
        aftbody_surface_area = (body_width + body_width_pass) * diag_a_c + cabin_height * diag_a_a

        d_fb_area_d_cabin_height = diag_f_b + body_width * cabin_height * 0.25 / diag_f_c
        d_fb_area_d_fb_len = (
            cabin_height * forebody_len / diag_f_b + body_width * forebody_len / diag_f_c
        )
        d_fb_area_d_wody_width = cabin_height * body_width * 0.25 / diag_f_b + diag_f_c

        d_cabin_area_d_body_width = 2.0 * (fuselage_station_aft - forebody_len)
        d_cabin_area_d_station_aft = 2.0 * body_width
        d_cabin_area_d_fb_len = -2.0 * body_width
        d_cabin_area_d_nose_area = 2.0

        d_aft_d_body_width = 2.0 * diag_a_c
        d_aft_d_additional_width = -diag_a_c + cabin_height * additional_width * 0.25 * diag_a_c
        d_aft_d_aft_len = (
            2 * body_width - additional_width
        ) * aftbody_len / diag_a_c + cabin_height * aftbody_len / diag_a_a
        d_aft_d_cabin_height = (
            2 * body_width - additional_width
        ) * cabin_height * 0.25 / diag_a_c + diag_a_a

        #
        # fuselage_wetted_area:
        #     (forebody_surface_area + cabin_surface_area + aftbody_surface_area)
        #     * fus_SA_scaler
        #
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.WETTED_AREA_SCALER] = (
            forebody_surface_area + cabin_surface_area + aftbody_surface_area
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL] = (
            d_aft_d_additional_width * fus_SA_scaler
        )
        J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.AVG_DIAMETER] = fus_SA_scaler * (
            d_fb_area_d_wody_width + d_cabin_area_d_body_width + d_aft_d_body_width
        )

        J[Aircraft.Fuselage.WETTED_AREA, 'cabin_height'] = (
            d_fb_area_d_cabin_height + d_aft_d_cabin_height
        ) * fus_SA_scaler
        J[Aircraft.Fuselage.WETTED_AREA, 'forebody_len'] = (
            d_fb_area_d_fb_len + d_cabin_area_d_fb_len
        ) * fus_SA_scaler
        J[Aircraft.Fuselage.WETTED_AREA, 'fuselage_station_aft'] = (
            d_cabin_area_d_station_aft * fus_SA_scaler
        )
        J[Aircraft.Fuselage.WETTED_AREA, 'aftbody_len'] = d_aft_d_aft_len * fus_SA_scaler
        J[Aircraft.Fuselage.WETTED_AREA, 'nose_area'] = d_cabin_area_d_nose_area * fus_SA_scaler


class BWBFuselageGroup(om.Group):
    """
    Group to pull together BWBFuselageParameters1, BWBCabinLayout, BWBFuselageParameters2
    and BWBFuselageSize.
    """

    def setup(self):
        self.add_subsystem(
            'parameters1',
            BWBFuselageParameters1(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'] + ['nose_length', 'cabin_height'],
        )

        self.add_subsystem(
            'layout',
            BWBCabinLayout(),
            promotes_inputs=['aircraft:*'] + ['nose_length'],
            promotes_outputs=['fuselage_station_aft'],
        )

        self.add_subsystem(
            'parameters2',
            BWBFuselageParameters2(),
            promotes_inputs=['aircraft:*']
            + ['nose_length', 'cabin_height', 'fuselage_station_aft'],
            promotes_outputs=['aircraft:*']
            + ['forebody_len', 'nose_area', 'aftbody_len', 'cabin_len'],
        )

        self.add_subsystem(
            'size',
            BWBFuselageSize(),
            promotes_inputs=['aircraft:*']
            + [
                'nose_length',
                'cabin_height',
                'fuselage_station_aft',
                'forebody_len',
                'nose_area',
                'aftbody_len',
                'cabin_len',
            ],
            promotes_outputs=['aircraft:*'],
        )
