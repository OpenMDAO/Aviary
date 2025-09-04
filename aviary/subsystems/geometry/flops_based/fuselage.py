"""Contains any preliminary calculations on the fuselage."""

import numpy as np
import openmdao.api as om

from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission, Settings


class FuselagePrelim(om.ExplicitComponent):
    """
    Calculate fuselage average diameter and planform area defined by:
    Aircraft.Fuselage.AVG_DIAMETER = 0.5 * (max_height + max_width)
    Aircraft.Fuselage.PLANFORM_AREA = length * max_width.
    """

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2')

    def setup_partials(self):
        self.declare_partials(
            of=[Aircraft.Fuselage.AVG_DIAMETER],
            wrt=[Aircraft.Fuselage.MAX_HEIGHT, Aircraft.Fuselage.MAX_WIDTH],
            val=0.5,
        )
        self.declare_partials(
            of=[Aircraft.Fuselage.PLANFORM_AREA],
            wrt=[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.MAX_WIDTH],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]
        if length <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Fuselage.LENGTH must be positive.')

        avg_diameter = 0.5 * (max_height + max_width)
        outputs[Aircraft.Fuselage.AVG_DIAMETER] = avg_diameter

        outputs[Aircraft.Fuselage.PLANFORM_AREA] = length * max_width

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]

        partials[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.LENGTH] = max_width

        partials[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.MAX_WIDTH] = length


class SimpleCabinLayout(om.ExplicitComponent):
    """Given fuselage length, compute passenger compartment length."""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, units='ft')

    def setup_partials(self):
        self.declare_partials(
            of=[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH],
            wrt=[Aircraft.Fuselage.LENGTH],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        length = inputs[Aircraft.Fuselage.LENGTH]
        max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        if length <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Fuselage.LENGTH must be positive to use simple cabin layout.')
        if max_height <= 0.0 or max_width <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print(
                    'Aircraft.Fuselage.MAX_HEIGHT & Aircraft.Fuselage.MAX_WIDTH must be positive.'
                )

        pax_compart_length = 0.6085 * length * (np.arctan(length / 59.0)) ** 1.1
        if pax_compart_length > 190.0:
            if verbosity > Verbosity.BRIEF:
                print(
                    'Passenger compartiment lenght is longer than recommended maximum length. '
                    'Suggest use detailed laylout algorithm.'
                )
        outputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH] = pax_compart_length

    def compute_partials(self, inputs, partials):
        length = inputs[Aircraft.Fuselage.LENGTH]
        atan = np.arctan(length / 59.0)
        deriv = 0.6085 * (
            (atan) ** 1.1 + 1.1 * length * atan**0.1 / (1 + (length / 59.0) ** 2) / 59.0
        )
        partials[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, Aircraft.Fuselage.LENGTH] = deriv


class DetailedCabinLayout(om.ExplicitComponent):
    """Compute fuselage length and passenger compartment length."""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)
        add_aviary_option(self, Aircraft.Fuselage.NUM_FUSELAGES)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST)
        add_aviary_option(self, Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST)
        add_aviary_option(self, Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST)
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Mission.Design.RANGE, units='NM')

        add_aviary_output(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')

        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        num_first_class_pax = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        num_tourist_class_pax = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]
        fuselage_multiplier = 1.0
        num_fuselage = self.options[Aircraft.Fuselage.NUM_FUSELAGES]
        if num_fuselage > 1:
            if verbosity > Verbosity.BRIEF:
                print('Multiple fuselage configuration is not implemented yet.')

        num_seat_abreast_first = self.options[Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST]
        num_seat_abreast_tourist = self.options[
            Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST
        ]
        # The 200 was derived from B757 - the largest single aisle western desig.
        if num_tourist_class_pax > 200:
            if num_seat_abreast_tourist <= 0:
                num_seat_abreast_tourist = 8
            if num_seat_abreast_first > 0 and num_seat_abreast_first <= 0:
                num_seat_abreast_first = num_seat_abreast_tourist - 2

        if num_seat_abreast_first <= 0 and num_first_class_pax > 0:
            num_seat_abreast_first = 4
        if num_seat_abreast_tourist <= 0 and num_tourist_class_pax > 0:
            num_seat_abreast_tourist = 6

        # Though these are not user definable, the values here are typical for most transport
        aisle_width_first_class = 20.0  # inch
        aisle_width_tourist_class = 18.0  # inch

        # If there are less than 60 passengers on board, then the aisle should be slightly narrow.
        # Also, if the number of passengers abreast was not specified, then set it to 5 as 6 is too much
        # for a typical short range transport.
        if num_tourist_class_pax < 60:
            if num_seat_abreast_tourist <= 0:
                num_seat_abreast_tourist = 5
            aisle_width_tourist_class = 15.0

        if num_seat_abreast_tourist > 6:
            num_aisles = 2
        else:
            num_aisles = 1

        # Even though the widebody gives you more room, the aisles in it are a little narrower.
        # That is what is set here. It is assumed that if the number of abreast is greater than
        # 4 or 6 as shown below that we are working with a widebody aircraft.
        if num_seat_abreast_first > 4:
            aisle_width_first_class = 18.0
        if num_seat_abreast_tourist > 6:
            aisle_width_tourist_class = 15.0

        seat_pitch_first = self.options[Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST][0]
        if seat_pitch_first <= 0 and num_first_class_pax > 0:
            seat_pitch_first = 38.0  # inch
        seat_pitch_tourist = self.options[Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST][0]
        if seat_pitch_tourist <= 0 and num_tourist_class_pax > 0:
            seat_pitch_tourist = 34.0  # inch

        # set maximum number of galleys based on statistics (this block is not from FLOPS)
        num_pax = num_first_class_pax + num_tourist_class_pax
        if num_pax < 80:
            max_galleys = 1
            max_lav = 1
            max_closets = 1
        elif num_pax < 150:
            max_galleys = 2
            max_lav = 2
            max_closets = 2
        elif num_pax < 250:
            max_galleys = 3
            max_lav = 4
            max_closets = 3
        elif num_pax < 320:
            max_galleys = 4
            max_lav = 6
            max_closets = 4
        elif num_pax < 350:
            max_galleys = 5
            max_lav = 8
            max_closets = 5
        elif num_pax < 410:
            max_galleys = 6
            max_lav = 12
            max_closets = 6
        elif num_pax < 450:
            max_galleys = 7
            max_lav = 13
            max_closets = 7
        elif num_pax < 500:
            max_galleys = 8
            max_lav = 14
            max_closets = 8
        elif num_pax < 550:
            max_galleys = 9
            max_lav = 15
            max_closets = 9
        elif num_pax < 600:
            max_galleys = 10
            max_lav = 16
            max_closets = 10
        else:
            max_galleys = 10
            max_lav = 16
            max_closets = 10
        # The above settings are necessary because FLOPS didn't cover all the scenarios.
        # They will be over written if FLOPS covered a particular scenario as we see below.

        # Set constraints on the maximum number of galleys and other items so that we don't have
        # a flying kitchen or closet or whatever.
        # Note: Some of these may need relaxing for larger aircraft.
        if num_first_class_pax == 0:
            design_range = inputs[Mission.Design.RANGE]
            if design_range < 1250.0:
                max_lav = 2
            else:
                max_lav = 3
            if num_tourist_class_pax < 180:
                fuselage_multiplier = 0.91
            max_galleys = 2
            max_closets = 2

        if num_first_class_pax == 0 and num_tourist_class_pax < 110:
            max_lav = 1
            max_galleys = 1
            max_closets = 1

        if num_tourist_class_pax > 320:
            max_lav = 4
            max_galleys = 4
            max_closets = 4
        elif num_tourist_class_pax > 600:
            max_lav = 8
            max_galleys = 8
            max_closets = 8

        if num_first_class_pax > 0 and num_seat_abreast_tourist < 8:
            fuselage_multiplier = 0.95

        # Calculate the number of galleys, lavatories and closets
        num_galleys = int(1 + ((num_first_class_pax + num_tourist_class_pax) / 100))
        if num_galleys > max_galleys:
            num_galleys = max_galleys
        num_lavas = int(1 + (num_tourist_class_pax / 60)) + int(1 + (num_first_class_pax / 100))
        if num_lavas > max_lav:
            num_lavas = max_lav
        num_closets = int(1 + (num_first_class_pax / 30)) + int(1 + (num_tourist_class_pax / 60))
        if num_closets > max_closets:
            num_closets = max_closets

        # Calculate the passenger compartment length

        num_engines = self.options[Aircraft.Engine.NUM_ENGINES][0]
        eng_flag = num_engines - 2 * int(num_engines / 2)  # a center mounted engine if 1.
        first_class_len = num_first_class_pax * seat_pitch_first / num_seat_abreast_first
        tourist_class_len = num_tourist_class_pax * seat_pitch_tourist / num_seat_abreast_tourist
        pax_compart_length = (
            first_class_len
            + (num_galleys + num_lavas) * 36.0
            + tourist_class_len
            + num_closets * 12.0
        ) / 12.0

        # Correct for doors that may be in the way
        num_doors = 1 + int((pax_compart_length / 50.0) * num_seat_abreast_tourist / 6.0)

        # Final passenger compartment length
        pax_compart_length = (pax_compart_length + num_doors * 2.96) * fuselage_multiplier
        fuselage_length = pax_compart_length + 25.0 * eng_flag + 40.0
        outputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH] = pax_compart_length
        outputs[Aircraft.Fuselage.LENGTH] = fuselage_length

        # Calculate the number of rows of each class of passenger (not needed)
        if num_first_class_pax > 0:
            num_rows_first = int(np.ceil(num_first_class_pax / num_seat_abreast_first))
        if num_tourist_class_pax > 0:
            num_rows_tourist = int(np.ceil(num_tourist_class_pax / num_seat_abreast_tourist))

        # Calculate the fuselage width of the passenger seats
        width_first_class = (
            num_aisles * aisle_width_first_class + num_seat_abreast_first * 20.0
        ) / 12.0
        width_tourist_class = (
            num_aisles * aisle_width_tourist_class + num_seat_abreast_tourist * 25.0
        ) / 12.0
        width_fuselage = np.maximum(width_first_class, width_tourist_class) * 1.06
        outputs[Aircraft.Fuselage.MAX_WIDTH] = width_fuselage
        outputs[Aircraft.Fuselage.MAX_HEIGHT] = width_fuselage + 0.9
