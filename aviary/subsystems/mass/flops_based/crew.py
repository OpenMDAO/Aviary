'''
Define utilities to calculate the estimated mass of the crew (both flight and
non-flight) as well as their baggage.
'''
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class NonFlightCrewMass(om.ExplicitComponent):
    '''
    Calculate the estimated mass for the non-flight and their baggage.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self, Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER, 1.)

        add_aviary_output(self, Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, 0)

    def setup_partials(self):
        self.declare_partials(
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER
        )

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        aviary_options: AviaryValues = self.options['aviary_options']
        flight_attendants_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS)
        galley_crew_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW)

        mass_per_flight_attendant = self._mass_per_flight_attendant
        mass_per_galley_crew = self._mass_per_galley_crew

        mass_scaler = \
            inputs[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER]

        outputs[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS] = (
            flight_attendants_count * mass_per_flight_attendant
            + galley_crew_count * mass_per_galley_crew
        ) * mass_scaler

    def compute_partials(self, inputs, J, discrete_inputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        flight_attendants_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS)
        galley_crew_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW)

        mass_per_flight_attendant = self._mass_per_flight_attendant
        mass_per_galley_crew = self._mass_per_galley_crew

        J[
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER
        ] = (
            flight_attendants_count * mass_per_flight_attendant
            + galley_crew_count * mass_per_galley_crew
        )

    _mass_per_flight_attendant = 155.  # lbm
    _mass_per_galley_crew = 200.  # lbm


class FlightCrewMass(om.ExplicitComponent):
    '''
    Calculate the estimated mass for the flight crew and their baggage.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self, Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER, 1.)

        add_aviary_output(self, Aircraft.CrewPayload.FLIGHT_CREW_MASS, 0.)

    def setup_partials(self):
        self.declare_partials(
            Aircraft.CrewPayload.FLIGHT_CREW_MASS,
            Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER
        )

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        aviary_options: AviaryValues = self.options['aviary_options']
        flight_crew_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)

        mass_per_flight_crew = self._mass_per_flight_crew(inputs)

        mass_scaler = inputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER]

        outputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS] = \
            flight_crew_count * mass_per_flight_crew * mass_scaler

    def compute_partials(self, inputs, J, discrete_inputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        flight_crew_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)

        mass_per_flight_crew = self._mass_per_flight_crew(inputs)

        J[
            Aircraft.CrewPayload.FLIGHT_CREW_MASS,
            Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER
        ] = flight_crew_count * mass_per_flight_crew

    def _mass_per_flight_crew(self, inputs):
        '''
        Return the mass, in pounds, of one member of the flight crew and
        their baggage.
        '''
        aviary_options: AviaryValues = self.options['aviary_options']
        mass_per_flight_crew = 225.0  # lbm

        # account for machine precision error
        if 0.9 <= aviary_options.get_val(Aircraft.LandingGear.CARRIER_BASED):
            mass_per_flight_crew -= 35.0  # lbm

        return mass_per_flight_crew
