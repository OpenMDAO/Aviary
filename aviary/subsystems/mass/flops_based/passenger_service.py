'''
Define utilities to calculate the estimated mass of passenger service
equipment.
'''
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class PassengerServiceMass(om.ExplicitComponent):
    '''
    Define the default component to calculate the estimated mass of passenger
    service equipment. The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
            val=1.,
        )

        add_aviary_input(
            self,
            Mission.Design.RANGE,
            val=0.0,
        )

        add_aviary_output(
            self,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            val=0.0,
        )

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        first_class_count = aviary_options.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS)

        business_class_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS)

        tourist_class_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS)
        design_range = inputs[Mission.Design.RANGE]
        max_mach = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        passenger_service_mass_scaler = \
            inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER]

        passenger_service_weight = (
            (
                5.164 * first_class_count
                + 3.846 * business_class_count
                + 2.529 * tourist_class_count
            ) * (design_range / max_mach)**0.225
        ) * passenger_service_mass_scaler

        outputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS] = \
            passenger_service_weight / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J, discrete_inputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        first_class_count = aviary_options.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS)

        business_class_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS)

        tourist_class_count = \
            aviary_options.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS)
        design_range = inputs[Mission.Design.RANGE]
        max_mach = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        passenger_service_mass_scaler = \
            inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER]

        J[
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            Mission.Design.RANGE
        ] = passenger_service_mass_scaler * (
            5.164 * first_class_count
            + 3.846 * business_class_count
            + 2.529 * tourist_class_count
        ) * 0.225 * ((design_range / max_mach)**-0.775) / max_mach / GRAV_ENGLISH_LBM

        J[
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
        ] = (
            (
                5.164 * first_class_count
                + 3.846 * business_class_count
                + 2.529 * tourist_class_count
            ) * (design_range / max_mach)**0.225
        ) / GRAV_ENGLISH_LBM


class AltPassengerServiceMass(om.ExplicitComponent):
    '''
    Define the alternate component to calculate the estimated mass of
    passenger service equipment. The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, val=1.)

        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        aviary_options: AviaryValues = self.options['aviary_options']
        passenger_count = aviary_options.get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')

        passenger_service_mass_scaler = \
            inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER]

        passenger_service_weight = \
            31.7 * passenger_count * passenger_service_mass_scaler

        outputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS] = \
            passenger_service_weight / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J, discrete_inputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        passenger_count = aviary_options.get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')

        J[
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER
        ] = 31.7 * passenger_count / GRAV_ENGLISH_LBM
