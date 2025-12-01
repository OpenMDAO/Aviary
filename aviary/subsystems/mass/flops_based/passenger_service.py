"""
Define utilities to calculate the estimated mass of passenger service
equipment.
"""

import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class PassengerServiceMass(om.ExplicitComponent):
    """
    Define the default component to calculate the estimated mass of passenger
    service equipment. The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS)
        add_aviary_option(self, Mission.Constraints.MAX_MACH)

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, units='unitless')
        add_aviary_input(self, Mission.Design.RANGE, units='NM')

        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        first_class_count = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        business_class_count = self.options[Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS]
        tourist_class_count = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]

        design_range = inputs[Mission.Design.RANGE]
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        passenger_service_mass_scaler = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER]

        passenger_service_weight = (
            (5.164 * first_class_count + 3.846 * business_class_count + 2.529 * tourist_class_count)
            * (design_range / max_mach) ** 0.225
        ) * passenger_service_mass_scaler

        outputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS] = (
            passenger_service_weight / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J, discrete_inputs=None):
        first_class_count = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        business_class_count = self.options[Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS]
        tourist_class_count = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]

        design_range = inputs[Mission.Design.RANGE]
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        passenger_service_mass_scaler = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER]

        J[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, Mission.Design.RANGE] = (
            passenger_service_mass_scaler
            * (
                5.164 * first_class_count
                + 3.846 * business_class_count
                + 2.529 * tourist_class_count
            )
            * 0.225
            * ((design_range / max_mach) ** -0.775)
            / max_mach
            / GRAV_ENGLISH_LBM
        )

        J[
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
        ] = (
            (5.164 * first_class_count + 3.846 * business_class_count + 2.529 * tourist_class_count)
            * (design_range / max_mach) ** 0.225
        ) / GRAV_ENGLISH_LBM


class AltPassengerServiceMass(om.ExplicitComponent):
    """
    Define the alternate component to calculate the estimated mass of
    passenger service equipment. The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        passenger_count = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        passenger_service_mass_scaler = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER]

        passenger_service_weight = 31.7 * passenger_count * passenger_service_mass_scaler

        outputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS] = (
            passenger_service_weight / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J, discrete_inputs=None):
        passenger_count = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        J[
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
        ] = 31.7 * passenger_count / GRAV_ENGLISH_LBM
