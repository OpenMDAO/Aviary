import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor, distributed_thrust_factor)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class TransportAirCondMass(om.ExplicitComponent):
    '''
    Calculates the mass of the air conditioning group using the transport/general
    aviation method. The methodology is based on the FLOPS weight equations,
    modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS_SCALER, val=1.0)

        add_aviary_input(self, Aircraft.Avionics.MASS, val=0.0)

        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, val=0.0)

        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, val=0.0)

        add_aviary_output(self, Aircraft.AirConditioning.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        pax = aviary_options.get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')

        scaler = inputs[Aircraft.AirConditioning.MASS_SCALER]
        avionics_wt = inputs[Aircraft.Avionics.MASS] * GRAV_ENGLISH_LBM
        height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        max_mach = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        outputs[Aircraft.AirConditioning.MASS] = \
            ((3.2 * (planform * height)**0.6 + 9 * pax**0.83)
             * max_mach + 0.075 * avionics_wt) * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        pax = aviary_options.get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')

        scaler = inputs[Aircraft.AirConditioning.MASS_SCALER]
        avionics_wt = inputs[Aircraft.Avionics.MASS] * GRAV_ENGLISH_LBM
        height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        max_mach = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        planform_exp = planform**0.6
        height_exp = height**0.6
        pax_exp = pax**0.83

        J[Aircraft.AirConditioning.MASS, Aircraft.AirConditioning.MASS_SCALER] = (
            (3.2 * planform_exp * height_exp + 9 * pax_exp) *
            max_mach + 0.075 * avionics_wt) / GRAV_ENGLISH_LBM

        J[Aircraft.AirConditioning.MASS, Aircraft.Avionics.MASS] = \
            0.075 * scaler

        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.MAX_HEIGHT] = \
            1.92 * planform_exp * height**-0.4 * max_mach * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.PLANFORM_AREA] = \
            1.92 * planform**-0.4 * height_exp * max_mach * scaler / GRAV_ENGLISH_LBM


class AltAirCondMass(om.ExplicitComponent):
    '''
    Calculates the mass of the air conditioning group using the alternate method.
    The methodology is based on the FLOPS weight equations, modified to output
    mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.AirConditioning.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(of=Aircraft.AirConditioning.MASS, wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_pax = aviary_options.get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')

        scaler = inputs[Aircraft.AirConditioning.MASS_SCALER]

        outputs[Aircraft.AirConditioning.MASS] = \
            26.0 * num_pax * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_pax = aviary_options.get_val(
            Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')

        J[Aircraft.AirConditioning.MASS, Aircraft.AirConditioning.MASS_SCALER] = \
            26.0 * num_pax / GRAV_ENGLISH_LBM
