import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportCargoContainersMass(om.ExplicitComponent):
    '''
    Calculate the estimated mass for cargo containers. Use for both
    traditional and blended-wing-body type transports. The methodology is based on
    the FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, val=1.0)

        add_aviary_input(self, Aircraft.CrewPayload.CARGO_MASS, val=0.0)

        add_aviary_input(self, Aircraft.CrewPayload.BAGGAGE_MASS, val=0.0)

        add_aviary_output(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        scaler = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER]
        cargo = inputs[Aircraft.CrewPayload.CARGO_MASS]
        baggage = inputs[Aircraft.CrewPayload.BAGGAGE_MASS]

        temp = (cargo + baggage) / 950.0 + 0.99
        container_count = sin_int4(temp)
        cargo_container_weight = container_count * 175.0 * scaler

        outputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS] = \
            cargo_container_weight / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J, discrete_inputs=None):
        scaler = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER]
        cargo = inputs[Aircraft.CrewPayload.CARGO_MASS]
        baggage = inputs[Aircraft.CrewPayload.BAGGAGE_MASS]

        temp = (cargo + baggage) / 950.0 + 0.99
        container_count = sin_int4(temp)

        partial = dydx_sin_int4(temp) / 950.0 * 175.0

        J[
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER
        ] = container_count * 175.0 / GRAV_ENGLISH_LBM

        J[
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            Aircraft.CrewPayload.CARGO_MASS
        ] = partial * scaler / GRAV_ENGLISH_LBM

        J[
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            Aircraft.CrewPayload.BAGGAGE_MASS
        ] = partial * scaler / GRAV_ENGLISH_LBM


# region TODO: move this to an appropriate module for import
def sin_int4(val):
    '''
    Define a smooth, differentialbe approximation to the 'int' function.
    '''
    return sin_int(sin_int(sin_int(sin_int(val)))) - 0.5


def dydx_sin_int4(val):
    '''
    Define the derivative (dy/dx) of sin_int4, at x = val.
    '''
    y0 = sin_int(val)
    y1 = sin_int(y0)
    y2 = sin_int(y1)

    dydx3 = dydx_sin_int(y2)
    dydx2 = dydx_sin_int(y1)
    dydx1 = dydx_sin_int(y0)
    dydx0 = dydx_sin_int(val)

    dydx = dydx3 * dydx2 * dydx1 * dydx0

    return dydx


# 'int' function can be approximated by recursively applying this sin function
# which makes a smooth, differentialbe function
def sin_int(val):
    '''
    Define one step in approximating the 'int' function with a smooth,
    differentialbe function.
    '''
    int_val = val - np.sin(2*np.pi*(val+0.5))/(2*np.pi)

    return int_val


def dydx_sin_int(val):
    '''
    Define the derivative (dy/dx) of sin_int, at x = val.
    '''
    dydx = 1.0 - np.cos(2 * np.pi * (val + 0.5))

    return dydx
# endregion TODO: move this to an appropriate module for import
