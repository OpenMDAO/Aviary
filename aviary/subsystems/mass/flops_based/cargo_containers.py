import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.functions import sin_int4, dydx_sin_int4
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportCargoContainersMass(om.ExplicitComponent):
    """
    Calculate the estimated mass for cargo containers. Use for both
    traditional and blended-wing-body type transports. The methodology is based on
    the FLOPS weight equations, modified to output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.BAGGAGE_MASS, units='lbm')

        add_aviary_output(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        scaler = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER]
        cargo = inputs[Aircraft.CrewPayload.CARGO_MASS]
        baggage = inputs[Aircraft.CrewPayload.BAGGAGE_MASS]

        temp = (cargo + baggage) / 950.0 + 0.99
        container_count = sin_int4(temp)
        cargo_container_weight = container_count * 175.0 * scaler

        outputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS] = (
            cargo_container_weight / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J, discrete_inputs=None):
        scaler = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER]
        cargo = inputs[Aircraft.CrewPayload.CARGO_MASS]
        baggage = inputs[Aircraft.CrewPayload.BAGGAGE_MASS]

        temp = (cargo + baggage) / 950.0 + 0.99
        container_count = sin_int4(temp)

        partial = dydx_sin_int4(temp) / 950.0 * 175.0

        J[
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER,
        ] = container_count * 175.0 / GRAV_ENGLISH_LBM

        J[Aircraft.CrewPayload.CARGO_CONTAINER_MASS, Aircraft.CrewPayload.CARGO_MASS] = (
            partial * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.CrewPayload.CARGO_CONTAINER_MASS, Aircraft.CrewPayload.BAGGAGE_MASS] = (
            partial * scaler / GRAV_ENGLISH_LBM
        )
