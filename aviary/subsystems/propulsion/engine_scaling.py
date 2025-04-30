import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.utils import EngineModelVariables, max_variables
from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

MACH = EngineModelVariables.MACH
ALTITUDE = EngineModelVariables.ALTITUDE
THROTTLE = EngineModelVariables.THROTTLE
HYBRID_THROTTLE = EngineModelVariables.HYBRID_THROTTLE
THRUST = EngineModelVariables.THRUST
TAILPIPE_THRUST = EngineModelVariables.TAILPIPE_THRUST
GROSS_THRUST = EngineModelVariables.GROSS_THRUST
SHAFT_POWER = EngineModelVariables.SHAFT_POWER
SHAFT_POWER_CORRECTED = EngineModelVariables.SHAFT_POWER_CORRECTED
RAM_DRAG = EngineModelVariables.RAM_DRAG
FUEL_FLOW = EngineModelVariables.FUEL_FLOW
ELECTRIC_POWER = EngineModelVariables.ELECTRIC_POWER_IN
NOX_RATE = EngineModelVariables.NOX_RATE
TEMPERATURE = EngineModelVariables.TEMPERATURE_T4
# EXIT_AREA = EngineModelVariables.EXIT_AREA

# these variables are not outputs or are variables that should not get scaled
skip_variables = [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE, TEMPERATURE]


class EngineScaling(om.ExplicitComponent):
    """
    Scales an engine's thrust, fuel flow rate, nox rate, exit area, and electric power
    based on provided scaling factors.

    Can be vectorized for all unique engine present on aircraft. Each index represents a
    single instance of an engine model.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'engine_variables',
            types=dict,
            desc='dict of variables to be scaled for this engine with units',
        )

        add_aviary_option(self, Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, units='lbm/h')
        add_aviary_option(self, Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM)
        add_aviary_option(self, Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM)
        add_aviary_option(self, Aircraft.Engine.SCALE_PERFORMANCE)
        add_aviary_option(self, Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER)
        add_aviary_option(self, Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER)
        add_aviary_option(self, Mission.Summary.FUEL_FLOW_SCALER)

    def setup(self):
        nn = self.options['num_nodes']
        engine_variables = self.options['engine_variables']

        add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, val=1.0)

        self.add_input(
            Dynamic.Atmosphere.MACH,
            val=np.zeros(nn),
            desc='current Mach number',
            units='unitless',
        )

        # loop through all variables, special handling for fuel flow to output negative version
        # add outputs for 'max' counterpart of variables that have them
        for variable in engine_variables:
            if variable not in skip_variables:
                self.add_input(
                    variable.value + '_unscaled',
                    val=np.zeros(nn),
                    units=engine_variables[variable],
                )

                if variable is FUEL_FLOW:
                    self.add_output(
                        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
                        val=np.zeros(nn),
                        units=engine_variables[variable],
                    )
                else:
                    self.add_output(
                        variable.value,
                        val=np.zeros(nn),
                        units=engine_variables[variable],
                    )

                if variable in max_variables:
                    self.add_input(
                        max_variables[variable] + '_unscaled',
                        val=np.zeros(nn),
                        units=engine_variables[variable],
                    )

                    self.add_output(
                        max_variables[variable],
                        val=np.zeros(nn),
                        units=engine_variables[variable],
                    )

    def compute(self, inputs, outputs):
        options = self.options
        nn = options['num_nodes']
        engine_variables = options['engine_variables']
        scale_performance = options[Aircraft.Engine.SCALE_PERFORMANCE]

        subsonic_fuel_factor = options[Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER]
        supersonic_fuel_factor = options[Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER]
        constant_fuel_term = options[Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM]
        linear_fuel_term = options[Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM]
        constant_fuel_flow, _ = options[Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION]
        mission_fuel_scaler = options[Mission.Summary.FUEL_FLOW_SCALER]

        # thrust-based engine scaling factor
        engine_scale_factor = inputs[Aircraft.Engine.SCALE_FACTOR]
        mach_number = inputs[Dynamic.Atmosphere.MACH]

        scale_factor = 1
        fuel_flow_scale_factor = np.ones(nn, dtype=engine_scale_factor.dtype)

        # if len(scale_idx[0]) > 0:
        if scale_performance:
            # special factor to scale fuel flow based on thrust when scaling is permitted
            # NOTE mission-specific fuel flow scaling factor is overwritten by
            #      scale_performance = False

            # Calculate fuel flow rate scaling factor using FLOPS-derived equation
            fuel_flow_equation_scaling = (
                1 + constant_fuel_term + linear_fuel_term * (1 - engine_scale_factor)
            )

            # use dtype to make complex safe
            fuel_flow_mach_scaling = (
                np.ones(nn, dtype=engine_scale_factor.dtype) * subsonic_fuel_factor
            )
            supersonic_idx = np.where(mach_number >= 1.0)
            fuel_flow_mach_scaling[supersonic_idx] = supersonic_fuel_factor

            fuel_flow_scale_factor = (
                engine_scale_factor
                * fuel_flow_mach_scaling
                * fuel_flow_equation_scaling
                * mission_fuel_scaler
            )

            scale_factor = engine_scale_factor

        # loop through all variables, singling out fuel flow to have special scaling
        # compute 'max' counterpart of variables that have them
        for variable in engine_variables:
            if variable not in skip_variables:
                if variable is FUEL_FLOW:
                    outputs[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE] = -(
                        inputs['fuel_flow_rate_unscaled'] * fuel_flow_scale_factor
                        + constant_fuel_flow
                    )
                else:
                    outputs[variable.value] = inputs[variable.value + '_unscaled'] * scale_factor

                    if variable in max_variables:
                        outputs[variable.value + '_max'] = (
                            inputs[variable.value + '_max_unscaled'] * scale_factor
                        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        engine_variables = self.options['engine_variables']

        # matrix derivatives have known sparsity pattern - specified here
        r = np.arange(nn)
        c = np.tile(0, nn)

        for variable in engine_variables:
            if variable not in skip_variables:
                if variable is FUEL_FLOW:
                    self.declare_partials(
                        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
                        Aircraft.Engine.SCALE_FACTOR,
                        rows=r,
                        cols=c,
                    )
                    self.declare_partials(
                        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
                        'fuel_flow_rate_unscaled',
                        rows=r,
                        cols=r,
                    )
                else:
                    self.declare_partials(
                        variable.value,
                        Aircraft.Engine.SCALE_FACTOR,
                        rows=r,
                        cols=c,
                    )
                    self.declare_partials(
                        variable.value,
                        variable.value + '_unscaled',
                        rows=r,
                        cols=r,
                    )

                    if variable in max_variables:
                        self.declare_partials(
                            variable.value + '_max',
                            Aircraft.Engine.SCALE_FACTOR,
                            rows=r,
                            cols=c,
                        )
                        self.declare_partials(
                            variable.value + '_max',
                            variable.value + '_max_unscaled',
                            rows=r,
                            cols=r,
                        )

    def compute_partials(self, inputs, J):
        options = self.options
        nn = options['num_nodes']
        engine_variables = options['engine_variables']
        scale_performance = options[Aircraft.Engine.SCALE_PERFORMANCE]

        subsonic_fuel_factor = options[Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER]
        supersonic_fuel_factor = options[Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER]
        constant_fuel_term = options[Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM]
        linear_fuel_term = options[Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM]
        mission_fuel_scaler = options[Mission.Summary.FUEL_FLOW_SCALER]

        mach_number = inputs[Dynamic.Atmosphere.MACH]
        engine_scale_factor = inputs[Aircraft.Engine.SCALE_FACTOR]

        # determine which mach-based fuel flow scaler is applied at each node
        fuel_flow_mach_scaling = np.ones(nn) * subsonic_fuel_factor
        fuel_flow_mach_scaling[mach_number >= 1.0] = supersonic_fuel_factor

        fuel_flow_deriv = np.ones(nn, dtype=engine_scale_factor.dtype)
        fuel_flow_scale_deriv = np.zeros(nn, dtype=engine_scale_factor.dtype)
        scale_factor = 1
        deriv_factor = 0

        if scale_performance:
            if FUEL_FLOW in engine_variables:
                # Calculate fuel flow rate scaling factor using FLOPS-derived equation
                fuel_flow_equation_scaling = (
                    1 + constant_fuel_term + linear_fuel_term * (1 - engine_scale_factor)
                )

                fuel_flow_deriv = (
                    -engine_scale_factor
                    * fuel_flow_mach_scaling
                    * fuel_flow_equation_scaling
                    * mission_fuel_scaler
                )

                fuel_flow_scale_deriv = (
                    -fuel_flow_mach_scaling
                    * mission_fuel_scaler
                    * inputs['fuel_flow_rate_unscaled']
                    * (
                        1
                        + linear_fuel_term
                        + constant_fuel_term
                        - (2 * linear_fuel_term * engine_scale_factor)
                    )
                )

            scale_factor = engine_scale_factor
            deriv_factor = 1.0

        for variable in engine_variables:
            if variable not in skip_variables:
                if variable is FUEL_FLOW:
                    J[
                        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
                        'fuel_flow_rate_unscaled',
                    ] = fuel_flow_deriv
                    J[
                        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
                        Aircraft.Engine.SCALE_FACTOR,
                    ] = fuel_flow_scale_deriv
                else:
                    J[variable.value, variable.value + '_unscaled'] = scale_factor
                    J[variable.value, Aircraft.Engine.SCALE_FACTOR] = (
                        inputs[variable.value + '_unscaled'] * deriv_factor
                    )

                    if variable in max_variables:
                        J[variable.value + '_max', variable.value + '_max_unscaled'] = scale_factor
                        J[variable.value + '_max', Aircraft.Engine.SCALE_FACTOR] = (
                            inputs[variable.value + '_max_unscaled'] * deriv_factor
                        )
