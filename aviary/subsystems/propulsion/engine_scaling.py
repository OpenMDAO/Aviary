import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.propulsion.utils import EngineModelVariables, default_units
from aviary.variable_info.functions import add_aviary_input
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

independent_variables = [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE]


class EngineScaling(om.ExplicitComponent):
    '''
    Scales an engine's thrust, fuel flow rate, nox rate, exit area, and electric power
    based on provided scaling factors.

    Can be vectorized for all unique engine present on aircraft. Each index represents a
    single instance of an engine model.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

        self.options.declare(
            'engine_variables',
            types=list,
            desc='list of variables to be scaled for this engine',
        )

    def setup(self):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_variables = self.options['engine_variables']

        add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, val=1.0)

        self.add_input(Dynamic.Mission.MACH, val=np.zeros(nn),
                       desc='current Mach number', units='unitless')

        for variable in engine_variables:
            if variable not in independent_variables:
                self.add_input(
                    variable.value + '_unscaled',
                    val=np.zeros(nn),
                    units=default_units[variable],
                )
                if variable is FUEL_FLOW:
                    self.add_output(
                        Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        val=np.zeros(nn),
                        units=default_units[variable],
                    )
                else:
                    self.add_output(
                        variable.value, val=np.zeros(nn), units=default_units[variable]
                    )

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_variables = self.options['engine_variables']
        scale_performance = options.get_val(Aircraft.Engine.SCALE_PERFORMANCE)

        subsonic_fuel_factor = options.get_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER)
        supersonic_fuel_factor = options.get_val(
            Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER)
        constant_fuel_term = options.get_val(
            Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM)
        linear_fuel_term = options.get_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM)
        constant_fuel_flow = options.get_val(
            Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, units='lbm/h')
        mission_fuel_scaler = options.get_val(Mission.Summary.FUEL_FLOW_SCALER)

        # thrust-based engine scaling factor
        engine_scale_factor = inputs[Aircraft.Engine.SCALE_FACTOR]
        mach_number = inputs[Dynamic.Mission.MACH]

        independent_variables = [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE]

        scale_factor = 1
        fuel_flow_scale_factor = np.ones(nn, dtype=engine_scale_factor.dtype)

        # if len(scale_idx[0]) > 0:
        if scale_performance:
            # special factor to scale fuel flow based on thrust when scaling is permitted
            # NOTE mission-specific fuel flow scaling factor is overwritten by
            #      scale_performance = False

            # Calculate fuel flow rate scaling factor using FLOPS-derived equation
            fuel_flow_equation_scaling = (
                1 + constant_fuel_term + linear_fuel_term * (1 - engine_scale_factor))

            # use dtype to make complex safe
            fuel_flow_mach_scaling = np.ones(
                nn, dtype=engine_scale_factor.dtype) * subsonic_fuel_factor
            supersonic_idx = np.where(mach_number >= 1.0)
            fuel_flow_mach_scaling[supersonic_idx] = supersonic_fuel_factor

            fuel_flow_scale_factor = engine_scale_factor * fuel_flow_mach_scaling\
                * fuel_flow_equation_scaling * mission_fuel_scaler

            scale_factor = engine_scale_factor

        for variable in engine_variables:
            if variable not in independent_variables:
                if variable is FUEL_FLOW:
                    outputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE] = (
                        -(inputs['fuel_flow_rate_unscaled'] * fuel_flow_scale_factor)
                        - constant_fuel_flow
                    )
                else:
                    outputs[variable.value] = (
                        inputs[variable.value + '_unscaled'] * scale_factor
                    )

    def setup_partials(self):
        nn = self.options['num_nodes']
        engine_variables = self.options['engine_variables']

        # matrix derivatives have known sparsity pattern - specified here
        r = np.arange(nn)
        c = np.tile(0, nn)

        independent_variables = [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE]

        for variable in engine_variables:
            if variable not in independent_variables:
                if variable is FUEL_FLOW:
                    self.declare_partials(
                        Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        Aircraft.Engine.SCALE_FACTOR,
                        rows=r,
                        cols=c,
                        val=1.0,
                    )
                    self.declare_partials(
                        Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        'fuel_flow_rate_unscaled',
                        rows=r,
                        cols=r,
                        val=1.0,
                    )
                else:
                    self.declare_partials(
                        variable.value,
                        Aircraft.Engine.SCALE_FACTOR,
                        rows=r,
                        cols=c,
                        val=1.0,
                    )
                    self.declare_partials(
                        variable.value,
                        variable.value + '_unscaled',
                        rows=r,
                        cols=r,
                        val=1.0,
                    )

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_variables = self.options['engine_variables']
        scale_performance = options.get_val(Aircraft.Engine.SCALE_PERFORMANCE)

        subsonic_fuel_factor = options.get_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER)
        supersonic_fuel_factor = options.get_val(
            Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER)
        constant_fuel_term = options.get_val(
            Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, units='unitless')
        linear_fuel_term = options.get_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM)
        # constant_fuel_flow = options.get_val(
        #     Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, units='lbm/h')
        mission_fuel_scaler = options.get_val(Mission.Summary.FUEL_FLOW_SCALER)

        mach_number = inputs[Dynamic.Mission.MACH]
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
                    1
                    + constant_fuel_term
                    + linear_fuel_term * (1 - engine_scale_factor)
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
            if variable not in independent_variables:
                if variable is FUEL_FLOW:
                    J[
                        Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        'fuel_flow_rate_unscaled',
                    ] = fuel_flow_deriv
                    J[
                        Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        Aircraft.Engine.SCALE_FACTOR,
                    ] = fuel_flow_scale_deriv
                else:
                    J[variable.value, variable.value + '_unscaled'] = scale_factor
                    J[variable.value, Aircraft.Engine.SCALE_FACTOR] = (
                        inputs[variable.value + '_unscaled'] * deriv_factor
                    )
