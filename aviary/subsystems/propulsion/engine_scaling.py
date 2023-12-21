import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


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

    def setup(self):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']

        # count = len(options.get_val('engine_models'))  # number of unique engine models

        # add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, val=np.ones(count))
        add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, val=1.0)

        self.add_input(Dynamic.Mission.MACH, val=np.zeros(nn),
                       desc='current Mach number', units='unitless')

        # to vectorize, shape is (nn, count)
        self.add_input('thrust_net_unscaled', val=np.zeros(nn), units='lbf',
                       desc='Current net thrust produced (unscaled)')

        self.add_input('thrust_net_max_unscaled', val=np.zeros(nn),
                       units='lbf', desc='Current maximum possible net thrust (unscaled)')

        self.add_input('fuel_flow_rate_unscaled', val=np.zeros(nn),
                       units='lbm/h', desc='Current fuel flow rate (unscaled)')

        self.add_input('electric_power_unscaled', val=np.zeros(nn),
                       units='kW', desc='Current electric power consumption (unscaled)')

        self.add_input('nox_rate_unscaled', val=np.zeros(nn), units='lbm/h',
                       desc='Current NOx emission rate (unscaled)')

        # self.add_input('exit_area_unscaled', val=np.zeros(nn), units='ft**2',
        #                desc='Current engine nozzle exit area')

        self.add_output(Dynamic.Mission.THRUST, val=np.zeros(nn), units='lbf',
                        desc='Current net thrust produced')

        self.add_output(Dynamic.Mission.THRUST_MAX, val=np.zeros(nn),
                        units='lbf', desc='Current maximum possible net thrust')

        self.add_output(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE, val=np.zeros(nn),
                        units='lbm/h', desc='Current fuel flow rate (negative)')

        self.add_output(Dynamic.Mission.ELECTRIC_POWER, val=np.zeros(nn),
                        units='kW', desc='Current electric power consumption')

        self.add_output(Dynamic.Mission.NOX_RATE, val=np.zeros(nn),
                        units='lbm/h', desc='Current NOx emission rate')

        # self.add_output(Dynamic.Mission.EXIT_AREA, val=np.zeros(nn),
        #                 units='ft**2', desc='Current engine nozzle exit area')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        # count = len(options.get_val('engine_models'))  # number of unique engine models
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

        unscaled_net_thrust = inputs['thrust_net_unscaled']
        unscaled_max_thrust = inputs['thrust_net_max_unscaled']
        unscaled_fuel_flow_rate = inputs['fuel_flow_rate_unscaled']
        unscaled_electric_power = inputs['electric_power_unscaled']
        unscaled_nox_rate = inputs['nox_rate_unscaled']
        # unscaled_exit_area = inputs['exit_area_unscaled']

        scale_factor = 1
        fuel_flow_scale_factor = np.ones(nn, dtype=engine_scale_factor.dtype)
        # scale_idx = np.where(scale_performance)

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

            # fuel_flow_scale_factor[:, scale_idx[0]] = engine_scale_factor[scale_idx]\
            #     * fuel_flow_mach_scaling[:, scale_idx[0]]\
            #     * fuel_flow_equation_scaling[scale_idx]\
            #     * mission_fuel_scaler
            fuel_flow_scale_factor = engine_scale_factor * fuel_flow_mach_scaling\
                * fuel_flow_equation_scaling * mission_fuel_scaler

            scale_factor = engine_scale_factor

        # scale factor only applies if engine performance is scaled - default to 1 otherwise
        # scale_factor = np.ones(count, dtype=engine_scale_factor.dtype)
        # scale_factor[scale_idx] = engine_scale_factor[scale_idx]

        outputs[Dynamic.Mission.THRUST] = unscaled_net_thrust * scale_factor
        outputs[Dynamic.Mission.THRUST_MAX] = unscaled_max_thrust * scale_factor
        # user-specified constant_fuel_flow value is currently not scaled with engine
        outputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE] = - \
            (unscaled_fuel_flow_rate * fuel_flow_scale_factor) - constant_fuel_flow

        # TODO determine proper scaling for electric power, nox, and exit area
        outputs[Dynamic.Mission.ELECTRIC_POWER] = unscaled_electric_power * scale_factor
        outputs[Dynamic.Mission.NOX_RATE] = unscaled_nox_rate * scale_factor
        # outputs[Dynamic.Mission.EXIT_AREA] = unscaled_exit_area * scale_factor

    def setup_partials(self):
        nn = self.options['num_nodes']
        # options = self.options['aviary_options']
        # count = len(options.get_val('engine_models'))  # number of unique engine models

        # matrix derivatives have known sparsity pattern - specified here
        # r = np.arange(nn * count, dtype=int)
        # c = np.tile(np.arange(count, dtype=int), (nn))
        r = np.arange(nn)
        c = np.tile(0, nn)

        self.declare_partials(
            Dynamic.Mission.THRUST,
            Aircraft.Engine.SCALE_FACTOR,
            rows=r, cols=c,
            val=1.0)
        self.declare_partials(
            Dynamic.Mission.THRUST,
            'thrust_net_unscaled',
            rows=r, cols=r,
            val=1.0)

        self.declare_partials(
            Dynamic.Mission.THRUST_MAX,
            Aircraft.Engine.SCALE_FACTOR,
            rows=r, cols=c,
            val=1.0)
        self.declare_partials(
            Dynamic.Mission.THRUST_MAX,
            'thrust_net_max_unscaled',
            rows=r, cols=r,
            val=1.0)

        self.declare_partials(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            Aircraft.Engine.SCALE_FACTOR,
            rows=r, cols=c,
            val=1.0)
        self.declare_partials(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            'fuel_flow_rate_unscaled',
            rows=r, cols=r,
            val=1.0)

        self.declare_partials(
            Dynamic.Mission.ELECTRIC_POWER,
            Aircraft.Engine.SCALE_FACTOR,
            rows=r, cols=c,
            val=1.0)
        self.declare_partials(
            Dynamic.Mission.ELECTRIC_POWER,
            'electric_power_unscaled',
            rows=r, cols=r,
            val=1.0)

        self.declare_partials(
            Dynamic.Mission.NOX_RATE,
            Aircraft.Engine.SCALE_FACTOR,
            rows=r, cols=c,
            val=1.0)
        self.declare_partials(
            Dynamic.Mission.NOX_RATE,
            'nox_rate_unscaled',
            rows=r, cols=r,
            val=1.0)

        # self.declare_partials(
        #     Dynamic.Mission.EXIT_AREA,
        #     Aircraft.Engine.SCALE_FACTOR,
        #     rows=r, cols=c,
        #     val=1.0)
        # self.declare_partials(
        #     Dynamic.Mission.EXIT_AREA,
        #     'exit_area_unscaled',
        #     rows=r, cols=r,
        #     val=1.0)

        # self.declare_partials(
        #     Dynamic.Mission.EXIT_AREA,
        #     Aircraft.Engine.SCALE_FACTOR,
        #     rows=r, cols=c,
        #     val=1.0)
        # self.declare_partials(
        #     Dynamic.Mission.EXIT_AREA,
        #     'exit_area_unscaled',
        #     rows=r, cols=r,
        #     val=1.0)

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        # count = len(options.get_val('engine_models'))  # number of unique engine models
        scale_performance = options.get_val(Aircraft.Engine.SCALE_PERFORMANCE)

        subsonic_fuel_factor = options.get_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER)
        supersonic_fuel_factor = options.get_val(
            Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER)
        constant_fuel_term = options.get_val(
            Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, units='unitless')
        linear_fuel_term = options.get_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM)
        constant_fuel_flow = options.get_val(
            Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, units='lbm/h')
        mission_fuel_scaler = options.get_val(Mission.Summary.FUEL_FLOW_SCALER)

        mach_number = inputs[Dynamic.Mission.MACH]
        unscaled_net_thrust = inputs['thrust_net_unscaled']
        unscaled_max_thrust = inputs['thrust_net_max_unscaled']
        unscaled_fuel_flow_rate = inputs['fuel_flow_rate_unscaled']
        unscaled_electric_power = inputs['electric_power_unscaled']
        unscaled_nox_rate = inputs['nox_rate_unscaled']
        # unscaled_exit_area = inputs['exit_area_unscaled']

        engine_scale_factor = inputs[Aircraft.Engine.SCALE_FACTOR]

        # determine which mach-based fuel flow scaler is applied at each node
        fuel_flow_mach_scaling = np.ones(nn) * subsonic_fuel_factor
        fuel_flow_mach_scaling[mach_number >= 1.0] = supersonic_fuel_factor

        fuel_flow_deriv = np.ones(nn, dtype=engine_scale_factor.dtype)
        fuel_flow_scale_deriv = np.zeros(nn, dtype=engine_scale_factor.dtype)

        # scale factor only applies if engine performance is scaled - default to 1 otherwise
        # scale_factor = np.ones(count, dtype=engine_scale_factor.dtype)
        # scale_factor[scale_idx] = engine_scale_factor[scale_idx]
        scale_factor = 1

        # deriv_factor = np.zeros(nn)
        # deriv_factor[:, scale_idx[0]] = 1.0
        deriv_factor = 0

        # scale_idx = np.where(scale_performance)
        # if len(scale_idx[0]) > 0:
        if scale_performance:
            # Calculate fuel flow rate scaling factor using FLOPS-derived equation
            fuel_flow_equation_scaling = (
                1 + constant_fuel_term +
                linear_fuel_term * (1 - engine_scale_factor)
            )

            fuel_flow_deriv = -engine_scale_factor * fuel_flow_mach_scaling\
                * fuel_flow_equation_scaling * mission_fuel_scaler

            # fuel_flow_scale_deriv = fuel_flow_mach_scaling[:, scale_idx[0]]\
            #     * mission_fuel_scaler\
            #     * (unscaled_fuel_flow_rate[:, scale_idx[0]]
            #        + constant_fuel_flow[scale_idx]
            #        )\
            #     * (1 + linear_fuel_term[scale_idx]
            #        + constant_fuel_term[scale_idx]
            #        - (2 * linear_fuel_term[scale_idx]
            #           * engine_scale_factor[scale_idx]
            #           )
            #        )

            fuel_flow_scale_deriv = -fuel_flow_mach_scaling * mission_fuel_scaler\
                * unscaled_fuel_flow_rate \
                * (1 + linear_fuel_term + constant_fuel_term - (2 * linear_fuel_term
                                                                * engine_scale_factor))

            scale_factor = engine_scale_factor
            deriv_factor = 1.0

        # J[Dynamic.THRUST, 'thrust_net_unscaled'] = np.tile(
        #     scale_factor, nn)
        # J[Dynamic.THRUST, Aircraft.Engine.SCALE_FACTOR] = np.ravel(
        #     unscaled_net_thrust * deriv_factor)

        # J[Dynamic.THRUST_MAX, 'thrust_net_max_unscaled'] = np.tile(
        #     scale_factor, nn)
        # J[Dynamic.THRUST_MAX, Aircraft.Engine.SCALE_FACTOR] = np.ravel(
        #     unscaled_max_thrust * deriv_factor)

        # J[Dynamic.FUEL_FLOW_RATE_NEGATIVE, 'fuel_flow_rate_unscaled'] = -np.ravel(
        #     fuel_flow_deriv)
        # J[Dynamic.FUEL_FLOW_RATE_NEGATIVE, Aircraft.Engine.SCALE_FACTOR] = -np.ravel(
        #     fuel_flow_scale_deriv)

        # J[Dynamic.ELECTRIC_POWER, 'electric_power_unscaled'] = np.tile(
        #     scale_factor, nn)
        # J[Dynamic.ELECTRIC_POWER, Aircraft.Engine.SCALE_FACTOR] = np.ravel(
        #     unscaled_electric_power * deriv_factor)

        # J[Dynamic.NOX_RATE, 'nox_rate_unscaled'] = np.tile(
        #     scale_factor, nn)
        # J[Dynamic.NOX_RATE, Aircraft.Engine.SCALE_FACTOR] = np.ravel(
        #     unscaled_nox_rate * deriv_factor)

        # J[Dynamic.EXIT_AREA, 'exit_area_unscaled'] = np.tile(
        #     scale_factor, nn)
        # J[Dynamic.EXIT_AREA, Aircraft.Engine.SCALE_FACTOR] = np.ravel(
        #     unscaled_exit_area * deriv_factor)

        J[Dynamic.Mission.THRUST, 'thrust_net_unscaled'] = scale_factor
        J[Dynamic.Mission.THRUST, Aircraft.Engine.SCALE_FACTOR] = unscaled_net_thrust * deriv_factor

        J[Dynamic.Mission.THRUST_MAX, 'thrust_net_max_unscaled'] = scale_factor
        J[Dynamic.Mission.THRUST_MAX,
            Aircraft.Engine.SCALE_FACTOR] = unscaled_max_thrust * deriv_factor

        J[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            'fuel_flow_rate_unscaled'] = fuel_flow_deriv
        J[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            Aircraft.Engine.SCALE_FACTOR] = fuel_flow_scale_deriv

        J[Dynamic.Mission.ELECTRIC_POWER, 'electric_power_unscaled'] = scale_factor
        J[Dynamic.Mission.ELECTRIC_POWER,
            Aircraft.Engine.SCALE_FACTOR] = unscaled_electric_power * deriv_factor

        J[Dynamic.Mission.NOX_RATE, 'nox_rate_unscaled'] = scale_factor
        J[Dynamic.Mission.NOX_RATE, Aircraft.Engine.SCALE_FACTOR] = unscaled_nox_rate * deriv_factor

        # J[Dynamic.Mission.EXIT_AREA, 'exit_area_unscaled'] = scale_factor
        # J[Dynamic.Mission.EXIT_AREA, Aircraft.Engine.SCALE_FACTOR] = unscaled_exit_area * deriv_factor
