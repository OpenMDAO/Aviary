import numpy as np
import openmdao.api as om
from openmdao.utils import units
from scipy import interpolate
from simupy.block_diagram import DEFAULT_INTEGRATOR_OPTIONS, SimulationMixin
from simupy.systems import DynamicalSystem

from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variable_meta_data import _MetaData


def add_SGM_required_inputs(group: om.Group, inputs_to_add: dict):
    """
    This is a simple utility that can be used to add inputs that are necessary for SGM but not collocation.
    The SGM integrator expects that all states are inputs to the ODE. If any states aren't required by the EOM
    they can be added to the ODE with this function, in order to minimize the differences between ODEs that are
    used for both SGM and collocation.
    """
    blank_component = om.ExplicitComponent()
    for input, details in inputs_to_add.items():
        blank_component.add_input(input, **details)
    group.add_subsystem(
        'SGM_required_inputs',
        blank_component,
        promotes_inputs=list(inputs_to_add.keys()),
    )


def add_SGM_required_outputs(group: om.Group, outputs_to_add: dict):
    """
    This is a simple utility that can be used to add inputs that are necessary for SGM but not collocation.
    The SGM integrator expects that all state rates are outputs from the ODE. If any state rates aren't
    calculated by the EOM they can be added to the ODE with this function, in order to minimize the
    differences between ODEs that are used for both SGM and collocation.
    """
    iv_comp = om.IndepVarComp()
    for output, details in outputs_to_add.items():
        iv_comp.add_output(output, **details)
    group.add_subsystem(
        'SGM_required_outputs',
        iv_comp,
        promotes_outputs=list(outputs_to_add.keys()),
    )


class event_trigger:
    """
    event_trigger is used by SimuPyProblem to track the information that is required to trigger events during phases.
    The state name, trigger value, and trigger units are required. Optionally a channel_name can be specified if it is
    different from the state name.
    Users are expect to interact with triggers primarily through SimuPyProblem.add_trigger() and
    SimuPyProblem.clear_triggers(), but may want more detailed control over the triggers.
    """

    def __init__(
        self,
        state: str,
        value: tuple[str, float, int],
        units: str,
        channel_name: str = None,
    ):
        self.state = state
        self.value = value
        self.units = units
        if channel_name is None:
            channel_name = state
        self.channel_name = channel_name


class SimuPyProblem(SimulationMixin):
    """Subproblem used as a basis for forward in time integration phases."""

    def __init__(
        self,
        ode,
        aviary_options=None,
        meta_data=_MetaData,
        problem_name='',
        time_independent=False,
        t_name='t_curr',
        states=None,
        alternate_state_names=None,
        blocked_state_names=None,
        alternate_state_rate_names=None,
        parameters=None,
        outputs=None,
        controls=None,
        triggers=None,
        include_state_outputs=False,
        rate_suffix='_rate',
        verbosity=Verbosity.QUIET,
        max_allowable_time=1_000_000,
        adjoint_int_opts=DEFAULT_INTEGRATOR_OPTIONS.copy(),
    ):
        """
        states: a dictionary of the form {state_name:{'units':unit, 'rate':state_rate_name, 'rate_units':state_rate_units}}
        alternate_state_names: a dictionary of the form {state_to_replace:new_state_name}
        blocked_state_names: a list of the form [state_name_to_block]
        alternate_state_rate_names: a dictionary of the form {state_to_modify:new_state_rate_name}
        parameters: a dictionary of the form {parameter_name:unit}
        outputs: a dictionary of the form {output_name:unit}
        controls: a dictionary of the form {control_name:unit}
        include_state_outputs : automatically add the state to the input
        works well for auto-parsed naming, does not check for duplication before adding
        states, parameters, outputs, and controls can also be input as a list of keys for the dictionary.
        """
        default_om_list_args = dict(prom_name=True, val=False, out_stream=None, units=True)

        self.verbosity = verbosity
        self.max_allowable_time = max_allowable_time
        self.adjoint_int_opts = adjoint_int_opts
        self.adjoint_int_opts['nsteps'] = 5000
        self.adjoint_int_opts['name'] = 'dop853'

        self.dt = 0.0
        prob = om.Problem()
        if aviary_options:
            from aviary.interface.methods_for_level2 import AviaryGroup

            prob.model = AviaryGroup(aviary_options=aviary_options, aviary_metadata=meta_data)
        prob.model.add_subsystem(
            'ODE_group',
            ode,
            promotes=['*'],
        )
        self.ode = ode

        self.prob = prob
        setup_model_options(prob, aviary_inputs=aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

        # TODO - This is a hack to mimic the behavior of the old paramport, which
        # contains some initial default values. It is unclear how actual "parameter"
        # values are supposed to propagate from the pre-mission and top ivcs into
        # the SGM phases.
        from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests

        set_params_for_unit_tests(prob)

        prob.final_setup()

        if triggers is None:
            triggers = []
        elif not isinstance(triggers, list):
            triggers = [triggers]
        self.triggers = triggers
        self.event_channel_names = [trigger.channel_name for trigger in triggers]

        self.time_independent = time_independent
        if type(states) is list:
            states = {
                state: {'units': None, 'rate': state + rate_suffix, 'rate_units': None}
                for state in states
            }
        if type(parameters) is list:
            parameters = {parameter: None for parameter in parameters}
        if type(outputs) is list:
            outputs = {output: None for output in outputs}
        if controls is None:
            controls = {}
        elif type(controls) is list:
            controls = {control: None for control in controls}
        if alternate_state_names is None:
            alternate_state_names = {}
        if alternate_state_rate_names is None:
            alternate_state_rate_names = {}

        data = prob.model.list_inputs(includes=controls.keys(), **default_om_list_args)
        control_data = {val.pop('prom_name'): val for key, val in data}
        for control_name, control_units in controls.items():
            if control_units is None:
                controls[control_name] = control_data[control_name]['units']
        self.controls = controls

        if (
            states is None or parameters is None or outputs is None  # or
            # event_names is None
        ):
            data = prob.model.list_outputs(**default_om_list_args)
            model_outputs = [val['prom_name'] for key, val in data]
            data = prob.model.list_inputs(**default_om_list_args)
            model_inputs = [val['prom_name'] for key, val in data]

        if states is None:
            states = {
                output.replace(rate_suffix, ''): {'units': None, 'rate': output}
                for output in model_outputs
                if output.endswith(rate_suffix) or output in alternate_state_names.keys()
            }

            if blocked_state_names is not None:
                for name in blocked_state_names:
                    if name in states.keys():
                        states.pop(name)
            if alternate_state_names:
                for state_key, val in alternate_state_names.items():
                    if state_key in states.keys():  # Used to rename an existing state
                        states[val] = states.pop(state_key)
                    else:  # Used to add a new state
                        states[val] = {'units': None, 'rate': val + rate_suffix}

        if alternate_state_rate_names:
            for state_name, val in alternate_state_rate_names.items():
                if state_name in states.keys():
                    states[state_name]['rate'] = val

        state_rate_names = [val['rate'] for _, val in states.items()]
        data = prob.model.list_outputs(includes=state_rate_names, **default_om_list_args)
        rate_data = {val.pop('prom_name'): val for key, val in data}
        for state_name, state_data in states.items():
            if state_data['rate_units'] is None:
                state_data['rate_units'] = rate_data[state_data['rate']]['units']
            if state_data['units'] is None:
                rate_unit = rate_data[state_data['rate']]['units']
                states[state_name]['units'] = units.simplify_unit('s*' + rate_unit)

        if parameters is None:
            parameters = {
                inp: None
                for inp in set(model_inputs)
                if inp not in list(states.keys()) + list(controls.keys()) + [t_name]
            }

        data = prob.model.list_inputs(includes=parameters.keys(), **default_om_list_args)
        parameter_data = {val.pop('prom_name'): val for key, val in data}
        for parameter_name, parameter_units in parameters.items():
            if parameter_units is None:
                parameters[parameter_name] = parameter_data[parameter_name]['units']

        if outputs is None:
            outputs = {
                outp: None
                for outp in model_outputs
                if outp not in state_rate_names  # + event_names
            }

        for output_name, output_units in outputs.items():
            if output_units is None:
                outputs[output_name] = next(
                    iter(
                        prob.model.get_io_metadata(
                            includes=[output_name], metadata_keys=['units']
                        ).values()
                    )
                )['units']

        if include_state_outputs or outputs == {}:  # prevent empty outputs
            outputs.update({state: data['units'] for state, data in states.items()})

        self.t_name = t_name
        self.states = states
        self.state_names = list(states.keys())

        self.parameters = parameters
        self.outputs = outputs

        self.dim_state = len(states)
        self.dim_output = len(outputs)
        self.dim_input = len(controls)
        self.dim_parameters = len(parameters)
        # TODO: add defensive checks to make sure dimensions match in both setup and
        # calls
        if verbosity >= Verbosity.VERBOSE:
            if problem_name:
                problem_name = '_' + problem_name
            om.n2(prob, outfile='n2_simupy_problem' + problem_name + '.html', show_browser=False)
            with open('input_list_simupy' + problem_name + '.txt', 'w') as outfile:
                prob.model.list_inputs(
                    out_stream=outfile,
                )
            print(states)

    def add_parameter(self, name, units, **kwargs):
        self.parameters[name] = units
        self.dim_parameters = len(self.parameters)

    @property
    def time(self):
        return self.prob.get_val(self.t_name)[0]

    @time.setter
    def time(self, value):
        if self.time_independent or self.time == value:
            return
        self.prob.set_val(self.t_name, value)

    @property
    def state(self):
        return np.array(
            [
                self.prob.get_val(state_name, units=state_data['units'])[0]
                for state_name, state_data in self.states.items()
            ]
        )

    @state.setter
    def state(self, value):
        if np.all(self.state == value):
            return
        for state_name, elem_val in zip(self.states.keys(), value):
            self.prob.set_val(state_name, elem_val, units=self.states[state_name]['units'])

    def compute_along_traj(self, ts, xs):
        self.prob.set_val(self.t_name, ts)
        for state_name, elem_val in zip(self.states.keys(), xs.T):
            self.prob.set_val(state_name, elem_val, units=self.states[state_name]['units'])

    @property
    def control(self):
        return np.array(
            [
                self.prob.get_val(control_name, units=unit)[0]
                for control_name, unit in self.controls.items()
            ]
        )

    @control.setter
    def control(self, value):
        if value is None:
            value = np.array([])
        if (self.control.size == value.size) and np.all(self.control == value):
            return
        for control_name, elem_val in zip(self.controls, value):
            self.prob.set_val(control_name, elem_val, units=self.controls[control_name])

    @property
    def parameter(self):
        return np.array(
            [
                self.prob.get_val(parameter_name, units=unit)[0]
                for parameter_name, unit in self.parameters.items()
            ]
        )

    @parameter.setter
    def parameter(self, value):
        if np.all(self.parameter == value):
            return
        for parameter_name, elem_val in zip(self.parameters, value):
            self.prob.set_val(parameter_name, elem_val)

    @property
    def state_rate(self):
        return np.array(
            [
                self.prob.get_val(state_data['rate'], units=state_data['rate_units'])[0]
                for _, state_data in self.states.items()
            ]
        )

    @property
    def output(self):
        return np.array(
            [
                self.prob.get_val(output_name, units=unit)[0]
                for output_name, unit in self.outputs.items()
            ]
        )

    @property
    def events(self):
        return np.array(
            [
                self.prob.get_val(event_name, units=unit)[0]
                for event_name, unit in zip(self.event_names, self.event_units)
            ]
        )

    @property
    def compute(self):
        # TODO: create cache -- t, x (optionally u) as hash
        # I guess just set_val for every input and output?
        # for compute_totals, force run_model. anywhere else that needs forced
        # run_model?
        return self.prob.run_model

    @property
    def compute_totals(self):
        return self.prob.compute_totals

    def state_equation_function(self, t, x, u=None):
        self.time = t
        self.state = x
        self.control = u
        self.compute()
        return self.state_rate

    def output_equation_function(self, t, x):
        if self.output_nan:
            return np.ones(self.dim_output) * np.nan
        self.time = t
        self.state = x
        self.compute()
        return self.output

    def prepare_to_integrate(self, t0, x0):
        self.output_nan = False
        # self.time = t0
        # self.state = x0
        # self.prob.run_model()
        return self.output_equation_function(t0, x0)

    def update_equation_function(self, t, x, event_channels=None):
        self.output_nan = True
        return x

    def add_trigger(self, state, value, units=None, channel_name=None):
        if isinstance(value, tuple):
            units = value[1]
        if units is None:
            units = self.states[state]['units']
        elif hasattr(self, units):
            units = getattr(self, units)
        if channel_name is None:
            channel_name = state

        self.triggers.append(event_trigger(state, value, units, channel_name))
        self.event_channel_names.append(channel_name)
        self.num_events = len(self.event_channel_names)

    def clear_triggers(self):
        self.triggers = []
        self.event_channel_names = []
        self.num_events = 0

    def event_equation_function(self, t, x):
        self.output_equation_function(t, x)
        self.compute()
        event_values = [self.evaluate_trigger(trigger) for trigger in self.triggers]
        # print(event_values)
        return np.array(event_values)

    def evaluate_trigger(self, trigger: event_trigger):
        trigger_value = trigger.value
        if isinstance(trigger_value, str):
            if hasattr(self, trigger_value):
                trigger_value = getattr(self, trigger_value)
                if isinstance(trigger_value, tuple):
                    trigger_value, trigger.units = trigger_value
            else:
                trigger_value = self.get_val(trigger_value, units=trigger.units).squeeze()
        current_value = self.get_val(trigger.state, units=trigger.units).squeeze()
        return current_value - trigger_value

    @property
    def get_val(self):
        return self.prob.get_val

    @property
    def set_val(self):
        return self.prob.set_val


class SGMTrajBase(om.ExplicitComponent):
    """
    SGMTrajBase is intended to mimic the dymos trajectory used in collocation problems as closely as possible.
    Users are expected to mostly use FlexibleTraj which has some additions and helper functions that are more
    specific to Aviary and aircraft simulation, whereas this base class is intended to be more generic and
    provides a framework for any sort of SGM based trajectory.
    """

    def initialize(self, verbosity=Verbosity.QUIET):
        # needs to get passed to each ODE
        # TODO: param_dict
        self.options.declare('param_dict', default=ParamPort.param_data)
        self.verbosity = verbosity
        self.max_allowable_time = 1_000_000
        self.adjoint_int_opts = DEFAULT_INTEGRATOR_OPTIONS.copy()
        self.adjoint_int_opts['nsteps'] = 5000
        self.adjoint_int_opts['name'] = 'dop853'
        self.additional_parameters = {}

    def add_parameter(self, name, units=None, **kwargs):
        self.additional_parameters['parameters:' + name] = {'units': units}

    def setup_params(
        self,
        ODEs,
        promote_all_auto_ivc=False,
        traj_final_state_output=None,
        traj_promote_final_output=None,
        traj_promote_initial_input=None,
        traj_initial_state_input=None,
        traj_event_trigger_input=None,
        traj_intermediate_state_output=None,
        # traj_promote_intermediate_output=None,
    ):
        """
        API requirements:
            pass ODE's,
            next_problem = f(current_problem, current_result)
            initial_state/time/etc
            next_state from last state/output/event information.

            pass in terminal and integrand output functions with derivatives (components)
            -- anything special for final state, final time?
            declare initial state(s) as parameters to take derivative wrt
            assume all other inputs are parameters for deriv?
        """
        if traj_promote_initial_input is None:
            traj_promote_initial_input = {}
        if traj_initial_state_input is None:
            traj_initial_state_input = []
        # if traj_promote_intermediate_output is None:
        #     traj_promote_intermediate_output = []
        if traj_intermediate_state_output is None:
            traj_intermediate_state_output = []
        if traj_final_state_output is None:
            traj_final_state_output = []
        if traj_promote_final_output is None:
            traj_promote_final_output = []
        if traj_event_trigger_input is None:
            traj_event_trigger_input = []

        self.phase_names = [ode.phase_name for ode in ODEs]

        if promote_all_auto_ivc:
            for ode in ODEs:
                ode: SimuPyProblem
                ode.auto_ivc_vars = ode.prob.model.list_inputs(
                    is_indep_var=True, units=True, out_stream=None
                )
                for abs_name, data in ode.auto_ivc_vars:
                    prom_name = data.pop('prom_name')
                    if '.' in prom_name:
                        continue
                    traj_promote_initial_input[prom_name] = data

            # TODO - This is a hack to mimic the behavior of the old paramport, which
            # contains some initial default values. It is unclear how actual "parameter"
            # values are supposed to propagate from the pre-mission and top ivcs into
            # the SGM phases.
            from aviary.mission.gasp_based.ode.params import params_for_unit_tests

            traj_promote_initial_input = {**params_for_unit_tests, **traj_promote_initial_input}

        self.traj_promote_initial_input = {
            **self.options['param_dict'],
            **traj_promote_initial_input,
            **self.additional_parameters,
        }
        for name, kwargs in self.traj_promote_initial_input.items():
            self.add_input(name, **kwargs)

        final_suffix = '_final'
        self.traj_final_state_output = {
            final_state_output: {
                **dict(
                    name=final_state_output + final_suffix,
                    state_name=final_state_output,
                ),
                **self.add_output(
                    final_state_output + final_suffix,
                    units=ODEs[-1].states[final_state_output]['units'],
                ),
            }
            for final_state_output in traj_final_state_output
        }
        self.traj_intermediate_state_output = {
            phase_name + '_' + intermediate_state_output + final_suffix: {
                **dict(
                    phase_name=phase_name,
                    state_name=intermediate_state_output,
                ),
                **self.add_output(
                    phase_name + '_' + intermediate_state_output + final_suffix,
                    units=ODEs[self.phase_names.index(phase_name)].states[
                        intermediate_state_output
                    ]['units'],
                ),
            }
            for phase_name, intermediate_state_output in traj_intermediate_state_output
        }
        self.traj_promote_final_output = {
            promoted_final_output: {
                **dict(
                    name=promoted_final_output + final_suffix,
                    output_name=promoted_final_output,
                ),
                **self.add_output(
                    promoted_final_output + final_suffix,
                    units=ODEs[-1].outputs[promoted_final_output],
                ),
            }
            for promoted_final_output in traj_promote_final_output
        }
        # self.traj_promote_intermediate_output = {
        #     promoted_intermediate_output: {
        #         **dict(
        #             name=phase_name+'_'+promoted_intermediate_output+final_suffix,
        #             output_name=promoted_intermediate_output,
        #         ),
        #         **self.add_output(
        #             phase_name+'_'+promoted_intermediate_output,
        #             units=ODEs[-1].outputs[promoted_intermediate_output],
        #         ),
        #     }
        #     for phase_name,promoted_intermediate_output in traj_promote_intermediate_output
        # }
        self.all_traj_outputs = {
            **self.traj_intermediate_state_output,
            **self.traj_final_state_output,
            # **self.traj_promote_intermediate_output,
            **self.traj_promote_final_output,
        }
        initial_suffix = '_initial'
        self.traj_initial_state_input = {
            initial_state_input: {
                **dict(name=initial_state_input + initial_suffix),
                **self.add_input(
                    initial_state_input + initial_suffix,
                    units=ODEs[0].states[initial_state_input]['units'],
                ),
            }
            for initial_state_input in traj_initial_state_input
        }

        # TODO: assumes state, not output
        trigger_suffix = 'trigger'
        self.traj_event_trigger_input = {
            event_trigger_input: {
                **dict(
                    name='_'.join(
                        [
                            ODEs[self.phase_names.index(event_trigger_input[0])].__class__.__name__,
                            event_trigger_input[1],
                            trigger_suffix,
                        ]
                    )
                ),
                **self.add_input(
                    '_'.join(
                        [
                            ODEs[self.phase_names.index(event_trigger_input[0])].__class__.__name__,
                            event_trigger_input[1],
                            trigger_suffix,
                        ]
                    ),
                    units=ODEs[self.phase_names.index(event_trigger_input[0])].states[
                        event_trigger_input[1]
                    ]['units'],
                ),
            }
            for event_trigger_input in traj_event_trigger_input
        }
        self.ODEs = ODEs
        self.declare_partials(
            ['*'],
            ['*'],
        )

    def compute_params(self, inputs):
        for input in self.traj_promote_initial_input.keys():
            for ode in self.ODEs:
                try:
                    ode.set_val(input.removeprefix('parameters:'), inputs[input])
                except KeyError:
                    if self.verbosity >= Verbosity.VERBOSE:
                        print('*** Input not found:', ode, input)
                    pass

    def compute_traj_loop(self, first_problem, inputs, outputs, t0=0.0, state0=None):
        if self.verbosity >= Verbosity.VERBOSE:
            print('initializing compute_traj_loop')
        sim_results = []
        sim_problems = [first_problem]
        t = t0
        if state0 is not None:
            state = state0
        else:
            state = np.array(
                [
                    inputs[state_name + '_initial'].squeeze()
                    if state_name in self.traj_initial_state_input
                    else 0.0
                    for state_name in first_problem.state_names
                ]
            ).squeeze()

        while True:
            current_problem = sim_problems[-1]
            current_problem.initial_condition = state

            sim_result = current_problem.simulate(
                (t, self.max_allowable_time),
            )
            if sim_result.t.shape[0] == 2:
                print('\n' * 3, 'IMMEDIATE PHASE TERMINATION', current_problem, '\n' * 2)
            sim_results.append(sim_result)

            t = sim_result.t[-1]
            x = sim_result.x[-1, :]

            for output_name, output in self.traj_intermediate_state_output.items():
                if output['phase_name'] == current_problem.phase_name:
                    outputs[output_name] = sim_result.x[
                        -1, current_problem.state_names.index(output['state_name'])
                    ]

            # TODO: is there a better way to do this? Perhaps don't use for loop -- use
            # while True ?Ij
            try:
                try_next_problem = yield current_problem, sim_result
            except GeneratorExit:
                if self.verbosity >= 2:
                    print('stop iteration 1')
                break

            if try_next_problem is not None:
                next_problem = try_next_problem
            else:
                try:
                    next_problem = yield current_problem, sim_result
                except GeneratorExit:
                    if self.verbosity >= Verbosity.VERBOSE:
                        print('stop iteration 2')
                    break

                if self.verbosity >= Verbosity.VERBOSE:
                    print(' was on problem:', current_problem, '\n got back:', next_problem)
            # compute the output at the final condition to make sure all outputs are current
            current_problem.output_equation_function(t, x)
            state = np.array(
                [
                    current_problem.prob.get_val(state_name, units=state_data['units'])
                    for state_name, state_data in next_problem.states.items()
                ]
            ).squeeze()
            sim_problems.append(next_problem)

        if self.verbosity >= Verbosity.VERBOSE:
            print('ended loop')

        # wrap main loop
        self.sim_results = sim_results
        self.sim_problems = sim_problems

        # trajectory-specific outputs
        for output in self.traj_final_state_output:
            output_name = self.traj_final_state_output[output]['name']
            state_name = self.traj_final_state_output[output]['state_name']

            outputs[output_name] = sim_results[-1].x[
                -1, sim_problems[-1].state_names.index(state_name)
            ]

        for output in self.traj_promote_final_output:
            promoted_name = self.traj_promote_final_output[output]['name']
            output_name = self.traj_promote_final_output[output]['output_name']

            outputs[promoted_name] = sim_results[-1].y[
                -1, list(sim_problems[-1].outputs.keys()).index(output_name)
            ]

        self.last_inputs = np.array(list(inputs.values()))

    def compute_partials(self, inputs, J):
        self.compute_params(inputs)
        # defensive check -- should really make sure ALL inputs are the same, need a
        # deep copy
        # just calling compute_params doesn't fix it -- possibly different trajectory!

        if np.any(self.last_inputs != np.array(list(inputs.values()))):
            raise ValueError(
                'Attempting to run compute_partials when the last compute inputs did not match',
            )

        param_dict = self.options['param_dict']

        # assume the first problem has the most states?
        costate_reses = {output: [] for output in self.all_traj_outputs}
        tf_total = self.sim_results[-1].t[-1]

        next_res = self.sim_results[-1]
        next_prob = self.sim_problems[-1]

        df_dxs = []
        df_dparams = []
        dg_dxs = []
        f_minuses = []
        f_pluses = [np.zeros(self.sim_problems[-1].dim_state)]
        state_updates = [next_res.x[-1, :]]
        dh_dxs = [np.eye(next_prob.dim_state)]

        dh_dparams = [np.zeros((next_prob.dim_state, len(param_dict)))]

        # keep directionality of forward for plant, backward for adjoint by caching
        # everything in order to ensure fewest number of ODE calls with smallest
        # step-size sum(abs(x[i]-x[i-1]), )
        costate_ics = []
        param_derivs = []

        for output in self.all_traj_outputs:
            output_name = self.all_traj_outputs[output]['name']
            costate = np.zeros(next_prob.dim_state)
            param_deriv = np.zeros(len(param_dict))

            if output in self.traj_final_state_output:
                costate[next_prob.state_names.index(output)] = 1.0
            else:  # in self.traj_promote_final_output
                next_prob.state_equation_function(next_res.t[-1], next_res.x[-1, :])
                costate[:] = next_prob.compute_totals(
                    output, next_prob.state_names, return_format='array'
                ).squeeze()

                param_deriv[:] = next_prob.compute_totals(
                    output, list(param_dict.keys()), return_format='array'
                ).squeeze()

            param_derivs.append(param_deriv)
            costate_ics.append(costate)

        # pre-compute data for adjoint
        for phase_idx, res, prob in zip(
            range(len(self.sim_results), 0, -1),
            self.sim_results[::-1],
            self.sim_problems[::-1],
        ):
            # build time-varying co-state matrix
            df_dx_data = np.empty(res.x.shape + (res.x.shape[-1],))

            if param_dict:
                df_dparam_data = np.empty(res.x.shape + (len(param_dict),))

            last_res_idx = res.t.shape[0] - 1

            num_active_event_channels = 0

            state_rate = prob.state_equation_function(res.t[-1], res.x[-1, :])
            if prob is not self.sim_problems[0]:
                f_minuses.append(state_rate)

            for channel_idx, channel_name in enumerate(prob.event_channel_names):
                if np.argmin(np.abs(res.e[-1, :])) not in [channel_idx]:
                    continue

                num_active_event_channels += 1
                dg_dx = np.zeros((1, prob.dim_state))

                if channel_name in prob.state_names:
                    dg_dx[0, prob.state_names.index(channel_name)] = 1.0
                else:
                    dg_dx[0, :] = prob.compute_totals(
                        [channel_name], prob.state_names, return_format='array'
                    )

                # TODO: actually save dg_dts
                if channel_name == prob.t_name:
                    dg_dt = 1.0
                else:
                    dg_dt = 0.0

            dg_dxs.append(dg_dx)

            if num_active_event_channels != 1:
                raise ValueError(
                    'Somehow ended a phase without an event? assume '
                    'time in the future?? but currently no time-based '
                    'events are used'
                )

            for idx, (t, x) in enumerate(zip(res.t[::-1], res.x[::-1, :])):
                state_rate = prob.state_equation_function(t, x)

                if (idx == last_res_idx) and (prob is not self.sim_problems[0]):
                    next_prob = self.sim_problems[self.sim_problems.index(prob) - 1]

                    f_plus = np.zeros(next_prob.dim_state)
                    plus_rate = state_rate

                    # NOTE / TODO: should enforce that all states in all ODEs exist
                    # in eachother (even if only as output). Don't like assuming
                    # zero
                    # state_update = np.zeros(next_prob.dim_state)
                    state_update = np.ones(next_prob.dim_state) * np.inf
                    dh_dx = np.zeros((next_prob.dim_state,) * 2)
                    dh_dparam = np.zeros((next_prob.dim_state, len(param_dict)))

                    # here and co-state assume number of states is only decreasing
                    # forward in time
                    for state_name in next_prob.state_names:
                        state_idx = next_prob.state_names.index(state_name)

                        if state_name in prob.state_names:
                            f_plus[state_idx] = plus_rate[prob.state_names.index(state_name)]

                            # state_update[
                            #    next_prob.state_names.index(state_name)
                            # ] = x[prob.state_names.index(state_name)]

                            # TODO: make sure index multiplying next_pronb costate
                            # lines up -- since costate is pre-filled to next_prob's
                            # order, the continuous terms should be right
                            # column should map to
                            dh_dx[state_idx, state_idx] = 1.0

                        elif state_name in prob.outputs.keys():
                            state_update[state_idx] = res.y[
                                -1, list(prob.outputs.keys()).index(state_name)
                            ]

                            dh_j_dx = prob.compute_totals(
                                [state_name], prob.state_names, return_format='array'
                            ).squeeze()

                            dh_dparam[state_idx, :] = prob.compute_totals(
                                [state_name], list(param_dict.keys()), return_format='array'
                            ).squeeze()

                            for state_name_2 in prob.state_names:
                                # I'm actually computing dh_dx.T
                                # dh_dx rows are new state, columns are old state
                                # now, dh_dx.T rows are old state, columns are new
                                # so I think this is right
                                dh_dx[
                                    next_prob.state_names.index(state_name_2),
                                    state_idx,
                                ] = dh_j_dx[prob.state_names.index(state_name_2)]

                        else:
                            state_update[state_idx] = 0.0

                    f_pluses.append(f_plus)
                    state_updates.append(state_update)
                    dh_dxs.append(dh_dx)
                    dh_dparams.append(dh_dparam)

                state_rate_names = [val['rate'] for _, val in prob.states.items()]
                df_dx_data[idx, :, :] = prob.compute_totals(
                    state_rate_names, prob.state_names, return_format='array'
                ).T
                if param_dict:
                    df_dparam_data[idx, ...] = prob.compute_totals(
                        state_rate_names, list(param_dict.keys()), return_format='array'
                    )

            k = min(3, res.t.shape[0] - 1)
            skip_interp = (k == 1) and np.isclose(res.t[0], res.t[1])

            # TODO: why is this failing?
            if skip_interp:
                mean_df_dx = np.mean(df_dx_data, axis=0)

                df_dxs.append(lambda t: mean_df_dx)
            else:
                try:
                    df_dxs.append(
                        interpolate.make_interp_spline(tf_total - res.t[::-1], df_dx_data, k=k)
                    )
                except Exception as e:
                    print('EXCEPTION!', k, res.t, df_dx_data, sep='\n')
                    print('check: ', (k == 1) and (res.t[0] == res.t[1]))
                    breakpoint()
                    raise e

            if param_dict:
                if skip_interp:
                    mean_df_dparam = np.mean(df_dparam_data, axis=0)
                    df_dparams.append(lambda t: mean_df_dparam)
                else:
                    df_dparams.append(
                        interpolate.make_interp_spline(tf_total - res.t[::-1], df_dparam_data, k=k)
                    )
            else:
                df_dparams.append(None)

        if self.verbosity == Verbosity.DEBUG:
            print('data....')
            print('dgs', dg_dxs)
            print('f-', f_minuses)
            print('f+', f_pluses)

            print(
                'size check:',
                len(self.sim_problems),
                len(dg_dxs),
                len(f_minuses),
                len(f_pluses),
            )

        # main loop
        for output, costate_ic, param_deriv in zip(
            self.all_traj_outputs, costate_ics, param_derivs
        ):
            output_name = self.all_traj_outputs[output]['name']
            next_prob = self.sim_problems[-1]
            costate = costate_ic
            lamda_dot_plus = np.zeros_like(costate)

            # self.sim_results[-1].x[-1, next_prob.state_names.index(output)]
            if self.verbosity >= Verbosity.VERBOSE:
                print('\nstarting partial for %s' % output, costate)

            dg_dt = 0.0

            for (
                phase_idx,
                res,
                prob,
                df_dx,
                df_dparam,
                dg_dx,
                f_minus,
                f_plus,
                state_update,
                dh_dx,
                dh_dparam,
            ) in zip(
                range(len(self.sim_results), 0, -1),
                self.sim_results[::-1],
                self.sim_problems[::-1],
                df_dxs,
                df_dparams,
                dg_dxs,
                f_minuses,
                f_pluses,
                state_updates,
                dh_dxs,
                dh_dparams,
            ):
                t0, tf = tf_total - res.t[[-1, 0]]

                # assumes only 1 of time, state, or output dependence
                # assume no discontinuous state update, would need an API for that in
                # compute as well --
                # but assume some form of event has happened

                # already checked that event_channel_names was well-defined in the
                # pre-compute, so will just assign the co-state just once
                for channel_idx, channel_name in enumerate(prob.event_channel_names):
                    if np.argmin(np.abs(res.e[-1, :])) not in [channel_idx]:
                        continue

                    state_disc = res.x[-1] - state_update
                    state_disc[np.where(np.isinf(state_update))] = 0.0

                    if channel_name != prob.t_name:
                        lamda_dot = df_dx(res.t[-1]) @ costate
                        # lamda_dot_plus = lamda_dot
                        if self.verbosity == Verbosity.DEBUG:
                            if np.any(state_disc):
                                print(
                                    'update is non-zero!',
                                    prob,
                                    prob.state_names,
                                    state_disc,
                                    costate,
                                    lamda_dot,
                                )
                                print(
                                    'inner product becomes...',
                                    state_disc[None, :] @ dh_dx @ lamda_dot_plus[:, None],
                                    state_disc[None, :] @ dh_dx.T @ lamda_dot_plus[:, None],
                                )
                            print('dh_dx for', prob, prob.state_names, '\n', dh_dx)
                            print('costate', costate)
                        costate_update_terms = [
                            dh_dx.T @ costate[:, None],
                            # costate[:, None],
                            # TODO: should this be f_plus? probably not
                            (dg_dx.T @ (f_plus - f_minus)[None, :] @ costate[:, None])
                            / (dg_dx @ f_minus),
                            # don't believe in lamda_dot terms anymore
                            # -(dg_dx.T @ state_disc[None, :] @ dh_dx.T @ lamda_dot_plus[:, None]) / (dg_dx@f_minus),
                        ]

                        # TODO: is this wrong?
                        costate[:] = np.sum(costate_update_terms, axis=0).squeeze()

                    if (
                        event_key := (prob, channel_name, channel_idx)
                    ) in self.traj_event_trigger_input:
                        event_trigger_name = self.traj_event_trigger_input[event_key]['name']
                        if self.verbosity >= Verbosity.VERBOSE:
                            print('setting event trigger data', event_trigger_name)
                        J[output_name, event_trigger_name] = (
                            +costate[None, :] @ (f_minus - f_plus) / (dg_dt + dg_dx @ f_minus)
                            # +(lamda_dot_plus[None, :] @ dh_dx @ state_disc[None, :])/(dg_dt + dg_dx@f_minus)
                        )

                    # how to account for terminal event? through costate IC.
                    # TODO: Is this wrong?
                    param_deriv += (costate[None, :] @ dh_dparam).squeeze()

                # build co-state systems

                def co_state_rate(t, costate, *args):
                    return df_dx(t) @ costate

                if self.verbosity >= Verbosity.VERBOSE:
                    print('dim_state:', prob.dim_state, 'ic:', costate)

                costate_sys = DynamicalSystem(
                    state_equation_function=co_state_rate, dim_state=prob.dim_state
                )
                costate_sys.initial_condition = costate

                # simulate co-state system
                co_res = costate_sys.simulate((t0, tf), integrator_options=self.adjoint_int_opts)
                costate_reses[output].append(co_res)

                if param_dict:
                    df_dparam_val = df_dparam(co_res.t)
                    param_deriv_integrand_data = np.matmul(
                        co_res.x[:, None, :], df_dparam_val
                    ).squeeze()
                    try:
                        param_deriv_integrand = interpolate.make_interp_spline(
                            co_res.t,
                            np.atleast_1d(param_deriv_integrand_data),
                            # k=df_dparam.k
                            k=min(3, co_res.t.shape[0] - 1),
                        )
                    except ValueError as e:
                        print(
                            'HIT VALUE ERROR!',
                            output,
                            prob,
                            co_res.t.shape,
                            co_res.x.shape,
                            df_dparam_val.shape,
                            df_dparam.k,
                            'final_results:\n\n',
                            t0,
                            tf,
                            co_res.t,
                            co_res.x,
                        )
                        raise e
                    param_deriv_integrand_antideriv = param_deriv_integrand.antiderivative()

                    # TODO: is the sign wrong here?
                    param_deriv -= param_deriv_integrand_antideriv(
                        t0
                    ) - param_deriv_integrand_antideriv(tf)

                # consume initial condition
                if prob is not self.sim_problems[0]:
                    next_prob = self.sim_problems[self.sim_problems.index(prob) - 1]
                else:
                    break
                costate = np.zeros(next_prob.dim_state)
                lamda_dot_plus = np.zeros_like(costate)
                lamda_dot_plus_rate = co_state_rate(co_res.t[-1], co_res.x[-1])

                # TODO: do co-states need unit changes? probably not...
                for state_name in prob.state_names:
                    costate[next_prob.state_names.index(state_name)] = co_res.x[
                        -1, prob.state_names.index(state_name)
                    ]
                    lamda_dot_plus[next_prob.state_names.index(state_name)] = lamda_dot_plus_rate[
                        prob.state_names.index(state_name)
                    ]

            for state_to_deriv, metadata in self.traj_initial_state_input.items():
                param_name = metadata['name']
                J[output_name, param_name] = costate_reses[output][-1].x[
                    -1, prob.state_names.index(state_to_deriv)
                ]
            for param_deriv_val, param_deriv_name in zip(param_deriv, param_dict):
                J[output_name, param_deriv_name] = param_deriv_val
        self.costate_reses = costate_reses


class _killer_comp(om.ExplicitComponent):
    """
    This component will stop the execution of the integration during compute,
    this is useful for debugging trajectories by getting the initial value
    of variables for a phase and then exiting.
    """

    def compute(self, inputs, outputs):
        print(f'exit in {self.name}')
        exit()
