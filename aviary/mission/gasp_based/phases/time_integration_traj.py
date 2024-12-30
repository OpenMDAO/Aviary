import numpy as np
from aviary.mission.gasp_based.ode.time_integration_base_classes import SGMTrajBase
from aviary.mission.gasp_based.phases.time_integration_phases import SGMGroundroll, SGMRotation
from aviary.variable_info.enums import Verbosity


class TimeIntegrationTrajBase(SGMTrajBase):
    """Base class for time integration trajectory"""

    def initialize(self):
        super().initialize()
        self.options.declare("cruise_mach", default=0.8)
        self.options.declare("ode_args", types=dict, default=dict())


class FlexibleTraj(TimeIntegrationTrajBase):
    '''
    A traj that can be given a list of phases to build flexible trajectories.
    This is useful for simulating/testing phases one at a time as well as
    allowing users to quickly redefine the phase order of the tracjectory.
    '''

    def initialize(self):
        super().initialize()
        self.options.declare('Phases', default=None)
        self.options.declare('promote_all_auto_ivc', default=False)
        self.options.declare('traj_intermediate_state_output', default=None)
        self.options.declare('traj_final_state_output', default=None)
        self.options.declare('traj_promote_final_output', default=None)
        self.options.declare('traj_promote_initial_input', default=None)
        self.options.declare('traj_initial_state_input', default=None)
        self.options.declare('traj_event_trigger_input', default=None)

    def setup(self):

        ODEs = []
        for phase_name, phase_info in self.options['Phases'].items():
            kwargs = phase_info.get('kwargs', {})
            next_phase = phase_info['builder'](**kwargs)
            next_phase.phase_name = phase_name
            ODEs.append(next_phase)

        self.setup_params(
            ODEs=ODEs,
            promote_all_auto_ivc=self.options['promote_all_auto_ivc'],
            traj_final_state_output=self.options['traj_final_state_output'],
            traj_promote_final_output=self.options['traj_promote_final_output'],

            traj_promote_initial_input=self.options['traj_promote_initial_input'],
            traj_initial_state_input=self.options['traj_initial_state_input'],
            traj_event_trigger_input=self.options['traj_event_trigger_input'],
            traj_intermediate_state_output=self.options['traj_intermediate_state_output'],
        )
        self.declare_partials(["*"], ["*"],)

    def compute(self, inputs, outputs):
        self.compute_params(inputs)

        for phase in self.ODEs:
            phase_name = phase.phase_name
            vals_to_set = self.options['Phases'][phase_name]['user_options']
            if vals_to_set:
                for name, data in vals_to_set.items():
                    var, units = data
                    if name.startswith('attr:'):
                        if isinstance(var, str):
                            val = np.squeeze(self.convert2units(var, inputs[var], units))
                            data = (val, units)
                        setattr(phase, name.replace('attr:', ''), data)
                    elif name.startswith('rotation.'):
                        phase.rotation.set_val(name.replace(
                            'rotation.', ''), var, units=units)
                    else:
                        phase.set_val(name, var, units=units)

        ode_index = 0
        sim_gen = self.compute_traj_loop(self.ODEs[0], inputs, outputs)
        if self.verbosity >= Verbosity.BRIEF:
            print('*'*40)
            print('Starting: '+self.ODEs[ode_index].phase_name)
        for current_problem, sim_result in sim_gen:
            t_final = sim_result.t[-1]
            x_final = sim_result.x[-1, :]
            if type(current_problem) is SGMGroundroll:
                t_start_rotation = t_final

            ode_index += 1
            try:
                next_problem = self.ODEs[ode_index]
            except IndexError:
                next_problem = None

            if self.verbosity >= Verbosity.BRIEF:
                print('Finished: '+current_problem.phase_name)

            if next_problem is not None:
                if type(current_problem) is SGMGroundroll:
                    next_problem.prob.set_val("start_rotation", t_start_rotation)
                elif type(current_problem) is SGMRotation:
                    next_problem.rotation.set_val("start_rotation", t_start_rotation)

                if self.verbosity >= Verbosity.BRIEF:
                    print('Starting: '+next_problem.phase_name)
                sim_gen.send(next_problem)
            else:
                if self.verbosity >= Verbosity.BRIEF:
                    print('Reached the end of the Trajectory!')
                sim_gen.close()
                break

        if self.verbosity >= Verbosity.BRIEF:
            print('t_final', t_final)
            print('x_final', x_final)
            print(self.ODEs[-1].states)
