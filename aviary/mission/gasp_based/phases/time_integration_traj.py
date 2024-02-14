import numpy as np
import openmdao.api as om
from aviary.mission.gasp_based.ode.time_integration_base_classes import SimuPyProblem, SGMTrajBase
from aviary.mission.gasp_based.phases.time_integration_phases import SGMGroundroll, SGMRotation, SGMAscentCombined, SGMAccel, SGMClimb, SGMCruise, SGMDescent

from aviary.variable_info.enums import SpeedType
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Mission, Dynamic

DEBUG = 0


class TimeIntegrationTrajBase(SGMTrajBase):
    def initialize(self):
        super().initialize()
        self.options.declare("cruise_mach", default=0.8)
        self.options.declare("ode_args", types=dict, default=dict())
        self.options.declare("ode_args_pyc", types=dict, default=dict())
        self.options.declare("pyc_phases", default=list())


class FlexibleTraj(TimeIntegrationTrajBase):
    '''
    A traj that can be given a list of phases to build flexible trajectories.
    This is useful for simulating/testing phases one at a time as well as
    allowing users to quickly redefine the phase order during the tracjectory.
    '''

    def initialize(self):
        super().initialize()
        self.options.declare('Phases', default=None)
        self.options.declare('promote_all_auto_ivc', default=False)
        self.options.declare('traj_final_state_output', default=None)
        self.options.declare('traj_promote_final_output', default=None)
        self.options.declare('traj_promote_initial_input', default=None)
        self.options.declare('traj_initial_state_input', default=None)
        self.options.declare('traj_event_trigger_input', default=None)

    def setup(self):

        ODEs = []
        for phase_name, phase_info in self.options['Phases'].items():
            next_phase = phase_info['ode']
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
        )
        self.declare_partials(["*"], ["*"],)

    def compute(self, inputs, outputs):
        self.compute_params(inputs)

        for phase in self.ODEs:
            phase_name = phase.phase_name
            vals_to_set = self.options['Phases'][phase_name]['vals_to_set']
            if vals_to_set:
                for name, data in vals_to_set.items():
                    if name.startswith('attr:'):
                        setattr(phase, name.replace('attr:', ''), inputs[data['val']])
                    elif name.startswith('rotation.'):
                        phase.rotation.set_val(name.replace(
                            'rotation.', ''), data['val'], units=data['units'])
                    else:
                        phase.set_val(name, data['val'], units=data['units'])

        ode_index = 0
        sim_gen = self.compute_traj_loop(self.ODEs[0], inputs, outputs)
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

            print('Finished: '+current_problem.phase_name)
            # print([name+'_'+data['units']
            #       for name, data in current_problem.states.items()])
            # print(x_final)
            # print('outputs')
            # for output_name, unit in current_problem.outputs.items():
            #     val = current_problem.get_val(output_name, units=unit)[0]
            #     print(output_name+':', val, unit)

            # print(self.outputs.items())
            # print(current_problem.output)
            if next_problem is not None:
                if type(current_problem) is SGMGroundroll:
                    next_problem.prob.set_val("start_rotation", t_start_rotation)
                elif type(current_problem) is SGMRotation:
                    next_problem.rotation.set_val("start_rotation", t_start_rotation)
                # print('\n\n')
                print('Starting: '+next_problem.phase_name)
                # print([name+'_'+data['units']
                #       for name, data in next_problem.states.items()])
                sim_gen.send(next_problem)
            else:
                print('Reached the end of the Trajectory!')
                sim_gen.close()
                break

        print('t_final', t_final)
        print('x_final', x_final)
        print(self.ODEs[-1].states)
