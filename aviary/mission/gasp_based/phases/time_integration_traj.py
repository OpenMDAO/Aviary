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
        self.options.declare('traj_final_state_output', default=None)
        self.options.declare('traj_promote_final_output', default=None)
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
            traj_final_state_output=self.options['traj_final_state_output'],
            traj_promote_final_output=self.options['traj_promote_final_output'],

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
            if next_problem is not None:
                if type(current_problem) is SGMGroundroll:
                    next_problem.prob.set_val("start_rotation", t_start_rotation)
                elif type(current_problem) is SGMRotation:
                    next_problem.rotation.set_val("start_rotation", t_start_rotation)
                print('Starting: '+next_problem.phase_name)
                sim_gen.send(next_problem)
            else:
                print('Reached the end of the Trajectory!')
                sim_gen.close()
                break

        print('t_final', t_final)
        print('x_final', x_final)
        print('states', self.ODEs[-1].state_names)
        print('units', self.ODEs[-1].state_units)


# class SGMTraj1(TimeIntegrationTrajBase):
#     '''
#     This combines the phases from brake release to landing
#     '''

#     def initialize(self):
#         super().initialize()

#     def setup(self):
#         """
#         API requirements:
#             pass ODE's,
#             next_problem = f(current_problem, current_result)
#             initial_state/time/etc
#             next_state from last state/output/event information

#             pass in terminal and integrand output functions with derivatives (components)
#             -- anything special for final state, final time?
#             declare initial state(s) as parameters to take derivative wrt
#             assume all other inputs are parameters for deriv?
#         """

#         # distinction is really "meta" inputs to the trajectory itself and ODE
#         # parameters that should get automatically passed (previously `passed_inputs`)
#         # may also need to just have a specific call out for initial condition
#         # parameters?

#         # the "meta" inputs are really initial condition and/or event termination
#         # paramters, like initial states or state triggers
#         # need an API for that
#         # I guess there are similarly at least two categories of outputs, terminal and
#         # integral terms. or just say ode must provide it as a state so theyre all meyer
#         # terms

#         # actual ODE setup
#         groundroll = SGMGroundroll(
#             ode_args=(self.options["ode_args_pyc"] if "groundroll" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )
#         rotation = SGMRotation(
#             ode_args=(self.options["ode_args_pyc"] if "groundroll" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )
#         ascent = SGMAscentCombined(
#             ode_args=(self.options["ode_args_pyc"] if "ascent" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )
#         accel = SGMAccel(
#             ode_args=(self.options["ode_args_pyc"] if "accel" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )
#         climb1 = SGMClimb(
#             input_speed_type=SpeedType.EAS,
#             input_speed_units="kn",
#             ode_args=(self.options["ode_args_pyc"] if "climb" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )

#         climb2 = SGMClimb(
#             input_speed_type=SpeedType.EAS,
#             input_speed_units="kn",
#             ode_args=(self.options["ode_args_pyc"] if "climb" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )
#         climb3 = SGMClimb(
#             input_speed_type=SpeedType.MACH,
#             input_speed_units="unitless",
#             ode_args=(self.options["ode_args_pyc"] if "climb" in
#                       self.options["pyc_phases"] else self.options["ode_args"])
#         )
#         self.setup_params(
#             ODEs=[
#                 groundroll, rotation, ascent, accel, climb1, climb2, climb3
#             ],
#             traj_final_state_output=[Dynamic.Mission.MASS, Dynamic.Mission.DISTANCE,],
#             traj_promote_final_output=[Dynamic.Mission.ALTITUDE_RATE,
#                                        Dynamic.Mission.FLIGHT_PATH_ANGLE, "TAS"],

#             traj_initial_state_input=[Dynamic.Mission.MASS],
#             traj_event_trigger_input=[
#                 # specify ODE, output_name, with units that SimuPyProblem expects
#                 # assume event function is of form ODE.output_name - value
#                 # third key is event_idx associated with input
#                 (groundroll, "TAS", 0,),
#                 (climb3, Dynamic.Mission.ALTITUDE, 0,),
#             ],
#         )

#     def compute(self, inputs, outputs):
#         self.compute_params(inputs)

#         # ODE setup
#         (groundroll, rotation, ascent, accel, climb1, climb2, climb3) = self.ODEs

#         cruise_alt = inputs["SGMClimb_"+Dynamic.Mission.ALTITUDE+"_trigger"]
#         cruise_mach = self.options["cruise_mach"]

#         groundroll.VR_value = inputs["SGMGroundroll_TAS_trigger"]

#         climb1.set_val("alt_trigger", 10_000, units="ft")
#         climb2.set_val("alt_trigger", cruise_alt, units="ft")
#         climb3.set_val("alt_trigger", cruise_alt, units="ft")

#         speed_string = {SpeedType.EAS: 'EAS',
#                         SpeedType.TAS: 'TAS', SpeedType.MACH: 'mach'}
#         climb1.set_val(speed_string[climb1.input_speed_type],
#                        250., units=climb1.input_speed_units)
#         climb2.set_val(speed_string[climb2.input_speed_type],
#                        270., units=climb2.input_speed_units)
#         climb3.set_val(speed_string[climb3.input_speed_type], cruise_mach,
#                        units=climb3.input_speed_units)

#         climb1.set_val("speed_trigger", cruise_mach, units=None)
#         climb2.set_val("speed_trigger", cruise_mach, units=None)
#         climb3.set_val("speed_trigger", 0.0, units=None)

#         for t_var in ["t_init_gear", "t_init_flaps"]:
#             ascent.set_val(t_var, 10_000.)
#         ascent.rotation.set_val("start_rotation", 10_000.)

#         sim_gen = self.compute_traj_loop(groundroll, inputs, outputs)

#         for current_problem, sim_result in sim_gen:

#             t = sim_result.t[-1]
#             x = sim_result.x[-1, :]
#             event_idx = np.argmin(np.abs(sim_result.e[-1, :]))

#             # trajectory-specific phase switching
#             if current_problem is groundroll:
#                 if DEBUG:
#                     print("starting rotation")
#                 rotation.prob.set_val("start_rotation", t)
#                 ascent.rotation.set_val("start_rotation", t)
#                 next_problem = rotation
#             elif current_problem is rotation:
#                 if DEBUG:
#                     print("starting ascent")
#                 next_problem = ascent
#             elif current_problem is ascent:
#                 next_problem = accel
#                 if DEBUG:
#                     print("starting accel")
#             elif current_problem is accel:
#                 if DEBUG:
#                     print("climb1")
#                 next_problem = climb1
#             elif current_problem is climb1:
#                 if DEBUG:
#                     print("climb2")
#                 if event_idx != 0:
#                     if DEBUG:
#                         print("expected to hit alt trigger for climb1")
#                 next_problem = climb2
#             elif current_problem is climb2:
#                 if DEBUG:
#                     print("climb3")
#                 if event_idx != 1:
#                     if DEBUG:
#                         print("expected to hit speed trigger for climb2")
#                 next_problem = climb3
#             elif current_problem is climb3:
#                 if DEBUG:
#                     print("climb ending")
#                 if event_idx != 0:
#                     if DEBUG:
#                         print("expected to hit alt trigger for climb3")
#                 next_problem = None
#             else:
#                 if DEBUG:
#                     print("unexpected termination")
#                 next_problem = None

#             if next_problem is not None:
#                 sim_gen.send(next_problem)
#             else:
#                 sim_gen.close()
#                 break


# class SGMTraj2(TimeIntegrationTrajBase):
#     '''
#     This combines the phases from end of cruise to landing
#     '''

#     def setup(self):
#         """
#         API requirements:
#             pass ODE's,
#             next_problem = f(current_problem, current_result)
#             initial_state/time/etc
#             next_state from last state/output/event information

#             pass in terminal and integrand output functions with derivatives (components)
#             -- anything special for final state, final time?
#             declare initial state(s) as parameters to take derivative wrt
#             assume all other inputs are parameters for deriv?
#         """

#         desc1 = SGMDescent(
#             input_speed_type=SpeedType.MACH,
#             input_speed_units="unitless",
#             speed_trigger_units='kn',
#             ode_args=self.options["ode_args"],
#         )

#         desc2 = SGMDescent(
#             input_speed_type=SpeedType.EAS,
#             input_speed_units="kn",
#             speed_trigger_units='kn',
#             ode_args=self.options["ode_args"],
#         )
#         desc3 = SGMDescent(
#             input_speed_type=SpeedType.EAS,
#             input_speed_units="kn",
#             speed_trigger_units='kn',
#             ode_args=self.options["ode_args"],
#         )

#         self.setup_params(
#             ODEs=[desc1, desc2, desc3],
#             traj_final_state_output=[Dynamic.Mission.MASS, Dynamic.Mission.DISTANCE],
#             traj_initial_state_input=[Dynamic.Mission.MASS,
#                                       Dynamic.Mission.DISTANCE, Dynamic.Mission.ALTITUDE],
#         )
#         self.declare_partials(["*"], ["*"],)

#     def compute(self, inputs, outputs):

#         self.compute_params(inputs)

#         # ODE setup
#         desc1, desc2, desc3 = self.ODEs

#         cruise_mach = self.options["cruise_mach"]

#         desc1.set_val("alt_trigger", 10_000, units="ft")
#         desc2.set_val("alt_trigger", 10_000, units="ft")
#         desc3.set_val("alt_trigger", 1_000, units="ft")

#         speed_string = {SpeedType.EAS: 'EAS',
#                         SpeedType.TAS: 'TAS', SpeedType.MACH: 'mach'}
#         desc1.set_val(speed_string[desc1.input_speed_type], cruise_mach,
#                       units=desc1.input_speed_units)
#         desc2.set_val(speed_string[desc2.input_speed_type],
#                       350., units=desc2.input_speed_units)
#         desc3.set_val(speed_string[desc3.input_speed_type],
#                       250., units=desc3.input_speed_units)

#         desc1.set_val("speed_trigger", 350.0, units=desc1.speed_trigger_units)
#         desc2.set_val("speed_trigger", 0.0, units=desc2.speed_trigger_units)
#         desc3.set_val("speed_trigger", 0.0, units=desc3.speed_trigger_units)

#         # main loop
#         sim_gen = self.compute_traj_loop(desc1, inputs, outputs)

#         for current_problem, sim_result in sim_gen:
#             t = sim_result.t[-1]
#             x = sim_result.x[-1, :]

#             event_idx = np.argmin(np.abs(sim_result.e[-1, :]))

#             # trajectory-specific phase switching
#             if current_problem is desc1:
#                 if DEBUG:
#                     print("desc2")
#                 if event_idx != 1:
#                     if DEBUG:
#                         print("expected to hit speed trigger for desc1")
#                 sim_gen.send(desc2)
#             elif current_problem is desc2:
#                 if DEBUG:
#                     print("desc3")
#                 if event_idx != 0:
#                     if DEBUG:
#                         print("expected to hit alt trigger for desc2")
#                 sim_gen.send(desc3)
#             elif current_problem is desc3:
#                 if DEBUG:
#                     print("desc ending")
#                 if event_idx != 0:
#                     if DEBUG:
#                         print("expected to hit alt trigger for desc3")
#                 sim_gen.close()
#             else:
#                 if DEBUG:
#                     print("unexpected termination")
#                 sim_gen.close()
