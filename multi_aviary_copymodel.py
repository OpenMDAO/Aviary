"""
Goal: use single aircraft description but optimize it for multiple missions simultaneously,
i.e. all missions are on the range-payload line instead of having excess performance
Aircraft csv: defines plane, but also defines payload (passengers, cargo) which can vary with mission
    These will have to be specified in some alternate way such as a list correspond to mission #
Phase info: defines a particular mission, will have multiple phase infos
"""
import sys
import warnings
import aviary.api as av
import openmdao.api as om
import dymos as dm
import numpy as np
from aviary.variable_info.enums import ProblemType, AnalysisScheme
from c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info
from aviary.variable_info.variable_meta_data import _MetaData as MetaData
from aviary.variable_info.variables import Mission, Dynamic
from aviary.interface.methods_for_level2 import wrapped_convert_units
from aviary.variable_info.enums import EquationsOfMotion

TWO_DEGREES_OF_FREEDOM = EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY
SOLVED_2DOF = EquationsOfMotion.SOLVED_2DOF

# "comp?.a can be used to reference multiple comp1.a comp2.a etc"


class MultiMissionProblem(om.Problem):
    def __init__(self, planes, phase_infos, weights):
        super().__init__()
        self.num_missions = len(planes)
        # phase infos and planes length must match - this maybe unnecessary if
        # different planes (payloads) fly same mission (say pax vs cargo)
        # or if same payload flies 2 different missions (altitude/mach differences)
        if self.num_missions != len(phase_infos):
            raise Exception("Length of planes and phase_infos must be the same!")

        # if fewer weights than planes are provided, assign equal weights for all planes
        if len(weights) < self.num_missions:
            weights = [1]*self.num_missions
        # if more weights than planes, raise exception
        elif len(weights) > self.num_missions:
            raise Exception("Length of weights cannot exceed length of planes!")
        self.weights = weights

        self.group_prefix = 'group'
        self.probs = []
        # define individual aviary problems
        for i, (plane, phase_info) in enumerate(zip(planes, phase_infos)):
            prob = av.AviaryProblem()
            prob.load_inputs(plane, phase_info)
            prob.check_and_preprocess_inputs()
            prob.add_pre_mission_systems()
            prob.add_phases()
            prob.add_post_mission_systems()
            prob.link_phases()
            prob.add_design_variables()  # should not work at super prob level
            self.probs.append(prob)

            self.model.add_subsystem(
                self.group_prefix + f'_{i}', prob.model,
                promotes=['mission:design:gross_mass'])

    def add_design_variables(self):
        self.model.add_design_var('mission:design:gross_mass', lower=10., upper=900e3)

    def add_driver(self):
        self.driver = om.ScipyOptimizeDriver()
        self.driver.options['optimizer'] = 'SLSQP'
        # self.driver.declare_coloring()
        self.model.linear_solver = om.DirectSolver()

    def add_objective(self):
        weights = self.weights
        fuel_burned_vars = [f"fuel_{i}" for i in range(self.num_missions)]
        weighted_str = "+".join([f"{fuel}*{weight}"
                                for fuel, weight in zip(fuel_burned_vars, weights)])
        # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]

        # adding compound execComp to super problem
        self.model.add_subsystem('compound_fuel_burn_objective', om.ExecComp(
            "compound = "+weighted_str), promotes=["compound", *fuel_burned_vars])

        for i in range(self.num_missions):
            # connecting each subcomponent's fuel burn to super problem's unique fuel variables
            self.model.connect(
                self.group_prefix+f"_{i}.mission:summary:fuel_burned", f"fuel_{i}")
        self.model.add_objective('compound')

    def setup_wrapper(self):
        """Wrapper for om.Problem setup with warning ignoring and setting options"""
        for prob in self.probs:
            prob.model.options['aviary_options'] = prob.aviary_inputs
            prob.model.options['aviary_metadata'] = prob.meta_data
            prob.model.options['phase_info'] = prob.phase_info

        # Aviary's problem setup wrapper uses these ignored warnings to suppress
        # some warnings related to variable promotion. Replicating that here with
        # setup for the super problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", om.OpenMDAOWarning)
            warnings.simplefilter("ignore", om.PromotionWarning)
            self.setup()

    def create_N2(self, outfile='multi_aviary_copymodel.html'):
        om.n2(self, outfile=outfile)

    def run(self):
        self.run_model()
        self.check_totals(method='fd')
        # dm.run_problem(self)


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False
    planes = ['c5_maxpayload.csv', 'c5_intermediate.csv']
    phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
    weights = [1, 1]
    super_prob = MultiMissionProblem(planes, phase_infos, weights)
    super_prob.add_driver()
    super_prob.add_design_variables()
    super_prob.add_objective()
    super_prob.setup_wrapper()
    for i, prob in enumerate(super_prob.probs):
        super_prob.set_val(
            super_prob.group_prefix +
            f"_{i}.aircraft:design:landing_to_takeoff_mass_ratio", 0.5)
        prob.set_initial_guesses(super_prob, super_prob.group_prefix+f"_{i}.")
        print(super_prob.get_val(super_prob.group_prefix +
                                 f"_{i}.aircraft:design:landing_to_takeoff_mass_ratio"))
        print(super_prob.get_val(super_prob.group_prefix +
                                 f"_{i}.mission:summary:range"))
    if makeN2:
        super_prob.create_N2()
    super_prob.run()

    # def initvals(self):
    #     """attempting to copy over aviary code for setting initial values and changing references"""
    #     for i, prob in enumerate(self.probs):
    #         setvalprob.set_val(parent_prefix+self.group_prefix +
    #                      f'_{i}.aircraft:design:landing_to_takeoff_mass_ratio', 0.5)
    #         # Grab the trajectory object from the model
    #         traj = prob.model.traj

    #         # Determine which phases to loop over, fetching them from the trajectory
    #         phase_items = traj._phases.items()

    #         # Loop over each phase and set initial guesses for the state and control variables
    #         for idx, (phase_name, phase) in enumerate(phase_items):
    #             # If not, fetch the initial guesses specific to the phase
    #             # check if guesses exist for this phase
    #             if "initial_guesses" in prob.phase_info[phase_name]:
    #                 guesses = prob.phase_info[phase_name]['initial_guesses']
    #             else:
    #                 guesses = {}

    # # ||||||||||||| _add_subsystem_guesses
    #             # Get all subsystems associated with the phase
    #             all_subsystems = prob._get_all_subsystems(
    #                 prob.phase_info[phase_name]['external_subsystems'])

    #             # Loop over each subsystem
    #             for subsystem in all_subsystems:
    #                 # Fetch the initial guesses for the subsystem
    #                 initial_guesses = subsystem.get_initial_guesses()

    #                 # Loop over each guess
    #                 for key, val in initial_guesses.items():
    #                     # Identify the type of the guess (state or control)
    #                     type = val.pop('type')
    #                     if 'state' in type:
    #                         path_string = 'states'
    #                     elif 'control' in type:
    #                         path_string = 'controls'

    #                     # Process the guess variable (handles array interpolation)
    #                     val['val'] = prob._process_guess_var(val['val'], key, phase)

    #                     # Set the initial guess in the problem
    #                     setvalprob.set_val(parent_prefix+
    #                         self.group_prefix +
    #                         f'_{i}.traj.{phase_name}.{path_string}:{key}', **val)

    #             # Set initial guesses for states and controls for each phase

    # # |||||||||||||||||||||| _add_guesses
    #             # If using the GASP model, set initial guesses for the rotation mass and flight duration
    #             if prob.mission_method in (HEIGHT_ENERGY, SOLVED_2DOF):
    #                 control_keys = ["mach", "altitude"]
    #                 state_keys = ["mass", Dynamic.Mission.DISTANCE]
    #             prob_keys = ["tau_gear", "tau_flaps"]

    #             # for the simple mission method, use the provided initial and final mach and altitude values from phase_info
    #             if prob.mission_method in (HEIGHT_ENERGY, SOLVED_2DOF):
    #                 initial_altitude = wrapped_convert_units(
    #                     prob.phase_info[phase_name]['user_options']
    #                     ['initial_altitude'],
    #                     'ft')
    #                 final_altitude = wrapped_convert_units(
    #                     prob.phase_info[phase_name]['user_options']['final_altitude'], 'ft')
    #                 initial_mach = prob.phase_info[phase_name]['user_options'][
    #                     'initial_mach']
    #                 final_mach = prob.phase_info[phase_name]['user_options'][
    #                     'final_mach']

    #                 guesses["mach"] = ([initial_mach[0], final_mach[0]], "unitless")
    #                 guesses["altitude"] = ([initial_altitude, final_altitude], 'ft')

    #             if prob.mission_method is HEIGHT_ENERGY:
    #                 # if time not in initial guesses, set it to the average of the initial_bounds and the duration_bounds
    #                 if 'time' not in guesses:
    #                     initial_bounds = wrapped_convert_units(
    #                         prob.phase_info[phase_name]['user_options']['initial_bounds'], 's')
    #                     duration_bounds = wrapped_convert_units(
    #                         prob.phase_info[phase_name]['user_options']['duration_bounds'], 's')
    #                     guesses["time"] = ([np.mean(initial_bounds[0]), np.mean(
    #                         duration_bounds[0])], 's')

    #                 # if time not in initial guesses, set it to the average of the initial_bounds and the duration_bounds
    #                 if 'time' not in guesses:
    #                     initial_bounds = prob.phase_info[phase_name]['user_options'][
    #                         'initial_bounds']
    #                     duration_bounds = prob.phase_info[phase_name]['user_options'][
    #                         'duration_bounds']
    #                     # Add a check for the initial and duration bounds, raise an error if they are not consistent
    #                     if initial_bounds[1] != duration_bounds[1]:
    #                         raise ValueError(
    #                             f"Initial and duration bounds for {phase_name} are not consistent.")
    #                     guesses["time"] = ([np.mean(initial_bounds[0]), np.mean(
    #                         duration_bounds[0])], initial_bounds[1])

    #             for guess_key, guess_data in guesses.items():
    #                 val, units = guess_data

    #                 # Set initial guess for time variables
    #                 if 'time' == guess_key and prob.mission_method is not SOLVED_2DOF:
    #                     setvalprob.set_val(parent_prefix+
    #                         self.group_prefix + f'_{i}.traj.{phase_name}.t_initial',
    #                         val[0],
    #                         units=units)
    #                     setvalprob.set_val(parent_prefix+
    #                         self.group_prefix + f'_{i}.traj.{phase_name}.t_duration',
    #                         val[1],
    #                         units=units)

    #                 else:
    #                     # Set initial guess for control variables
    #                     if guess_key in control_keys:
    #                         try:
    #                             setvalprob.set_val(parent_prefix+self.group_prefix +
    #                                          f'_{i}.traj.{phase_name}.controls:{guess_key}',
    #                                          prob._process_guess_var(
    #                                              val, guess_key, phase),
    #                                          units=units)
    #                         except KeyError:
    #                             try:
    #                                 setvalprob.set_val(parent_prefix+
    #                                     self.group_prefix +
    #                                     f'_{i}.traj.{phase_name}.polynomial_controls:{guess_key}',
    #                                     prob._process_guess_var(val, guess_key, phase),
    #                                     units=units)
    #                             except KeyError:
    #                                 setvalprob.set_val(parent_prefix+
    #                                     self.group_prefix +
    #                                     f'_{i}.traj.{phase_name}.bspline_controls:{guess_key}',
    #                                     prob._process_guess_var(val, guess_key, phase),
    #                                     units=units)

    #                     if guess_key in control_keys:
    #                         pass
    #                     # Set initial guess for state variables
    #                     elif guess_key in state_keys:
    #                         setvalprob.set_val(parent_prefix+self.group_prefix +
    #                                      f'_{i}.traj.{phase_name}.states:{guess_key}',
    #                                      prob._process_guess_var(
    #                                          val, guess_key, phase),
    #                                      units=units)
    #                     elif guess_key in prob_keys:
    #                         setvalprob.set_val(parent_prefix+
    #                             self.group_prefix+f'_{i}.'+guess_key, val, units=units)
    #                     elif ":" in guess_key:
    #                         setvalprob.set_val(parent_prefix+
    #                             self.group_prefix + f'_{i}.traj.{phase_name}.{guess_key}',
    #                             prob._process_guess_var(val, guess_key, phase),
    #                             units=units)
    #                     else:
    #                         # raise error if the guess key is not recognized
    #                         raise ValueError(
    #                             f"Initial guess key {guess_key} in {phase_name} is not recognized.")

    #             # We need some special logic for these following variables because GASP computes
    #             # initial guesses using some knowledge of the mission duration and other variables
    #             # that are only available after calling `create_vehicle`. Thus these initial guess
    #             # values are not included in the `phase_info` object.

    #             base_phase = phase_name
    #             if 'mass' not in guesses:
    #                 mass_guess = prob.aviary_inputs.get_val(
    #                     Mission.Design.GROSS_MASS, units='lbm')
    #                 # Set the mass guess as the initial value for the mass state variable
    #                 setvalprob.set_val(parent_prefix+self.group_prefix+f'_{i}.traj.{phase_name}.states:mass',
    #                              mass_guess, units='lbm')

    #             # if 'time' not in guesses:
    #             #     # Determine initial time and duration guesses depending on the phase name
    #             #     if 'desc1' == base_phase:
    #             #         t_initial = flight_duration*.9
    #             #         t_duration = flight_duration*.04
    #             #     elif 'desc2' in base_phase:
    #             #         t_initial = flight_duration*.94
    #             #         t_duration = 5000
    #             #     # Set the time guesses as the initial values for the time-related trajectory variables
    #             #     setvalprob.set_val(parent_prefix+f"traj.{phase_name}.t_initial",
    #             #                  t_initial, units='s')
    #             #     setvalprob.set_val(parent_prefix+f"traj.{phase_name}.t_duration",
    #             #                  t_duration, units='s')


# ========================================================================= old code
    # super_prob = om.Problem()
    # num_missions = len(weights)
    # probs = []
    # prefix = "problem_"

    # makeN2 = False
    # if len(sys.argv) > 1:
    #     if "n2" in sys.argv:
    #         makeN2 = True

    # # define individual aviary problems
    # for i, (plane, phase_info) in enumerate(zip(planes, phase_infos)):
    #     prob = av.AviaryProblem()
    #     prob.load_inputs(plane, phase_info)
    #     prob.check_and_preprocess_inputs()
    #     prob.add_pre_mission_systems()
    #     traj = prob.add_phases()  # save dymos traj to add to super problem as a subsystem
    #     prob.add_post_mission_systems()
    #     prob.link_phases()  # this is half working / connect statements from outside of traj to inside are failing
    #     prob.problem_type = ProblemType.ALTERNATE  # adds summary gross mass as design var
    #     prob.add_design_variables()
    #     probs.append(prob)

    #     group = om.Group()  # this group will contain all the promoted aviary vars
    #     group.add_subsystem("pre", prob.pre_mission)
    #     group.add_subsystem("traj", traj)
    #     group.add_subsystem("post", prob.post_mission)

    #     # setting defaults for these variables to suppress errors
    #     longlst = [
    #         'mission:summary:gross_mass', 'aircraft:wing:sweep',
    #         'aircraft:wing:thickness_to_chord', 'aircraft:wing:area',
    #         'aircraft:wing:taper_ratio', 'mission:design:gross_mass']
    #     for var in longlst:
    #         group.set_input_defaults(
    #             var, val=MetaData[var]['default_value'],
    #             units=MetaData[var]['units'])

    #     # add group and promote design gross mass (common input amongst multiple missions)
    #     # in this way it represents the MTOW
    #     super_prob.model.add_subsystem(prefix+f'{i}', group, promotes=[
    #                                    'mission:design:gross_mass'])

    # # add design gross mass as a design var
    # super_prob.model.add_design_var(
    #     'mission:design:gross_mass', lower=100e3, upper=1000e3)

    # for i in range(num_missions):
    #     # connecting each subcomponent's fuel burn to super problem's unique fuel variables
    #     super_prob.model.connect(
    #         prefix+f"{i}.mission:summary:fuel_burned", f"fuel_{i}")

    #     # create constraint to force each mission's summary gross mass to not
    #     # exceed the common mission design gross mass (aka MTOW)
    #     super_prob.model.add_subsystem(f'MTOW_constraint{i}', om.ExecComp(
    #         'mtow_resid = design_gross_mass - summary_gross_mass'),
    #         promotes=[('summary_gross_mass', prefix+f'{i}.mission:summary:gross_mass'),
    #                   ('design_gross_mass', 'mission:design:gross_mass')])

    #     super_prob.model.add_constraint(f'MTOW_constraint{i}.mtow_resid', lower=0.)

    # # creating variable strings that will represent fuel burn from each mission
    # fuel_burned_vars = [f"fuel_{i}" for i in range(num_missions)]
    # weighted_str = "+".join([f"{fuel}*{weight}"
    #                          for fuel, weight in zip(fuel_burned_vars, weights)])
    # # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]

    # # adding compound execComp to super problem
    # super_prob.model.add_subsystem('compound_fuel_burn_objective', om.ExecComp(
    #     "compound = "+weighted_str), promotes=["compound", *fuel_burned_vars])

    # super_prob.driver = om.ScipyOptimizeDriver()
    # super_prob.driver.options['optimizer'] = 'SLSQP'
    # super_prob.model.add_objective('compound')  # output from execcomp goes here

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", om.OpenMDAOWarning)
    #     warnings.simplefilter("ignore", om.PromotionWarning)
    #     super_prob.setup()

    # if makeN2:
    #     om.n2(super_prob, outfile="multi_mission_importTraj_N2.html")  # create N2 diagram

    # # cannot use this b/c initial guesses (i.e. setval func) has to be called on super prob level
    # # for prob in probs:
    # #     # prob.setup()
    # #     prob.set_initial_guesses()

    # # dm.run_problem(super_prob)


"""
Ferry mission phase info:
Times (min):   0,    50,   812, 843
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 7001 nmi
Notes: 32k in 30 mins too fast for aviary, climb to low alt then slow rise through cruise

Intermediate mission phase info:
Times (min):   0,    50,   560, 590
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 4839 nmi

Max Payload mission phase info:
Times (min):   0,    50,   260, 290
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 2272 nmi

Hard to find multiple payload/range values for FwFm (737), so use C-5 instead
Based on:
    https://en.wikipedia.org/wiki/Lockheed_C-5_Galaxy#Specifications_(C-5M),
    https://www.af.mil/About-Us/Fact-Sheets/Display/Article/1529718/c-5-abc-galaxy-and-c-5m-super-galaxy/

MTOW: 840,000 lb
Max Payload: 281,000 lb
Max Fuel: 341,446 lb
Empty Weight: 380,000 lb -> leaves 460,000 lb for fuel+payload (max fuel + max payload = 622,446 lb)

Payload/range:
    281,000 lb payload -> 2,150 nmi range (AF.mil) [max payload case]
    120,000 lb payload -> 4,800 nmi range (AF.mil) [intermediate case]
          0 lb payload -> 7,000 nmi range (AF.mil) [ferry case]

Flight characteristics:
    Cruise at M0.77 at 33k ft
    Max rate of climb: 2100 ft/min
"""
