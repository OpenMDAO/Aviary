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
from c5_models.c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_models.c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_models.c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info
from easy_phase_info_inter import phase_info as easy_inter
from easy_phase_info_max import phase_info as easy_max
from aviary.variable_info.variables import Mission, Aircraft
from aviary.variable_info.enums import ProblemType

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
            prob.problem_type = ProblemType.ALTERNATE
            prob.add_design_variables()  # should not work at super prob level
            self.probs.append(prob)

            self.model.add_subsystem(
                self.group_prefix + f'_{i}', prob.model,
                promotes=[Mission.Design.GROSS_MASS])

    def add_design_variables(self):
        self.model.add_design_var('mission:design:gross_mass', lower=10., upper=900e3)

    def add_driver(self):
        # pyoptsparse SLSQP errors out w pos directional derivative line search (obj scaler = 1) and
        # inequality constraints incompatible (obj scaler = -1) - fuel burn obj
        # pyoptsparse IPOPT keeps iterating (seen upto 1000+ iters) in the IPOPT.out file but no result
        # scipy SLSQP reaches iter limit and fails optimization
        self.driver = om.pyOptSparseDriver()
        self.driver.options['optimizer'] = 'SLSQP'
        # self.driver.options['maxiter'] = 1e3
        # self.driver.declare_coloring()
        # self.model.linear_solver = om.DirectSolver()
        """scipy SLSQP results
            Iteration limit reached    (Exit mode 9)
            Current function value: -43.71865402878029
            Iterations: 200
            Function evaluations: 1018
            Gradient evaluations: 200
            Optimization FAILED.
            Iteration limit reached"""

    def add_objective(self):
        weights = [float(weight/sum(self.weights)) for weight in self.weights]
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
                self.group_prefix+f"_{i}.{Mission.Objectives.FUEL}", f"fuel_{i}")
        self.model.add_objective('compound', ref=1e4)

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
            self.setup(check='all')

    def run(self):
        # self.run_model()
        # self.check_totals(method='fd', compact_print=True)
        self.model.set_solver_print(0)

        # self.run_driver()
        dm.run_problem(self, make_plots=True)


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False
    planes = ['c5_models/c5_maxpayload.csv', 'c5_models/c5_intermediate.csv']
    # phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
    phase_infos = [easy_max, easy_inter]
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
    # super_prob.final_setup()
    if makeN2:
        from createN2 import createN2
        createN2(__file__, super_prob)
    super_prob.run()

    outputs = {Mission.Summary.FUEL_BURNED: [],
               Aircraft.Design.EMPTY_MASS: []}

    print("\n\n=========================\n")
    for key in outputs.keys():
        val1 = super_prob.get_val(f'group_0.{key}')[0]
        val2 = super_prob.get_val(f'group_1.{key}')[0]
        print(f"Variable: {key}")
        print(f"Values: {val1}, {val2}")

"""
Variable: mission:summary:fuel_burned
Values: 13730.910584707046, 15740.545749454643
Variable: aircraft:design:empty_mass
Values: 336859.7179064408, 337047.85745526763
"""

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


"""
input disconnected error:
WARNING: The following inputs are not connected:
  group_0.aircraft:air_conditioning:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.AC.aircraft:air_conditioning:mass_scaler)
  group_0.aircraft:anti_icing:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.anti_icing.aircraft:anti_icing:mass_scaler)
  group_0.aircraft:apu:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.apu.aircraft:apu:mass_scaler)
  group_0.aircraft:avionics:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.avionics.aircraft:avionics:mass_scaler)
  group_0.aircraft:canard:area (group_0.pre_mission.core_subsystems.core_geometry.canard.aircraft:canard:area)
  group_0.aircraft:canard:area (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:canard:area)
  group_0.aircraft:canard:area (group_0.pre_mission.core_subsystems.core_mass.canard.aircraft:canard:area)
  group_0.aircraft:canard:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:canard:aspect_ratio)
  group_0.aircraft:canard:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.canard.aircraft:canard:mass_scaler)
  group_0.aircraft:canard:taper_ratio (group_0.pre_mission.core_subsystems.core_mass.canard.aircraft:canard:taper_ratio)
  group_0.aircraft:canard:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.canard.aircraft:canard:thickness_to_chord)
  group_0.aircraft:canard:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:canard:thickness_to_chord)
  group_0.aircraft:canard:wetted_area_scaler (group_0.pre_mission.core_subsystems.core_geometry.canard.aircraft:canard:wetted_area_scaler)
  group_0.aircraft:crew_and_payload:cargo_container_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.cargo_containers.aircraft:crew_and_payload:cargo_container_mass_scaler)
  group_0.aircraft:crew_and_payload:cargo_mass (group_0.pre_mission.core_subsystems.core_mass.cargo_containers.aircraft:crew_and_payload:cargo_mass)
  group_0.aircraft:crew_and_payload:cargo_mass (group_0.pre_mission.core_subsystems.core_mass.total_mass.zero_fuel_mass.aircraft:crew_and_payload:cargo_mass)
  group_0.aircraft:crew_and_payload:flight_crew_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.flight_crew.aircraft:crew_and_payload:flight_crew_mass_scaler)
  group_0.aircraft:crew_and_payload:misc_cargo (group_0.pre_mission.core_subsystems.core_mass.cargo.aircraft:crew_and_payload:misc_cargo)
  group_0.aircraft:crew_and_payload:non_flight_crew_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.nonflight_crew.aircraft:crew_and_payload:non_flight_crew_mass_scaler)
  group_0.aircraft:crew_and_payload:passenger_service_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.pass_service.aircraft:crew_and_payload:passenger_service_mass_scaler)
  group_0.aircraft:crew_and_payload:wing_cargo (group_0.pre_mission.core_subsystems.core_mass.cargo.aircraft:crew_and_payload:wing_cargo)
  group_0.aircraft:design:base_area (group_0.traj.param_comp.parameters:aircraft:design:base_area)
  group_0.aircraft:design:empty_mass_margin_scaler (group_0.pre_mission.core_subsystems.core_mass.total_mass.empty_mass_margin.aircraft:design:empty_mass_margin_scaler)
  group_0.aircraft:design:external_subsystems_mass (group_0.pre_mission.core_subsystems.core_mass.total_mass.system_equip_mass.aircraft:design:external_subsystems_mass)
  group_0.aircraft:design:lift_dependent_drag_coeff_factor (group_0.traj.param_comp.parameters:aircraft:design:lift_dependent_drag_coeff_factor)
  group_0.aircraft:design:reserve_fuel_additional (group_0.post_mission.reserve_fuel.reserve_fuel_additional)
  group_0.aircraft:design:subsonic_drag_coeff_factor (group_0.traj.param_comp.parameters:aircraft:design:subsonic_drag_coeff_factor)
  group_0.aircraft:design:supersonic_drag_coeff_factor (group_0.traj.param_comp.parameters:aircraft:design:supersonic_drag_coeff_factor)
  group_0.aircraft:design:zero_lift_drag_coeff_factor (group_0.traj.param_comp.parameters:aircraft:design:zero_lift_drag_coeff_factor)
  group_0.aircraft:electrical:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.electrical.aircraft:electrical:mass_scaler)
  group_0.aircraft:engine:mass (group_0.pre_mission.core_subsystems.core_mass.wing_group.engine_pod_mass.aircraft:engine:mass)
  group_0.aircraft:engine:scaled_sls_thrust (group_0.pre_mission.core_propulsion.CF6.aircraft:engine:scaled_sls_thrust)
  group_0.aircraft:engine:scaled_sls_thrust (group_0.pre_mission.core_propulsion.propulsion_sum.aircraft:engine:scaled_sls_thrust)
  group_0.aircraft:engine:scaled_sls_thrust (group_0.pre_mission.core_subsystems.core_mass.engine_mass.aircraft:engine:scaled_sls_thrust)
  group_0.aircraft:engine:scaled_sls_thrust (group_0.pre_mission.core_subsystems.core_mass.nacelle.aircraft:engine:scaled_sls_thrust)
  group_0.aircraft:engine:scaled_sls_thrust (group_0.pre_mission.core_subsystems.core_mass.thrust_rev.aircraft:engine:scaled_sls_thrust)
  group_0.aircraft:engine:scaled_sls_thrust (group_0.pre_mission.core_subsystems.core_mass.wing_group.engine_pod_mass.aircraft:engine:scaled_sls_thrust)
  group_0.aircraft:engine:thrust_reversers_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.thrust_rev.aircraft:engine:thrust_reversers_mass_scaler)
  group_0.aircraft:engine:wing_locations (group_0.pre_mission.core_subsystems.core_mass.landing_group.main_landing_gear_length.aircraft:engine:wing_locations)
  group_0.aircraft:engine:wing_locations (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending_factor.aircraft:engine:wing_locations)
  group_0.aircraft:fins:area (group_0.pre_mission.core_subsystems.core_mass.fin.aircraft:fins:area)
  group_0.aircraft:fins:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.fin.aircraft:fins:mass_scaler)
  group_0.aircraft:fins:taper_ratio (group_0.pre_mission.core_subsystems.core_mass.fin.aircraft:fins:taper_ratio)
  group_0.aircraft:fuel:auxiliary_fuel_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.total_fuel_capacity.aircraft:fuel:auxiliary_fuel_capacity)
  group_0.aircraft:fuel:capacity_factor (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:fuel:capacity_factor)
  group_0.aircraft:fuel:density_ratio (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:fuel:density_ratio)
  group_0.aircraft:fuel:density_ratio (group_0.pre_mission.core_subsystems.core_mass.unusable_fuel.aircraft:fuel:density_ratio)
  group_0.aircraft:fuel:fuel_margin (group_0.post_mission.fuel_calc.fuel_margin)
  group_0.aircraft:fuel:fuel_system_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.fuel_system.aircraft:fuel:fuel_system_mass_scaler)
  group_0.aircraft:fuel:fuselage_fuel_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.auxiliary_fuel_capacity.aircraft:fuel:fuselage_fuel_capacity)
  group_0.aircraft:fuel:fuselage_fuel_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.total_fuel_capacity.aircraft:fuel:fuselage_fuel_capacity)
  group_0.aircraft:fuel:total_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.auxiliary_fuel_capacity.aircraft:fuel:total_capacity)
  group_0.aircraft:fuel:total_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.fuselage_fuel_capacity.aircraft:fuel:total_capacity)
  group_0.aircraft:fuel:total_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_system.aircraft:fuel:total_capacity)
  group_0.aircraft:fuel:total_capacity (group_0.pre_mission.core_subsystems.core_mass.unusable_fuel.aircraft:fuel:total_capacity)
  group_0.aircraft:fuel:unusable_fuel_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.unusable_fuel.aircraft:fuel:unusable_fuel_mass_scaler)
  group_0.aircraft:fuel:wing_ref_capacity (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:fuel:wing_ref_capacity)
  group_0.aircraft:fuel:wing_ref_capacity_area (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:fuel:wing_ref_capacity_area)
  group_0.aircraft:fuel:wing_ref_capacity_term_A (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:fuel:wing_ref_capacity_term_A)
  group_0.aircraft:fuel:wing_ref_capacity_term_B (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:fuel:wing_ref_capacity_term_B)
  group_0.aircraft:furnishings:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.furnishings.aircraft:furnishings:mass_scaler)
  group_0.aircraft:fuselage:laminar_flow_lower (group_0.traj.param_comp.parameters:aircraft:fuselage:laminar_flow_lower)
  group_0.aircraft:fuselage:laminar_flow_upper (group_0.traj.param_comp.parameters:aircraft:fuselage:laminar_flow_upper)
  group_0.aircraft:fuselage:length (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:fuselage:length)
  group_0.aircraft:fuselage:length (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:fuselage:length)
  group_0.aircraft:fuselage:length (group_0.pre_mission.core_subsystems.core_geometry.fuselage_prelim.aircraft:fuselage:length)
  group_0.aircraft:fuselage:length (group_0.pre_mission.core_subsystems.core_mass.electrical.aircraft:fuselage:length)
  group_0.aircraft:fuselage:length (group_0.pre_mission.core_subsystems.core_mass.fuselage.aircraft:fuselage:length)
  group_0.aircraft:fuselage:length (group_0.pre_mission.core_subsystems.core_mass.landing_group.main_landing_gear_length.aircraft:fuselage:length)
  group_0.aircraft:fuselage:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.fuselage.aircraft:fuselage:mass_scaler)
  group_0.aircraft:fuselage:max_height (group_0.pre_mission.core_subsystems.core_geometry.fuselage_prelim.aircraft:fuselage:max_height)
  group_0.aircraft:fuselage:max_height (group_0.pre_mission.core_subsystems.core_mass.AC.aircraft:fuselage:max_height)
  group_0.aircraft:fuselage:max_height (group_0.pre_mission.core_subsystems.core_mass.furnishings.aircraft:fuselage:max_height)
  group_0.aircraft:fuselage:max_width (group_0.pre_mission.core_subsystems.core_geometry.fuselage_prelim.aircraft:fuselage:max_width)
  group_0.aircraft:fuselage:max_width (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:fuselage:max_width)
  group_0.aircraft:fuselage:max_width (group_0.pre_mission.core_subsystems.core_mass.anti_icing.aircraft:fuselage:max_width)
  group_0.aircraft:fuselage:max_width (group_0.pre_mission.core_subsystems.core_mass.electrical.aircraft:fuselage:max_width)
  group_0.aircraft:fuselage:max_width (group_0.pre_mission.core_subsystems.core_mass.furnishings.aircraft:fuselage:max_width)
  group_0.aircraft:fuselage:max_width (group_0.pre_mission.core_subsystems.core_mass.landing_group.main_landing_gear_length.aircraft:fuselage:max_width)
  group_0.aircraft:fuselage:passenger_compartment_length (group_0.pre_mission.core_subsystems.core_mass.furnishings.aircraft:fuselage:passenger_compartment_length)
  group_0.aircraft:fuselage:planform_area (group_0.pre_mission.core_subsystems.core_mass.AC.aircraft:fuselage:planform_area)
  group_0.aircraft:fuselage:planform_area (group_0.pre_mission.core_subsystems.core_mass.apu.aircraft:fuselage:planform_area)
  group_0.aircraft:fuselage:planform_area (group_0.pre_mission.core_subsystems.core_mass.avionics.aircraft:fuselage:planform_area)
  group_0.aircraft:fuselage:planform_area (group_0.pre_mission.core_subsystems.core_mass.hydraulics.aircraft:fuselage:planform_area)
  group_0.aircraft:fuselage:planform_area (group_0.pre_mission.core_subsystems.core_mass.instruments.aircraft:fuselage:planform_area)
  group_0.aircraft:fuselage:wetted_area_scaler (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:fuselage:wetted_area_scaler)
  group_0.aircraft:horizontal_tail:area (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:horizontal_tail:area)
  group_0.aircraft:horizontal_tail:area (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:horizontal_tail:area)
  group_0.aircraft:horizontal_tail:area (group_0.pre_mission.core_subsystems.core_geometry.tail.aircraft:horizontal_tail:area)
  group_0.aircraft:horizontal_tail:area (group_0.pre_mission.core_subsystems.core_mass.htail.aircraft:horizontal_tail:area)
  group_0.aircraft:horizontal_tail:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:horizontal_tail:aspect_ratio)
  group_0.aircraft:horizontal_tail:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:horizontal_tail:aspect_ratio)
  group_0.aircraft:horizontal_tail:laminar_flow_lower (group_0.traj.param_comp.parameters:aircraft:horizontal_tail:laminar_flow_lower)
  group_0.aircraft:horizontal_tail:laminar_flow_upper (group_0.traj.param_comp.parameters:aircraft:horizontal_tail:laminar_flow_upper)
  group_0.aircraft:horizontal_tail:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.htail.aircraft:horizontal_tail:mass_scaler)
  group_0.aircraft:horizontal_tail:taper_ratio (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:horizontal_tail:taper_ratio)
  group_0.aircraft:horizontal_tail:taper_ratio (group_0.pre_mission.core_subsystems.core_mass.htail.aircraft:horizontal_tail:taper_ratio)
  group_0.aircraft:horizontal_tail:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:horizontal_tail:thickness_to_chord)
  group_0.aircraft:horizontal_tail:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:horizontal_tail:thickness_to_chord)
  group_0.aircraft:horizontal_tail:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:horizontal_tail:thickness_to_chord)
  group_0.aircraft:horizontal_tail:vertical_tail_fraction (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:horizontal_tail:vertical_tail_fraction)
  group_0.aircraft:horizontal_tail:vertical_tail_fraction (group_0.pre_mission.core_subsystems.core_geometry.tail.aircraft:horizontal_tail:vertical_tail_fraction)
  group_0.aircraft:horizontal_tail:wetted_area_scaler (group_0.pre_mission.core_subsystems.core_geometry.tail.aircraft:horizontal_tail:wetted_area_scaler)
  group_0.aircraft:hydraulics:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.hydraulics.aircraft:hydraulics:mass_scaler)
  group_0.aircraft:hydraulics:system_pressure (group_0.pre_mission.core_subsystems.core_mass.hydraulics.aircraft:hydraulics:system_pressure)
  group_0.aircraft:instruments:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.instruments.aircraft:instruments:mass_scaler)
  group_0.aircraft:landing_gear:main_gear_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.landing_group.landing_gear.aircraft:landing_gear:main_gear_mass_scaler)
  group_0.aircraft:landing_gear:nose_gear_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.landing_group.landing_gear.aircraft:landing_gear:nose_gear_mass_scaler)
  group_0.aircraft:nacelle:avg_diameter (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:nacelle:avg_diameter)
  group_0.aircraft:nacelle:avg_diameter (group_0.pre_mission.core_subsystems.core_geometry.nacelles.aircraft:nacelle:avg_diameter)
  group_0.aircraft:nacelle:avg_diameter (group_0.pre_mission.core_subsystems.core_mass.anti_icing.aircraft:nacelle:avg_diameter)
  group_0.aircraft:nacelle:avg_diameter (group_0.pre_mission.core_subsystems.core_mass.landing_group.main_landing_gear_length.aircraft:nacelle:avg_diameter)
  group_0.aircraft:nacelle:avg_diameter (group_0.pre_mission.core_subsystems.core_mass.nacelle.aircraft:nacelle:avg_diameter)
  group_0.aircraft:nacelle:avg_diameter (group_0.pre_mission.core_subsystems.core_mass.starter.aircraft:nacelle:avg_diameter)
  group_0.aircraft:nacelle:avg_length (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:nacelle:avg_length)
  group_0.aircraft:nacelle:avg_length (group_0.pre_mission.core_subsystems.core_geometry.nacelles.aircraft:nacelle:avg_length)
  group_0.aircraft:nacelle:avg_length (group_0.pre_mission.core_subsystems.core_mass.nacelle.aircraft:nacelle:avg_length)
  group_0.aircraft:nacelle:laminar_flow_lower (group_0.traj.param_comp.parameters:aircraft:nacelle:laminar_flow_lower)
  group_0.aircraft:nacelle:laminar_flow_upper (group_0.traj.param_comp.parameters:aircraft:nacelle:laminar_flow_upper)
  group_0.aircraft:nacelle:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.nacelle.aircraft:nacelle:mass_scaler)
  group_0.aircraft:nacelle:wetted_area_scaler (group_0.pre_mission.core_subsystems.core_geometry.nacelles.aircraft:nacelle:wetted_area_scaler)
  group_0.aircraft:paint:mass_per_unit_area (group_0.pre_mission.core_subsystems.core_mass.paint.aircraft:paint:mass_per_unit_area)
  group_0.aircraft:propulsion:engine_oil_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.engine_oil.aircraft:propulsion:engine_oil_mass_scaler)
  group_0.aircraft:propulsion:misc_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.misc_engine.aircraft:propulsion:misc_mass_scaler)
  group_0.aircraft:vertical_tail:area (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:vertical_tail:area)
  group_0.aircraft:vertical_tail:area (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:vertical_tail:area)
  group_0.aircraft:vertical_tail:area (group_0.pre_mission.core_subsystems.core_geometry.tail.aircraft:vertical_tail:area)
  group_0.aircraft:vertical_tail:area (group_0.pre_mission.core_subsystems.core_mass.vert_tail.aircraft:vertical_tail:area)
  group_0.aircraft:vertical_tail:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:vertical_tail:aspect_ratio)
  group_0.aircraft:vertical_tail:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:vertical_tail:aspect_ratio)
  group_0.aircraft:vertical_tail:laminar_flow_lower (group_0.traj.param_comp.parameters:aircraft:vertical_tail:laminar_flow_lower)
  group_0.aircraft:vertical_tail:laminar_flow_upper (group_0.traj.param_comp.parameters:aircraft:vertical_tail:laminar_flow_upper)
  group_0.aircraft:vertical_tail:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.vert_tail.aircraft:vertical_tail:mass_scaler)
  group_0.aircraft:vertical_tail:taper_ratio (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:vertical_tail:taper_ratio)
  group_0.aircraft:vertical_tail:taper_ratio (group_0.pre_mission.core_subsystems.core_mass.vert_tail.aircraft:vertical_tail:taper_ratio)
  group_0.aircraft:vertical_tail:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:vertical_tail:thickness_to_chord)
  group_0.aircraft:vertical_tail:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:vertical_tail:thickness_to_chord)
  group_0.aircraft:vertical_tail:thickness_to_chord (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:vertical_tail:thickness_to_chord)
  group_0.aircraft:vertical_tail:wetted_area_scaler (group_0.pre_mission.core_subsystems.core_geometry.tail.aircraft:vertical_tail:wetted_area_scaler)
  group_0.aircraft:wing:aeroelastic_tailoring_factor (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:aeroelastic_tailoring_factor)
  group_0.aircraft:wing:aeroelastic_tailoring_factor (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending_factor.aircraft:wing:aeroelastic_tailoring_factor)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_geometry.wing.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_geometry.wing_prelim.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_mass.hydraulics.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_mass.surf_ctrl.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_mass.unusable_fuel.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_misc.aircraft:wing:area)
  group_0.aircraft:wing:area (group_0.traj.param_comp.parameters:aircraft:wing:area)
  group_0.aircraft:wing:aspect_ratio (group_0.pre_mission.core_subsystems.core_aerodynamics.design.aircraft:wing:aspect_ratio)
  group_0.aircraft:wing:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:wing:aspect_ratio)
  group_0.aircraft:wing:aspect_ratio (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:wing:aspect_ratio)
  group_0.aircraft:wing:aspect_ratio (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending_factor.aircraft:wing:aspect_ratio)
  group_0.aircraft:wing:aspect_ratio (group_0.traj.param_comp.parameters:aircraft:wing:aspect_ratio)
  group_0.aircraft:wing:aspect_ratio_reference (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending_factor.aircraft:wing:aspect_ratio_reference)
  group_0.aircraft:wing:bending_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:bending_mass_scaler)
  group_0.aircraft:wing:bwb_aft_body_mass (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_total.aircraft:wing:bwb_aft_body_mass)
  group_0.aircraft:wing:chord_per_semispan (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending_factor.aircraft:wing:chord_per_semispan)
  group_0.aircraft:wing:composite_fraction (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:composite_fraction)
  group_0.aircraft:wing:composite_fraction (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_misc.aircraft:wing:composite_fraction)
  group_0.aircraft:wing:composite_fraction (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_shear_control.aircraft:wing:composite_fraction)
  group_0.aircraft:wing:control_surface_area (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_shear_control.aircraft:wing:control_surface_area)
  group_0.aircraft:wing:control_surface_area_ratio (group_0.pre_mission.core_subsystems.core_mass.surf_ctrl.aircraft:wing:control_surface_area_ratio)
  group_0.aircraft:wing:dihedral (group_0.pre_mission.core_subsystems.core_mass.landing_group.main_landing_gear_length.aircraft:wing:dihedral)
  group_0.aircraft:wing:glove_and_bat (group_0.pre_mission.core_subsystems.core_geometry.characteristic_lengths.aircraft:wing:glove_and_bat)
  group_0.aircraft:wing:glove_and_bat (group_0.pre_mission.core_subsystems.core_geometry.fuselage.aircraft:wing:glove_and_bat)
  group_0.aircraft:wing:glove_and_bat (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:wing:glove_and_bat)
  group_0.aircraft:wing:glove_and_bat (group_0.pre_mission.core_subsystems.core_geometry.wing_prelim.aircraft:wing:glove_and_bat)
  group_0.aircraft:wing:incidence (group_0.traj.param_comp.parameters:aircraft:wing:incidence)
  group_0.aircraft:wing:laminar_flow_lower (group_0.traj.param_comp.parameters:aircraft:wing:laminar_flow_lower)
  group_0.aircraft:wing:laminar_flow_upper (group_0.traj.param_comp.parameters:aircraft:wing:laminar_flow_upper)
  group_0.aircraft:wing:load_fraction (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:load_fraction)
  group_0.aircraft:wing:load_path_sweep_dist (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending_factor.aircraft:wing:load_path_sweep_dist)
  group_0.aircraft:wing:mass_scaler (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_total.aircraft:wing:mass_scaler)
  group_0.aircraft:wing:max_camber_at_70_semispan (group_0.pre_mission.core_subsystems.core_aerodynamics.design.aircraft:wing:max_camber_at_70_semispan)
  group_0.aircraft:wing:max_camber_at_70_semispan (group_0.traj.param_comp.parameters:aircraft:wing:max_camber_at_70_semispan)
  group_0.aircraft:wing:misc_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:misc_mass_scaler)
  group_0.aircraft:wing:misc_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_misc.aircraft:wing:misc_mass_scaler)
  group_0.aircraft:wing:shear_control_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:shear_control_mass_scaler)
  group_0.aircraft:wing:shear_control_mass_scaler (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_shear_control.aircraft:wing:shear_control_mass_scaler)
  group_0.aircraft:wing:span (group_0.pre_mission.core_subsystems.core_geometry.prelim.aircraft:wing:span)
  group_0.aircraft:wing:span (group_0.pre_mission.core_subsystems.core_geometry.wing_prelim.aircraft:wing:span)
  group_0.aircraft:wing:span (group_0.pre_mission.core_subsystems.core_mass.anti_icing.aircraft:wing:span)
  group_0.aircraft:wing:span (group_0.pre_mission.core_subsystems.core_mass.fuel_capacity_group.wing_fuel_capacity.aircraft:wing:span)
  group_0.aircraft:wing:span (group_0.pre_mission.core_subsystems.core_mass.landing_group.main_landing_gear_length.aircraft:wing:span)
  group_0.aircraft:wing:span (group_0.pre_mission.core_subsystems.core_mass.wing_group.wing_bending.aircraft:wing:span)
"""
