"""
authors: Jatin Soni, Eliot Aretskin
Multi Mission Optimization Example using Aviary.

In this example, a monolithic optimization is created by instantiating two aviary problems
using typical AviaryProblem calls like load_inputs(), check_and_preprocess_payload(),
etc. Once those problems are setup and all of their phases are linked together, we copy
those problems as group into a super_problem. We then promote GROSS_MASS, RANGE, and
wing SWEEP from each of those sub-groups (group1 and group2) up to the super_probem so
the optimizer can control them. The fuel_burn results from each of the group1 and group2
dymos missions are summed and weighted to create the objective function the optimizer sees.

"""

import copy as copy
import sys
import warnings

import dymos as dm
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

import aviary.api as av
from aviary.examples.example_phase_info import phase_info
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings
from aviary.core.aviary_group import AviaryGroup

# fly the same mission twice with two different passenger loads
phase_info_primary = copy.deepcopy(phase_info)
phase_info_deadhead = copy.deepcopy(phase_info)
# get large single aisle values
aviary_inputs_primary = get_flops_inputs('LargeSingleAisle2FLOPS')
aviary_inputs_primary.set_val(Mission.Design.GROSS_MASS, val=100000, units='lbm')
aviary_inputs_primary.set_val(Settings.VERBOSITY, val=1)

aviary_inputs_deadhead = copy.deepcopy(aviary_inputs_primary)

# Due to current limitations in Aviary's ability to detect user input vs. default values,
# the only way to set an aircraft to zero passengers is by setting
# TOTAL_PAYLOAD_MASS = X CARGO_MASS + 0 PASSENGER_PAYLOAD_MASS.
# This zeros out passenger and baggage mass.
# Due to issue #610, setting PASSENGER_PAYLOAD_MASS = 0 will not work yet.
# aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 4077, 'lbm')

aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 1, 'unitless')
aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, 1, 'unitless')
aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 0, 'unitless')
aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 0, 'unitless')

Optimizer = 'SLSQP'  # SLSQP or SNOPT

prob = av.AviaryProblem(problem_type = ProblemType.MULTI_MISSION)

# set constraints in the background to allow Mission.Summary.GROSS_MASS to be acceptible as long as it's 
# less than Mission.Design.GROSS_MASS. Also turns on Mission.Constraints.RANGE_RESIDUAL =0, forcong
# the mission to fly the target_range specified in the phase_info

prob.add_aviary_group('mission1', aircraft=aviary_inputs_primary, mission=phase_info)

# Load aircraft in second configuration for same mission
prob.add_aviary_group('mission2', aircraft=aviary_inputs_deadhead, mission=phase_info)

prob.check_and_preprocess_inputs()

prob.build_model()

# Link Key design variables to ensure both aircraft are modelled the same:
prob.promote_inputs(['mission1', 'mission2'], [(Mission.Design.GROSS_MASS, 'Aircraft1:GROSS_MASS'), (Mission.Design.RANGE, 'Aircraft1:RANGE'), (Aircraft.Wing.SWEEP, 'Aircraft1:SWEEP')])

prob.add_design_var_default('Aircraft1:GROSS_MASS', lower=10.0, upper=900e3, units='lbm', default_val=100000)
prob.add_design_var_default('Aircraft1:SWEEP', lower=23.0, upper=27.0, units='deg', default_val=25)

# TODO: Do we have to run prob.add_design_variables() <- this adds some special stuff for multimission
prob.add_design_variables()

# Add objective
# Mission 1 is flown 2x more times than mission2
prob.add_composite_objective(('mission1', Mission.Summary.FUEL_BURNED, 2), ('mission2', Mission.Summary.FUEL_BURNED, 1))
# prob.add_composite_objective_adv(missions=['mission1', 'mission2'], mission_weights=[2,1], outputs=[Mission.Summary.FUEL_BURNED],  ref=1)

# optimizer and iteration limit are optional provided here
prob.add_driver(Optimizer, max_iter=50)

prob.setup()

# set_val goes here if needed

# Ensure that design_range is the same for similar aircraft to ensure that navigation gear is designed similarly
prob.set_design_range(('mission1', 'mission2'), range='Aircraft1:RANGE')

# TODO: how to handle "aircraft that the user says are the same but are not the same i.e. wing design is different"

prob.run_aviary_problem()




# class MultiMissionProblem(om.Problem):

#     def setup_wrapper(self):
#         """Wrapper for om.Problem setup with warning ignoring and setting options."""
#         for i, prob in enumerate(self.probs):
#             prob.model.options['aviary_options'] = prob.aviary_inputs
#             prob.model.options['aviary_metadata'] = prob.meta_data
#             prob.model.options['phase_info'] = prob.phase_info

#             # Use OpenMDAO's model options to pass all options through the system hierarchy.
#             prefix = self.group_prefix + f'_{i}'
#             setup_model_options(self, prob.aviary_inputs, prob.meta_data, prefix=f'{prefix}.')

#         # Aviary's problem setup wrapper uses these ignored warnings to suppress
#         # some warnings related to variable promotion. Replicating that here with
#         # setup for the super problem
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', om.OpenMDAOWarning)
#             warnings.simplefilter('ignore', om.PromotionWarning)
#             self.setup(check='all')

#     def run(self):
#         self.model.set_solver_print(0)
#         dm.run_problem(self, make_plots=False)

#     def create_timeseries_plots(self, plotvars=[], show=True):
#         """
#         Temporary create plots manually because graphing won't work for dual-trajectories.
#         Creates timeseries plots for any variables within timeseries. Specify variables
#         and units by setting plotvars = [('altitude','ft')]. Any number of vars can be added.
#         """
#         plt.figure()
#         for plotidx, (var, unit) in enumerate(plotvars):
#             plt.subplot(int(np.ceil(len(plotvars) / 2)), 2, plotidx + 1)
#             for i in range(self.num_missions):
#                 time = np.array([])
#                 yvar = np.array([])
#                 # this loop concatenates data from all phases
#                 for phase in self.phases[f'{self.group_prefix}_{i}']:
#                     rawt = self.get_val(
#                         f'{self.group_prefix}_{i}.traj.{phase}.timeseries.time', units='s'
#                     )
#                     rawy = self.get_val(
#                         f'{self.group_prefix}_{i}.traj.{phase}.timeseries.{var}', units=unit
#                     )
#                     time = np.hstack([time, np.ndarray.flatten(rawt)])
#                     yvar = np.hstack([yvar, np.ndarray.flatten(rawy)])
#                 plt.plot(time, yvar, linewidth=self.num_missions - i)
#             plt.xlabel('Time (s)')
#             plt.ylabel(f'{var.title()} ({unit})')
#             plt.grid()
#         plt.figlegend([f'Mission {i}' for i in range(self.num_missions)])
#         if show:
#             plt.show()

#     def print_vars(self, vars=[]):
#         """Specify vars with name and unit in a tuple, e.g. vars = [ (Mission.Summary.FUEL_BURNED, 'lbm') ]."""
#         print('\n\n=========================\n')
#         print(f'{"":40}', end=': ')
#         for i in range(self.num_missions):
#             name = f'Mission {i}'
#             print(f'{name:^30}', end='| ')
#         print()
#         for var, unit in vars:
#             varname = f'{var.replace(":", ".").upper()}'
#             print(f'{varname:40}', end=': ')
#             for i in range(self.num_missions):
#                 try:
#                     val = self.get_val(f'group_{i}.{var}', units=unit)[0]
#                     printstatement = f'{val} ({unit})'
#                 except:
#                     printstatement = f'unable get get_val({var})'
#                 print(f'{printstatement:^30}', end='| ')
#             print()


#     if show_plots:
#         printoutputs = [
#             (Mission.Design.GROSS_MASS, 'lbm'),
#             (Aircraft.Design.EMPTY_MASS, 'lbm'),
#             (Aircraft.Wing.SWEEP, 'deg'),
#             (Aircraft.LandingGear.MAIN_GEAR_MASS, 'lbm'),
#             (Aircraft.LandingGear.NOSE_GEAR_MASS, 'lbm'),
#             (Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 'unitless'),
#             (Aircraft.Furnishings.MASS, 'lbm'),
#             (Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 'lbm'),
#             (Mission.Summary.GROSS_MASS, 'lbm'),
#             (Mission.Summary.FUEL_BURNED, 'lbm'),
#             (Aircraft.CrewPayload.PASSENGER_MASS, 'lbm'),
#             (Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
#             (Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
#             (Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
#         ]
#         super_prob.print_vars(vars=printoutputs)

#         plotvars = [
#             ('altitude', 'ft'),
#             ('mass', 'lbm'),
#             ('drag', 'lbf'),
#             ('distance', 'nmi'),
#             ('throttle', 'unitless'),
#             ('mach', 'unitless'),
#         ]
#         super_prob.create_timeseries_plots(plotvars=plotvars, show=False)

#         plt.show()

#     return super_prob


# if __name__ == '__main__':
#     makeN2 = True if (len(sys.argv) > 1 and 'n2' in sys.argv[1]) else False

#     super_prob = large_single_aisle_example(makeN2=makeN2)

#     # Uncomment the following lines to see mass breakdown details for each mission.
#     # super_prob.model.group_1.list_vars(val=True, units=True, print_arrays=False)
#     # super_prob.model.group_2.list_vars(val=True, units=True, print_arrays=False)
