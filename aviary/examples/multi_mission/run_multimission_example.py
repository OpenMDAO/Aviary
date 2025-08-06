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

Optimizer = 'SNOPT'  # SLSQP or SNOPT

prob = av.AviaryProblem(problem_type = ProblemType.MULTI_MISSION)

# set constraints in the background to allow Mission.Summary.GROSS_MASS to be acceptible as long as it's
# less than Mission.Design.GROSS_MASS. Also turns on Mission.Constraints.RANGE_RESIDUAL =0, forcong
# the mission to fly the target_range specified in the phase_info

prob.add_aviary_group('mission1', aircraft=aviary_inputs_primary, mission=phase_info)

# Load aircraft in second configuration for same mission
prob.add_aviary_group('mission2', aircraft=aviary_inputs_deadhead, mission=phase_info)

prob.build_model()

# Link Key design variables to ensure both aircraft are modelled the same:
prob.promote_inputs(['mission1', 'mission2'], [(Mission.Design.GROSS_MASS, 'Aircraft1:GROSS_MASS'), (Mission.Design.RANGE, 'Aircraft1:RANGE'), (Aircraft.Wing.SWEEP, 'Aircraft1:SWEEP')])

prob.add_design_var_default('Aircraft1:GROSS_MASS', lower=10.0, upper=900e3, units='lbm', default_val=100000)
prob.add_design_var_default('Aircraft1:SWEEP', lower=23.0, upper=27.0, units='deg', default_val=25)

prob.add_design_variables()

# Add objective
# Mission 1 is flown 2x more times than mission2
prob.add_composite_objective(('mission1', Mission.Summary.FUEL_BURNED, 2), ('mission2', Mission.Summary.FUEL_BURNED, 1))
# prob.add_composite_objective_adv(missions=['mission1', 'mission2'], mission_weights=[2,1], outputs=[Mission.Summary.FUEL_BURNED],  ref=1)

# optimizer and iteration limit are optional provided here
prob.add_driver(Optimizer, max_iter=50)

prob.setup()

# set_val goes here if needed
prob.set_initial_guesses()

# Ensure that design_range is the same for similar aircraft to ensure that navigation gear is designed similarly
prob.set_design_range(('mission1', 'mission2'), range='Aircraft1:RANGE')

# TODO: how to handle "aircraft that the user says are the same but are not the same i.e. wing design is different"

prob.run_aviary_problem()

# Add Asserts here
