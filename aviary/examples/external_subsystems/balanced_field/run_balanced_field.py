"""
Balanced Field problem as an external subsystem in post-mission.
"""
from copy import deepcopy

import numpy as np

import openmdao.api as om

from aviary.examples.external_subsystems.balanced_field.balanced_field_builder import BalancedFieldBuilder
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.advanced_single_aisle.phase_info import phase_info
from aviary.variable_info.variables import Mission

local_phase_info = deepcopy(phase_info)

#local_phase_info['post_mission']['external_subsystems'] = [BalancedFieldBuilder()]
local_phase_info['post_mission']['balanced_field'] = True

prob = AviaryProblem()

prob.load_inputs(
    'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
    local_phase_info,
)

# A few values that aren't in the csv file.
prob.aviary_inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, 577.0, units='lbm')
prob.aviary_inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG, 17.35, units='unitless')
prob.aviary_inputs.set_val(Mission.Design.THRUST_TAKEOFF_PER_ENG, 24555.5, units='lbf')

# initial guess for mass
prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 135000.0, units='lbm')

prob.check_and_preprocess_inputs()

prob.build_model()
prob.add_driver('SNOPT', max_iter=50, verbosity=1)

prob.add_design_variables()

prob.add_objective()

prob.setup()

# TODO: N3CC optimization does not return success.
prob.run_aviary_problem()

#prob.model.list_vars(units=True, print_arrays=True)
