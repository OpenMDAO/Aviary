"""Run the a mission with a simple external component that computes the wing and horizontal tail mass."""

from copy import deepcopy

import aviary.api as av
from aviary.examples.external_subsystems.custom_mass.custom_mass_builder import WingMassBuilder

phase_info = deepcopy(av.default_height_energy_phase_info)
# Here we just add the simple weight system to only the pre-mission
phase_info['pre_mission']['external_subsystems'] = [WingMassBuilder()]

if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(
        'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv', phase_info
    )

    prob.check_and_preprocess_inputs()

    prob.build_model()

    prob.add_driver('SLSQP')

    prob.add_design_variables()

    prob.add_objective()

    prob.setup()

    prob.run_aviary_problem(suppress_solver_print=True)

    print('Engine Mass', prob.get_val(av.Aircraft.Engine.MASS))
    print('Wing Mass', prob.get_val(av.Aircraft.Wing.MASS))
    print('Horizontal Tail Mass', prob.get_val(av.Aircraft.HorizontalTail.MASS))
